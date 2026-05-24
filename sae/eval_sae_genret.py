"""
Evaluate SAE reconstruction quality on GenRet retrieval task.

Loads a trained GenRet model and a trained SAE, replaces decoder MLP activations
with SAE-reconstructed versions via forward hooks, and compares retrieval metrics.

Usage:
    python sae/eval_sae_genret.py \
        --model_name google-t5/t5-large \
        --genret_ckpt path/to/epoch.pt \
        --code_file path/to/epoch.pt.code \
        --sae_ckpt checkpoints/batchtopk_sae_t5_large_decoder/inference \
        --dev_data path/to/dev.json \
        --dev_seen path/to/dev.json.seen \
        --dev_unseen path/to/dev.json.unseen \
        --corpus_data path/to/corpus.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import Model, Tree, safe_load, BiDataset
from eval import eval_all
from sae.inference import load_sae, make_hf_replacement_hook


@torch.no_grad()
def run_eval(model, data_loader, tree, code_length, top_k=10):
    """Run constrained beam search and return predictions."""
    output_all = []
    for batch in data_loader:
        batch = {k: v.cuda() for k, v in batch.items() if v is not None}
        output = model.generate(
            input_ids=batch['query'].cuda(),
            attention_mask=batch['query'].ne(0).cuda(),
            max_length=code_length + 1,
            num_beams=top_k,
            num_return_sequences=top_k,
            prefix_allowed_tokens_fn=tree,
        )
        beam = []
        for line in output:
            if len(beam) >= top_k:
                output_all.append(beam)
                beam = []
            beam.append(line.cpu().tolist())
        output_all.append(beam)
    return output_all


def decode_predictions(output_all, corpus_ids, top_k=10):
    """Map generated code sequences to document indices."""
    docid_to_doc = defaultdict(list)
    for i, item in enumerate(corpus_ids):
        docid_to_doc[str(item)].append(i)

    predictions = []
    for line in output_all:
        new_line = []
        for s in line:
            s = str(s)
            if s not in docid_to_doc:
                continue
            tmp = docid_to_doc[s]
            new_line.extend(tmp)
            if len(new_line) > 100:
                break
        predictions.append(new_line)
    return predictions


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a readable string."""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE on GenRet")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-large")
    parser.add_argument("--genret_ckpt", type=str, required=True, help="Path to GenRet checkpoint (.pt)")
    parser.add_argument("--code_file", type=str, required=True, help="Path to code file (.pt.code)")
    parser.add_argument("--sae_ckpt", type=str, required=True, help="Path to SAE inference checkpoint dir")
    parser.add_argument("--dev_data", type=str, required=True, help="Path to dev data JSON")
    parser.add_argument("--dev_seen", type=str, required=True, help="Path to seen split indices")
    parser.add_argument("--dev_unseen", type=str, required=True, help="Path to unseen split indices")
    parser.add_argument("--corpus_data", type=str, required=True, help="Path to corpus JSON")
    parser.add_argument("--code_num", type=int, default=512)
    parser.add_argument("--code_length", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hook_layer", type=int, default=12, help="Decoder layer to hook (default: 12)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    data = json.load(open(args.dev_data))
    corpus = json.load(open(args.corpus_data))
    seen_split = json.load(open(args.dev_seen))
    unseen_split = json.load(open(args.dev_unseen))

    corpus_ids = [[0, *line] for line in json.load(open(args.code_file))]

    # ------------------------------------------------------------------
    # 2. Load GenRet model
    # ------------------------------------------------------------------
    print(f"Loading GenRet model from {args.genret_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = Model(
        model=t5, use_constraint=False,
        code_length=args.code_length, zero_inp=False,
        code_number=args.code_num,
    )
    safe_load(model, args.genret_ckpt)
    model = model.cuda().eval()

    # Build tree and data loader
    tree = Tree()
    tree.set_all(corpus_ids)

    dataset = BiDataset(
        data=data, corpus=corpus, tokenizer=tokenizer,
        max_doc_len=128, max_q_len=32, ids=corpus_ids, aux_ids=None,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn,
        batch_size=args.batch_size, shuffle=False, num_workers=8,
    )

    query_ids = [x[1] for x in data]

    # ------------------------------------------------------------------
    # 3. Baseline evaluation (no hook)
    # ------------------------------------------------------------------
    print("\n--- Baseline (no SAE) ---")
    with torch.no_grad():
        output_all = run_eval(model, data_loader, tree, args.code_length)
    predictions = decode_predictions(output_all, corpus_ids)
    baseline_metrics = eval_all(predictions, query_ids)
    print(f"  All:     {format_metrics(baseline_metrics)}")

    pred_seen = [predictions[j] for j in seen_split]
    pred_unseen = [predictions[j] for j in unseen_split]
    baseline_seen = eval_all(pred_seen, [query_ids[j] for j in seen_split])
    baseline_unseen = eval_all(pred_unseen, [query_ids[j] for j in unseen_split])
    print(f"  Seen:    {format_metrics(baseline_seen)}")
    print(f"  Unseen:  {format_metrics(baseline_unseen)}")

    # ------------------------------------------------------------------
    # 4. Load SAE and register hook
    # ------------------------------------------------------------------
    print(f"\nLoading SAE from {args.sae_ckpt}...")
    sae, meta = load_sae(args.sae_ckpt, device=device)
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, scaling_factor={meta['scaling_factor']:.4f}")

    hook_target = model.t5.decoder.block[args.hook_layer].layer[2].DenseReluDense
    hook_fn = make_hf_replacement_hook(sae, meta["scaling_factor"])
    hook = hook_target.register_forward_hook(hook_fn)
    print(f"  Hook registered on decoder.block[{args.hook_layer}].layer[2].DenseReluDense")

    # ------------------------------------------------------------------
    # 5. SAE replacement evaluation
    # ------------------------------------------------------------------
    print("\n--- SAE Replacement ---")
    with torch.no_grad():
        output_all_sae = run_eval(model, data_loader, tree, args.code_length)
    predictions_sae = decode_predictions(output_all_sae, corpus_ids)
    sae_metrics = eval_all(predictions_sae, query_ids)
    print(f"  All:     {format_metrics(sae_metrics)}")

    pred_sae_seen = [predictions_sae[j] for j in seen_split]
    pred_sae_unseen = [predictions_sae[j] for j in unseen_split]
    sae_seen = eval_all(pred_sae_seen, [query_ids[j] for j in seen_split])
    sae_unseen = eval_all(pred_sae_unseen, [query_ids[j] for j in unseen_split])
    print(f"  Seen:    {format_metrics(sae_seen)}")
    print(f"  Unseen:  {format_metrics(sae_unseen)}")

    hook.remove()

    # ------------------------------------------------------------------
    # 6. Comparison table
    # ------------------------------------------------------------------
    print("\n--- Comparison ---")
    all_keys = sorted(baseline_metrics.keys())
    header = f"{'Metric':<12} {'Baseline':>10} {'SAE':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for k in all_keys:
        b = baseline_metrics[k]
        s = sae_metrics[k]
        delta = s - b
        sign = "+" if delta >= 0 else ""
        print(f"{k:<12} {b:>10.4f} {s:>10.4f} {sign}{delta:>9.4f}")


if __name__ == "__main__":
    main()
