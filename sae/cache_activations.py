"""
Pre-collect decoder MLP activations for ALL 24 layers and save to disk.

Run this once, then train any layer with --cache_dir to skip T5 forward passes.

Usage:
    uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/dev.json \
        --cache_dir data/activation_cache \
        --n_gpus 1
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

N_LAYERS = 24


def load_dsi_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """Load a DSI semantic checkpoint into HookedEncoderDecoder."""
    import transformer_lens.loading_from_pretrained as loading
    from transformer_lens import HookedEncoderDecoder
    from transformer_lens.pretrained.weight_conversions.t5 import convert_t5_weights

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
    tokenizer.add_tokens([f'${i}$' for i in range(30)])
    target_vocab = len(tokenizer)

    hf_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")
    hf_model.resize_token_embeddings(target_vocab)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.eval()

    cfg = loading.get_pretrained_model_config(
        "google-t5/t5-large", fold_ln=False, device=None, n_devices=1,
    )
    cfg.d_vocab = target_vocab
    cfg.d_vocab_out = target_vocab
    tl_state_dict = convert_t5_weights(hf_model, cfg)

    model = HookedEncoderDecoder(cfg, tokenizer=tokenizer, move_to_device=False)
    model.load_state_dict(tl_state_dict, strict=False)

    # Verify embedding size matches tokenizer
    assert model.embed.W_E.shape[0] == target_vocab, \
        f"Embedding size mismatch: W_E={model.embed.W_E.shape[0]}, tokenizer={target_vocab}"

    model.to(device).eval()
    return model, tokenizer


def load_queries(data_path: str) -> list[str]:
    data = json.load(open(data_path))
    queries = []
    for item in data:
        q = item[0]
        if isinstance(q, list):
            q = q[0]
        if q and str(q).strip():
            queries.append(str(q))
    return queries


def collect_all_layers(
    checkpoint_path: str,
    queries: list[str],
    device: str,
    context_size: int,
) -> list[torch.Tensor]:
    """Collect activations for all 24 layers on a single GPU. Returns list of [n_tokens, d_in]."""
    model, tokenizer = load_dsi_model(checkpoint_path, device)
    hook_names = [f"decoder.{l}.hook_mlp_out" for l in range(N_LAYERS)]

    all_acts = [[] for _ in range(N_LAYERS)]

    for query in tqdm(queries, desc=f"Collecting on {device}"):
        enc_tokens = tokenizer(
            query, return_tensors="pt", truncation=True,
            max_length=context_size, padding=False,
        )
        dec_input = torch.full(
            (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device,
        )

        with torch.no_grad():
            _, cache = model.run_with_cache(
                enc_tokens.input_ids,
                one_zero_attention_mask=enc_tokens.attention_mask,
                decoder_input=dec_input,
                names_filter=lambda name: name in hook_names,
            )

        for l in range(N_LAYERS):
            acts = cache[hook_names[l]].squeeze(0).float().cpu()
            all_acts[l].append(acts)

    return [torch.cat(layer_acts, dim=0) for layer_acts in all_acts]


def main():
    parser = argparse.ArgumentParser(description="Pre-collect T5 decoder MLP activations for all layers")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="data/activation_cache")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--context_size", type=int, default=32)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    queries = load_queries(args.data_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching activations for {len(queries)} queries, {N_LAYERS} layers")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Cache dir:  {cache_dir}")

    if args.n_gpus > 1:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        print(f"  GPUs:       {args.n_gpus} (multiprocessing)")
    else:
        print(f"  GPU:        single (cuda:0)")
        layer_data = collect_all_layers(
            args.checkpoint, queries, "cuda:0", args.context_size,
        )

    # Save per layer
    print("Saving...")
    for l in tqdm(range(N_LAYERS), desc="Saving"):
        save_file({"activations": layer_data[l]}, str(cache_dir / f"layer_{l}.safetensors"))
        print(f"  Layer {l}: {layer_data[l].shape}")

    meta = {
        "n_queries": len(queries),
        "n_layers": N_LAYERS,
        "context_size": args.context_size,
        "checkpoint": args.checkpoint,
        "d_model": 1024,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Cached {N_LAYERS} layers to {cache_dir}")
    print(f"Now train with:")
    print(f"  python sae/train_semantic.py --cache_dir {cache_dir} --layers 0")


if __name__ == "__main__":
    main()
