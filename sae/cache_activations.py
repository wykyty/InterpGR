"""
Pre-collect decoder MLP activations for ALL 24 layers and save to disk.

Uses teacher-forcing: decoder input = target docid shifted right (prepend PAD).

Usage:
    uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 12
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

    assert model.embed.W_E.shape[0] == target_vocab, \
        f"Embedding size mismatch: W_E={model.embed.W_E.shape[0]}, tokenizer={target_vocab}"

    model.to(device).eval()
    return model, tokenizer


def load_data(data_path: str, semantic_id_path: str) -> tuple[list[str], list[list[int]]]:
    """Load queries and their semantic docids.

    Returns (queries, semantic_ids) where each semantic_id is e.g. [8, 21, 24, 0].
    """
    data = json.load(open(data_path))
    sem_ids = json.load(open(semantic_id_path))

    queries, doc_ids = [], []
    for item in data:
        q, doc_id = item[0], item[1]
        if isinstance(q, list):
            q = q[0]
        if q and str(q).strip():
            queries.append(str(q))
            doc_ids.append(doc_id)

    semantic_ids = [sem_ids[doc_id] for doc_id in doc_ids]
    print(f"Loaded {len(queries)} queries with semantic IDs")
    return queries, semantic_ids


def make_decoder_input(semantic_id: list[int], tokenizer, device: str) -> torch.Tensor:
    """Convert semantic ID to decoder input (target shifted right, prepend PAD).

    e.g. [8, 21, 24, 0] → token string '$8$$21$$24$$0$' → tokens [32108, 32121, 32124, 32100]
    decoder input: [PAD, 32108, 32121, 32124] (shifted right)
    """
    target_str = ''.join([f'${i}$' for i in semantic_id])
    target_tokens = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids
    # Shift right: prepend PAD token, drop last token
    pad_token = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
    dec_input = torch.cat([pad_token, target_tokens[:, :-1]], dim=1)
    return dec_input.to(device)


def collect_layers(
    checkpoint_path: str,
    queries: list[str],
    semantic_ids: list[list[int]],
    device: str,
    context_size: int,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Collect activations for specified layers on a single GPU.

    Returns {layer_idx: activations_tensor}.
    """
    model, tokenizer = load_dsi_model(checkpoint_path, device)
    hook_names = [f"decoder.{l}.hook_mlp_out" for l in layers]

    all_acts = {l: [] for l in layers}

    for query, sem_id in tqdm(zip(queries, semantic_ids), total=len(queries),
                              desc=f"Collecting on {device}"):
        enc_tokens = tokenizer(
            query, return_tensors="pt", truncation=True,
            max_length=context_size, padding=False,
        )
        dec_input = make_decoder_input(sem_id, tokenizer, device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                enc_tokens.input_ids,
                one_zero_attention_mask=enc_tokens.attention_mask,
                decoder_input=dec_input,
                names_filter=lambda name: name in hook_names,
            )

        for l in layers:
            acts = cache[f"decoder.{l}.hook_mlp_out"].squeeze(0).float().cpu()
            all_acts[l].append(acts)

    return {l: torch.cat(acts, dim=0) for l, acts in all_acts.items()}


def main():
    parser = argparse.ArgumentParser(description="Pre-collect T5 decoder MLP activations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--semantic_id_path", type=str, default="dataset/nq320k_id/id.semantic.bert.json")
    parser.add_argument("--cache_dir", type=str, default="data/activation_cache")
    parser.add_argument("--layer", type=int, nargs="+", default=None,
                        help="Layer(s) to cache (default: all 0-23)")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--context_size", type=int, default=32)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    layers = args.layer if args.layer else list(range(N_LAYERS))

    queries, semantic_ids = load_data(args.data_path, args.semantic_id_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caching activations for {len(queries)} queries, layers {layers}")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Semantic IDs:   {args.semantic_id_path}")
    print(f"  Cache dir:      {cache_dir}")

    if args.n_gpus > 1:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        print(f"  GPUs:           {args.n_gpus} (multiprocessing)")
        raise NotImplementedError("Multi-GPU not yet implemented. Use --n_gpus 1.")
    else:
        print(f"  GPU:            single (cuda:0)")
        layer_data = collect_layers(
            args.checkpoint, queries, semantic_ids, "cuda:0", args.context_size, layers,
        )

    # Save per layer
    print("Saving...")
    for l in tqdm(layer_data.keys(), desc="Saving"):
        save_file({"activations": layer_data[l]}, str(cache_dir / f"layer_{l}.safetensors"))
        print(f"  Layer {l}: {layer_data[l].shape}")

    meta = {
        "n_queries": len(queries),
        "layers": layers,
        "context_size": args.context_size,
        "checkpoint": args.checkpoint,
        "semantic_id_path": args.semantic_id_path,
        "d_model": 1024,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Cached {N_LAYERS} layers to {cache_dir}")
    print(f"Now train with:")
    print(f"  python sae/train_sae.py --cache_dir {cache_dir} --layer 0")


if __name__ == "__main__":
    main()
