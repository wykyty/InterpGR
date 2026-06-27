"""
SAE Inference Utilities

Supports two SAE checkpoint formats:
  - Training checkpoint (sae_config.json + sae_weights.safetensors): BatchTopKTrainingSAE
  - Inference checkpoint (cfg.json + sae_weights.safetensors): JumpReLUSAE

Two activation sources:
  1. Cached activations from disk (data/activation_cache_*/layer_X.safetensors)
  2. Live collection from DSI model (query/docid -> model -> decoder activations)

Usage:
    # Analyze cached activations
    uv run python sae/inference.py analyze \
        --checkpoint_dir out/sae_train_4x/layer_12 \
        --cache_dir data/activation_cache_train

    # Collect activations from model, get SAE latents
    uv run python sae/inference.py collect \
        --dsi_checkpoint out/dsi-semantic-bert/99.pt \
        --checkpoint_dir out/sae_train_4x/layer_12 \
        --data_path dataset/nq320k/train.json \
        --semantic_id_path dataset/nq320k_id/id.semantic.bert.json

    # Run model with SAE hook intervention
    uv run python sae/inference.py intervene \
        --dsi_checkpoint out/dsi-semantic-bert/99.pt \
        --checkpoint_dir out/sae_train_4x/layer_12 \
        --query "What is machine learning?"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------

def load_sae(checkpoint_path: Path, device: str):
    """Load BatchTopKTrainingSAE from training checkpoint."""
    from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig

    with open(checkpoint_path / "sae_config.json", "r") as f:
        cfg = json.load(f)

    d_in = cfg["d_in"]
    d_sae = cfg["d_sae"]
    k = cfg["k"]

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        aux_loss_coefficient=1.0,
        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.1,
        apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in",
        decoder_init_norm=0.01,
        device=device,
        dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    weights = load_file(str(checkpoint_path / "sae_weights.safetensors"))
    sae.W_enc.data = weights["W_enc"].to(device)
    sae.W_dec.data = weights["W_dec"].to(device)
    sae.b_enc.data = weights["b_enc"].to(device)
    sae.b_dec.data = weights["b_dec"].to(device)
    sae.topk_threshold = weights["topk_threshold"].to(device)
    sae.eval()

    print(f"Loaded BatchTopKTrainingSAE: d_in={d_in}, d_sae={d_sae}, k={k}")
    return sae, "batchtopk", cfg

# ---------------------------------------------------------------------------
# DSI model loading
# ---------------------------------------------------------------------------


def load_dsi_model(
    checkpoint_path: str,
    model_name: str = "google-t5/t5-large",
    device: str = "cuda",
):
    """Load a DSI checkpoint into HookedEncoderDecoder.

    Returns:
        (model, tokenizer)
    """
    import transformer_lens.loading_from_pretrained as loading
    from transformer_lens import HookedEncoderDecoder
    from transformer_lens.pretrained.weight_conversions.t5 import convert_t5_weights

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens([f"${i}$" for i in range(30)])
    target_vocab = len(tokenizer)

    hf_model = T5ForConditionalGeneration.from_pretrained(model_name)
    hf_model.resize_token_embeddings(target_vocab)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.eval()

    cfg = loading.get_pretrained_model_config(
        model_name, fold_ln=False, device=None, n_devices=1,
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_path: str, semantic_id_path: str):
    """Load queries and their semantic docids.

    Returns:
        (queries, semantic_ids) where each semantic_id is e.g. [8, 21, 24, 0].
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

    e.g. [8, 21, 24, 0] -> token string '$8$$21$$24$$0$' -> tokens [32108, 32121, 32124, 32100]
    decoder input: [PAD, 32108, 32121, 32124] (shifted right)
    """
    target_str = "".join([f"${i}$" for i in semantic_id])
    target_tokens = tokenizer(target_str, return_tensors="pt", add_special_tokens=False).input_ids
    pad_token = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long)
    dec_input = torch.cat([pad_token, target_tokens[:, :-1]], dim=1)
    return dec_input.to(device)


class ActivationDataset(Dataset):
    """Dataset for loading cached activations from safetensors."""
    def __init__(self, cache_dir: str, layer: int):
        cache_path = Path(cache_dir) / f"layer_{layer}.safetensors"
        print(f"Loading cached activations from {cache_path}...")
        data = load_file(str(cache_path))
        self.activations = data["activations"]  # (total_tokens, d_in)
        self.total_tokens = self.activations.shape[0]
        self.d_in = self.activations.shape[1]
        print(f"Loaded {self.total_tokens} tokens, d_in={self.d_in}")

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        return self.activations[idx]


# ---------------------------------------------------------------------------
# Activation collection from model
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_activations_from_model(
    model,
    tokenizer,
    queries: list[str],
    semantic_ids: list[list[int]],
    hook_name: str,
    context_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Collect decoder activations using teacher-forcing (query -> decoder with semantic_id).

    Args:
        model: HookedEncoderDecoder model.
        tokenizer: Tokenizer with custom tokens.
        queries: List of query strings.
        semantic_ids: List of semantic ID lists, e.g. [[8, 21, 24, 0], ...].
        hook_name: Hook point name, e.g. "decoder.12.hook_mlp_out".
        context_size: Max encoder context length.
        device: Device.

    Returns:
        Activations tensor, shape [total_tokens, d_model].
    """
    collected = []

    for query, sem_id in tqdm(zip(queries, semantic_ids), total=len(queries),
                              desc="Collecting activations"):
        enc_tokens = tokenizer(
            query, return_tensors="pt", truncation=True,
            max_length=context_size, padding=False,
        )
        dec_input = make_decoder_input(sem_id, tokenizer, device)

        _, cache = model.run_with_cache(
            enc_tokens.input_ids,
            one_zero_attention_mask=enc_tokens.attention_mask,
            decoder_input=dec_input,
            names_filter=lambda name: name == hook_name,
        )
        acts = cache[hook_name].squeeze(0).float().cpu()
        collected.append(acts)

    return torch.cat(collected, dim=0)


@torch.no_grad()
def collect_activations_from_text(
    model,
    tokenizer,
    texts: list[str],
    hook_name: str,
    context_size: int = 128,
    target_size: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    """Collect decoder activations from plain text (text as both encoder and decoder input).

    For quick exploration, not for DSI evaluation.
    """
    collected = []

    for text in texts:
        enc_tokens = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=context_size, padding=False,
        )
        dec_tokens = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=target_size, padding=False,
        )
        _, cache = model.run_with_cache(
            enc_tokens.input_ids,
            one_zero_attention_mask=enc_tokens.attention_mask,
            decoder_input=dec_tokens.input_ids,
            names_filter=lambda name: name == hook_name,
        )
        acts = cache[hook_name].squeeze(0).float().cpu()
        collected.append(acts)

    return torch.cat(collected, dim=0)


# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_activations(sae, activations: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Encode raw activations through SAE, returns feature activations.

    Args:
        sae: Loaded SAE (BatchTopKTrainingSAE or JumpReLUSAE).
        activations: Raw decoder activations, shape [n_tokens, d_in].

    Returns:
        Feature activations, shape [n_tokens, d_sae].
    """
    batch = activations.to(device)
    return sae.encode(batch)


@torch.no_grad()
def encode_activations_batched(
    sae,
    activations: torch.Tensor,
    batch_size: int = 4096,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode activations through SAE in batches to save memory.

    Returns:
        Feature activations, shape [n_tokens, d_sae].
    """
    results = []
    for i in range(0, activations.shape[0], batch_size):
        batch = activations[i : i + batch_size].to(device)
        with torch.no_grad():
            feats = sae.encode(batch)
        results.append(feats.cpu())
    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_features(
    sae,
    dataloader: DataLoader,
    device: str = "cuda",
    eval_batches: int = None,
) -> dict:
    """Run reconstruction quality analysis on cached activations.

    Returns:
        Dict with explained_variance, l0, dead_features, etc.
    """
    sae.eval()
    d_sae = sae.cfg.d_sae

    total_mse = 0.0
    total_variance = 0.0
    total_l0 = 0.0
    total_tokens = 0
    feature_activated_counts = torch.zeros(d_sae, device=device)

    # First pass: compute global mean
    print("Computing global mean...")
    global_mean = torch.zeros(dataloader.dataset.d_in, device=device)
    total_for_mean = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing mean")):
        if eval_batches and batch_idx >= eval_batches:
            break
        batch = batch.to(device)
        global_mean += batch.sum(dim=0)
        total_for_mean += batch.shape[0]
    global_mean /= total_for_mean

    # Second pass: evaluate
    print("Evaluating reconstruction...")
    # Reset dataloader by creating a new one
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if eval_batches and batch_idx >= eval_batches:
            break

        batch = batch.to(device)
        feature_acts = sae.encode(batch)
        reconstructions = sae.decode(feature_acts)

        per_token_mse = (reconstructions - batch).pow(2).sum(dim=-1)
        per_token_var = (batch - global_mean).pow(2).sum(dim=-1)

        total_mse += per_token_mse.sum().item()
        total_variance += per_token_var.sum().item()

        active_mask = feature_acts > 0
        total_l0 += active_mask.sum(-1).float().sum().item()
        feature_activated_counts += active_mask.float().sum(dim=0)
        total_tokens += batch.shape[0]

    if total_tokens == 0:
        print("  No tokens evaluated.")
        return {}

    final_ev = 1 - (total_mse / (total_variance + 1e-8))
    final_l0 = total_l0 / total_tokens
    dead_features = (feature_activated_counts == 0).sum().item()

    results = {
        "tokens": total_tokens,
        "explained_variance": final_ev,
        "l0": final_l0,
        "dead_features": dead_features,
        "dead_ratio": dead_features / d_sae,
    }

    print("\n" + "=" * 50)
    print("[ANALYSIS RESULTS]")
    print(f"  Tokens evaluated         : {total_tokens}")
    print(f"  Explained Variance (EV)  : {final_ev:.4f}")
    print(f"  L0 (avg active features) : {final_l0:.1f}")
    print(f"  Dead Features            : {dead_features}/{d_sae} ({dead_features/d_sae*100:.2f}%)")
    print("=" * 50)

    return results


@torch.no_grad()
def extract_latent_concepts(
    sae,
    dataloader: DataLoader,
    device: str = "cuda",
    threshold: float = 0.01,
    max_batches: int = None,
) -> list[dict]:
    """Extract SAE latent concepts (active features) for each token.

    Returns:
        List of dicts with 'token_id', 'ids' (active feature indices), 'weight' (activation values).
    """
    sae.eval()
    all_results = []
    token_id = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
        if max_batches and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        feature_acts = sae.encode(batch)

        for i in range(batch.shape[0]):
            acts = feature_acts[i]
            active_mask = acts > threshold
            active_indices = torch.where(active_mask)[0]
            active_weights = acts[active_mask]

            all_results.append({
                "token_id": token_id,
                "ids": active_indices.cpu().tolist(),
                "weight": active_weights.cpu().tolist(),
            })
            token_id += 1

    return all_results


# ---------------------------------------------------------------------------
# Hook functions for intervention
# ---------------------------------------------------------------------------


def make_sae_hook(sae) -> Callable:
    """Create forward hook: replace decoder MLP output with SAE reconstruction.

    Weights are assumed to be folded (no scaling_factor needed).
    Compatible with HuggingFace T5 decoder MLP:
        model.decoder.block[layer].layer[2].DenseReluDense.register_hook(hook)
    """
    def hook_fn(module, input, output):
        orig_shape = output.shape
        flat = output.reshape(-1, orig_shape[-1])
        features = sae.encode(flat)
        reconstructed = sae.decode(features)
        return reconstructed.reshape(orig_shape)
    return hook_fn


def make_ablation_hook(
    sae,
    feature_indices: list[int],
) -> Callable:
    """Create hook that ablates (zeros out) specific SAE features."""
    def hook_fn(module, input, output):
        orig_shape = output.shape
        flat = output.reshape(-1, orig_shape[-1])
        features = sae.encode(flat)
        features[:, feature_indices] = 0.0
        reconstructed = sae.decode(features)
        return reconstructed.reshape(orig_shape)
    return hook_fn


def make_patching_hook(
    sae,
    feature_indices: list[int] | None = None,
    mode: str = "keep_only",
) -> Callable:
    """Create hook that patches activations via SAE decomposition.

    Args:
        feature_indices: Features to keep/remove.
        mode: "keep_only" keeps only specified features, "remove" removes them.
    """
    def hook_fn(module, input, output):
        orig_shape = output.shape
        flat = output.reshape(-1, orig_shape[-1])
        features = sae.encode(flat)

        if feature_indices is not None:
            if mode == "keep_only":
                mask = torch.zeros_like(features)
                mask[:, feature_indices] = 1.0
                features = features * mask
            elif mode == "remove":
                features[:, feature_indices] = 0.0

        reconstructed = sae.decode(features)
        return reconstructed.reshape(orig_shape)
    return hook_fn


def make_tl_sae_hook(
    sae,
    feature_indices: list[int] | None = None,
    mode: str = "replace",
) -> Callable:
    """Create hook for TransformerLens HookedEncoderDecoder.

    Args:
        feature_indices: Features to ablate (mode="ablate") or keep (mode="keep_only").
        mode: "replace" (full SAE reconstruction), "ablate" (zero out features), "keep_only".
    """
    from transformer_lens.hook_points import HookPoint

    def hook_fn(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        features = sae.encode(activation)

        if feature_indices is not None:
            if mode == "ablate":
                features[:, :, feature_indices] = 0.0
            elif mode == "keep_only":
                mask = torch.zeros_like(features)
                mask[:, :, feature_indices] = 1.0
                features = features * mask

        return sae.decode(features)

    return hook_fn


# ---------------------------------------------------------------------------
# Model intervention
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_with_intervention(
    query: str,
    semantic_id: list[int],
    model,
    tokenizer,
    hook_name: str,
    hook_fn: Callable,
    max_new_tokens: int = 15,
    context_size: int = 32,
    device: str = "cuda",
) -> tuple[str, str]:
    """Run DSI model with and without hook intervention, compare outputs.

    Args:
        query: Input query string.
        semantic_id: Semantic ID for decoder input, e.g. [8, 21, 24, 0].
        model: HookedEncoderDecoder model.
        tokenizer: Tokenizer with custom tokens.
        hook_name: Hook point name.
        hook_fn: TransformerLens hook function.
        max_new_tokens: Max tokens to generate.
        context_size: Max encoder context length.

    Returns:
        (original_output, intervened_output) as decoded strings.
    """
    enc_tokens = tokenizer(
        query, return_tensors="pt", truncation=True,
        max_length=context_size, padding=False,
    )

    # Without intervention
    with model.hooks(fwd_hooks=[]):
        output_orig = model.generate(
            enc_tokens.input_ids,
            attention_mask=enc_tokens.attention_mask,
            max_new_tokens=max_new_tokens,
        )

    # With intervention
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output_intervened = model.generate(
            enc_tokens.input_ids,
            attention_mask=enc_tokens.attention_mask,
            max_new_tokens=max_new_tokens,
        )

    text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)
    text_intervened = tokenizer.decode(output_intervened[0], skip_special_tokens=True)

    return text_orig, text_intervened


# ---------------------------------------------------------------------------
# Logit Lens
# ---------------------------------------------------------------------------


@torch.no_grad()
def logit_lens(
    sae,
    model,
    n_features: int = 20,
    top_k: int = 5,
    feature_indices: list[int] | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """Map SAE features to vocabulary tokens via decoder weight projection.

    Args:
        sae: Loaded SAE.
        model: T5 model (for embedding matrix and tokenizer).
        n_features: Number of random features to analyze (ignored if feature_indices given).
        top_k: Number of top tokens per feature.
        feature_indices: Specific feature indices to analyze.

    Returns:
        Dict mapping feature index -> list of (token_string, score) tuples.
    """
    embed = model.W_dec  # T5 decoder embedding, shape [d_vocab, d_model]
    projection = sae.W_dec @ embed.T  # [d_sae, d_vocab]

    if feature_indices is None:
        feature_indices = torch.randint(0, projection.shape[0], (n_features,)).tolist()

    results = {}
    vals, inds = torch.topk(projection, top_k, dim=1)
    for idx in feature_indices:
        tokens = [model.to_string(i) for i in inds[idx]]
        scores = vals[idx].tolist()
        results[idx] = list(zip(tokens, scores))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_analyze(args):
    """Analyze cached activations with SAE."""
    device = args.device

    # Load SAE
    sae, sae_type, cfg = load_sae(args.checkpoint_dir, device)

    # Determine layer
    layer = cfg.get("layer")
    if layer is None:
        layer = int(Path(args.checkpoint_dir).parent.name.split("_")[-1])

    # Load cached activations
    dataset = ActivationDataset(args.cache_dir, layer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Analyze
    results = analyze_features(sae, dataloader, device, eval_batches=args.eval_batches)

    # Optionally extract latent concepts
    if args.extract_concepts:
        print("\nExtracting latent concepts...")
        concepts = extract_latent_concepts(
            sae, dataloader, device,
            threshold=args.threshold,
            max_batches=args.eval_batches,
        )
        output_path = args.output or f"results/latent_concepts/layer_{layer}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for c in concepts:
                f.write(json.dumps(c) + "\n")
        print(f"Saved {len(concepts)} concepts to {output_path}")


def cmd_collect(args):
    """Collect activations from DSI model and encode through SAE."""
    device = args.device

    # Load DSI model
    print(f"Loading DSI model from {args.dsi_checkpoint}...")
    model, tokenizer = load_dsi_model(args.dsi_checkpoint, device=device)

    # Load SAE
    sae, sae_type, cfg = load_sae(args.checkpoint_dir, device)
    layer = cfg.get("layer")
    if layer is None:
        layer = int(Path(args.checkpoint_dir).parent.name.split("_")[-1])
    hook_name = f"decoder.{layer}.hook_mlp_out"

    # Load data
    queries, semantic_ids = load_data(args.data_path, args.semantic_id_path)

    # Limit data
    if args.max_samples:
        queries = queries[:args.max_samples]
        semantic_ids = semantic_ids[:args.max_samples]

    # Collect activations
    print(f"Collecting activations for {len(queries)} samples, layer {layer}...")
    activations = collect_activations_from_model(
        model, tokenizer, queries, semantic_ids,
        hook_name, context_size=args.context_size, device=device,
    )
    print(f"Collected activations: {activations.shape}")

    # Encode through SAE
    print("Encoding through SAE...")
    feature_acts = encode_activations_batched(
        sae, activations, batch_size=args.batch_size, device=device,
    )
    print(f"SAE latents: {feature_acts.shape}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({
            "activations": activations,
            "feature_acts": feature_acts,
            "queries": queries,
            "semantic_ids": semantic_ids,
            "layer": layer,
            "hook_name": hook_name,
        }, args.output)
        print(f"Saved to {args.output}")

    # Print summary
    active_mask = feature_acts > 0
    l0 = active_mask.sum(-1).float().mean().item()
    print(f"\nL0 (avg active features): {l0:.1f}")
    print(f"Total non-zero features: {(active_mask.any(0)).sum().item()}")


def cmd_intervene(args):
    """Run model with SAE hook intervention."""
    device = args.device

    # Load DSI model
    print(f"Loading DSI model from {args.dsi_checkpoint}...")
    model, tokenizer = load_dsi_model(args.dsi_checkpoint, device=device)

    # Load SAE
    sae, sae_type, cfg = load_sae(args.checkpoint_dir, device)
    layer = cfg.get("layer")
    if layer is None:
        layer = int(Path(args.checkpoint_dir).parent.name.split("_")[-1])
    hook_name = f"decoder.{layer}.hook_mlp_out"

    # Use query from args or default
    query = args.query or "What is machine learning?"
    print(f"\nQuery: {query}")
    print(f"Layer: {layer}, Hook: {hook_name}")

    # Full SAE reconstruction
    print("\n--- Full SAE Reconstruction ---")
    recon_hook = make_tl_sae_hook(sae, mode="replace")
    orig, reconstructed = run_with_intervention(
        query, [], model, tokenizer, hook_name, recon_hook,
        max_new_tokens=args.max_new_tokens, context_size=args.context_size, device=device,
    )
    print(f"  Original:      {orig}")
    print(f"  Reconstructed: {reconstructed}")

    # Ablate top features
    if args.ablate_top_k:
        print(f"\n--- Ablate Top {args.ablate_top_k} Features ---")
        # Get activations for this query to find top features
        acts = collect_activations_from_text(
            model, tokenizer, [query], hook_name,
            context_size=args.context_size, device=device,
        )
        feats = encode_activations(sae, acts, device)
        top_features = feats.sum(dim=0).topk(args.ablate_top_k).indices.tolist()
        print(f"  Top features: {top_features}")

        ablate_hook = make_tl_sae_hook(sae, feature_indices=top_features, mode="ablate")
        orig, ablated = run_with_intervention(
            query, [], model, tokenizer, hook_name, ablate_hook,
            max_new_tokens=args.max_new_tokens, context_size=args.context_size, device=device,
        )
        print(f"  Original: {orig}")
        print(f"  Ablated:  {ablated}")


def main():
    parser = argparse.ArgumentParser(description="SAE Inference Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze cached activations with SAE")
    p_analyze.add_argument("--checkpoint_dir", type=str, required=True)
    p_analyze.add_argument("--cache_dir", type=str, required=True)
    p_analyze.add_argument("--batch_size", type=int, default=4096)
    p_analyze.add_argument("--eval_batches", type=int, default=None)
    p_analyze.add_argument("--extract_concepts", action="store_true", help="Extract and save latent concepts")
    p_analyze.add_argument("--threshold", type=float, default=0.01)
    p_analyze.add_argument("--output", type=str, default=None)
    p_analyze.add_argument("--device", type=str, default="cuda")

    # --- collect ---
    p_collect = subparsers.add_parser("collect", help="Collect activations from DSI model and encode through SAE")
    p_collect.add_argument("--dsi_checkpoint", type=str, required=True)
    p_collect.add_argument("--checkpoint_dir", type=str, required=True)
    p_collect.add_argument("--data_path", type=str, required=True)
    p_collect.add_argument("--semantic_id_path", type=str, default="dataset/nq320k_id/id.semantic.bert.json")
    p_collect.add_argument("--context_size", type=int, default=32)
    p_collect.add_argument("--batch_size", type=int, default=4096)
    p_collect.add_argument("--max_samples", type=int, default=None)
    p_collect.add_argument("--output", type=str, default=None)
    p_collect.add_argument("--device", type=str, default="cuda")

    # --- intervene ---
    p_intervene = subparsers.add_parser("intervene", help="Run model with SAE hook intervention")
    p_intervene.add_argument("--dsi_checkpoint", type=str, required=True)
    p_intervene.add_argument("--checkpoint_dir", type=str, required=True)
    p_intervene.add_argument("--query", type=str, default=None)
    p_intervene.add_argument("--max_new_tokens", type=int, default=15)
    p_intervene.add_argument("--context_size", type=int, default=32)
    p_intervene.add_argument("--ablate_top_k", type=int, default=None, help="Ablate top K active features")
    p_intervene.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "intervene":
        cmd_intervene(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
