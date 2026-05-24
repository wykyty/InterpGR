"""
BatchTopK SAE Inference on T5-large Decoder

Loads a trained BatchTopK SAE (converted to JumpReLU for inference) and provides
utilities for encoding activations, analyzing features, and running logit lens.

Usage:
    python inference.py --checkpoint checkpoints/batchtopk_sae_t5_large_decoder/inference
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file
from transformer_lens import HookedEncoderDecoder
from transformer_lens.hook_points import HookPoint

from sae_lens.saes.sae import SAE


def load_sae(ckpt_dir: str, device: str = "cuda") -> tuple[SAE, dict]:
    """Load inference SAE and metadata from checkpoint directory."""
    ckpt_dir = Path(ckpt_dir)

    sae = SAE.load_from_disk(str(ckpt_dir), device=device)

    # scaling_factor must be applied before encoding
    scaling_factor = 1.0
    sf_path = ckpt_dir / "scaling_factor.safetensors"
    if sf_path.exists():
        scaling_factor = load_file(str(sf_path))["scaling_factor"].item()

    # sparsity info
    sparsity = None
    sp_path = ckpt_dir / "sparsity.safetensors"
    if sp_path.exists():
        sparsity = load_file(str(sp_path))["sparsity"]

    meta = {
        "scaling_factor": scaling_factor,
        "sparsity": sparsity,
    }
    return sae, meta


@torch.no_grad()
def collect_activations(
    texts: list[str],
    model: HookedEncoderDecoder,
    hook_name: str,
    context_size: int = 128,
    target_size: int = 64,
) -> torch.Tensor:
    """Collect decoder activations from encoder-decoder model.

    For simplicity, uses each text as both encoder and decoder input.
    For summarization tasks, pass (source, summary) pairs and adjust accordingly.
    """
    tokenizer = model.tokenizer
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
            attention_mask=enc_tokens.attention_mask,
            decoder_input=dec_tokens.input_ids,
            names_filter=lambda name: name == hook_name,
        )
        acts = cache[hook_name].squeeze(0).float()
        collected.append(acts)

    return torch.cat(collected, dim=0)


@torch.no_grad()
def get_feature_activations(
    sae: SAE,
    activations: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """Encode activations through SAE, returns feature activations.

    Args:
        sae: Loaded JumpReLU SAE.
        activations: Raw decoder activations, shape [n_tokens, d_in].
        scaling_factor: From training, must match what was used during training.

    Returns:
        Feature activations, shape [n_tokens, d_sae].
    """
    scaled = activations * scaling_factor
    return sae.encode(scaled)


@torch.no_grad()
def logit_lens(
    sae: SAE,
    model: HookedEncoderDecoder,
    n_features: int = 20,
    top_k: int = 5,
    feature_indices: list[int] | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """Map SAE features to vocabulary tokens via decoder weight projection.

    Args:
        sae: Loaded JumpReLU SAE.
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


@torch.no_grad()
def analyze_features(
    sae: SAE,
    activations: torch.Tensor,
    scaling_factor: float,
    sparsity: torch.Tensor | None = None,
) -> dict:
    """Run full analysis on a set of activations.

    Returns explained variance, L0, dead features, and per-feature stats.
    """
    scaled = activations * scaling_factor
    feature_acts = sae.encode(scaled)
    reconstruction = sae.decode(feature_acts)

    # Reconstruction quality
    per_token_mse = (reconstruction - scaled).pow(2).sum(dim=-1)
    total_var = (scaled - scaled.mean(0)).pow(2).sum(dim=-1)
    explained_variance = 1 - per_token_mse.mean() / (total_var.mean() + 1e-8)

    # Sparsity
    active = feature_acts.bool().float()
    l0 = active.sum(-1).mean().item()
    feature_density = active.mean(0)
    dead_features = (feature_density < 1e-6).sum().item()

    result = {
        "explained_variance": explained_variance.item(),
        "l0": l0,
        "dead_features": dead_features,
        "total_features": feature_acts.shape[-1],
        "dead_ratio": dead_features / feature_acts.shape[-1],
        "mean_feature_density": feature_density.mean().item(),
    }

    if sparsity is not None:
        # Top most/least frequent features from training
        top_freq = torch.topk(sparsity, 10)
        bot_freq = torch.topk(sparsity, 10, largest=False)
        result["top_frequent_features"] = [
            {"index": i.item(), "log10_freq": v.item()} for i, v in zip(top_freq.indices, top_freq.values)
        ]
        result["rarest_features"] = [
            {"index": i.item(), "log10_freq": v.item()} for i, v in zip(bot_freq.indices, bot_freq.values)
        ]

    return result


@torch.no_grad()
def make_ablation_hook(
    sae: SAE,
    scaling_factor: float,
    feature_indices: list[int],
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create a hook that ablates (zeros out) specific SAE features.

    Args:
        sae: Loaded JumpReLU SAE.
        scaling_factor: From training.
        feature_indices: Features to ablate.

    Returns:
        Hook function for transformer_lens.
    """
    def hook_fn(activation: torch.Tensor, _hook: HookPoint) -> torch.Tensor:
        # Encode to feature space
        scaled = activation * scaling_factor
        features = sae.encode(scaled)

        # Zero out specified features
        features[:, :, feature_indices] = 0.0

        # Decode back to activation space
        reconstructed = sae.decode(features) / scaling_factor
        return reconstructed

    return hook_fn


@torch.no_grad()
def make_patching_hook(
    sae: SAE,
    scaling_factor: float,
    feature_indices: list[int] | None = None,
    mode: str = "keep_only",
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create a hook that patches activations via SAE decomposition.

    Args:
        sae: Loaded JumpReLU SAE.
        scaling_factor: From training.
        feature_indices: Features to keep/remove depending on mode.
        mode: "keep_only" keeps only specified features, "remove" removes them.

    Returns:
        Hook function for transformer_lens.
    """
    def hook_fn(activation: torch.Tensor, _hook: HookPoint) -> torch.Tensor:
        scaled = activation * scaling_factor
        features = sae.encode(scaled)

        if feature_indices is not None:
            if mode == "keep_only":
                mask = torch.zeros_like(features)
                mask[:, :, feature_indices] = 1.0
                features = features * mask
            elif mode == "remove":
                features[:, :, feature_indices] = 0.0

        reconstructed = sae.decode(features) / scaling_factor
        return reconstructed

    return hook_fn


@torch.no_grad()
def make_hf_replacement_hook(
    sae: SAE,
    scaling_factor: float,
) -> Callable:
    """Create a forward hook for HuggingFace T5 decoder MLP layer.

    Replaces MLP output with SAE-reconstructed version. Use with:
        model.t5.decoder.block[layer].layer[2].DenseReluDense.register_hook(hook)

    Args:
        sae: Loaded JumpReLU SAE.
        scaling_factor: From training.

    Returns:
        Hook function compatible with nn.Module.register_forward_hook().
    """
    def hook_fn(module, input, output):
        orig_shape = output.shape
        flat = output.reshape(-1, orig_shape[-1])
        scaled = flat * scaling_factor
        features = sae.encode(scaled)
        reconstructed = sae.decode(features) / scaling_factor
        return reconstructed.reshape(orig_shape)

    return hook_fn


@torch.no_grad()
def run_with_intervention(
    text: str,
    model: HookedEncoderDecoder,
    hook_name: str,
    hook_fn: Callable[[torch.Tensor, HookPoint], torch.Tensor],
    max_new_tokens: int = 50,
    context_size: int = 128,
) -> tuple[str, str]:
    """Run model with and without hook intervention, compare outputs.

    Args:
        text: Input text.
        model: HookedEncoderDecoder model.
        hook_name: Name of hook point to intervene on.
        hook_fn: Hook function for intervention.
        max_new_tokens: Max tokens to generate.
        context_size: Max encoder context length.

    Returns:
        Tuple of (original_output, intervened_output).
    """
    tokenizer = model.tokenizer
    enc_tokens = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=context_size, padding=False,
    )

    # Generate without intervention
    with model.hooks(fwd_hooks=[]):
        output_orig = model.generate(
            enc_tokens.input_ids,
            attention_mask=enc_tokens.attention_mask,
            max_new_tokens=max_new_tokens,
        )

    # Generate with intervention
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output_intervened = model.generate(
            enc_tokens.input_ids,
            attention_mask=enc_tokens.attention_mask,
            max_new_tokens=max_new_tokens,
        )

    text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)
    text_intervened = tokenizer.decode(output_intervened[0], skip_special_tokens=True)

    return text_orig, text_intervened


def main():
    parser = argparse.ArgumentParser(description="BatchTopK SAE Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to inference checkpoint dir")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hook_name", type=str, default="decoder.12.hook_mlp_out")
    parser.add_argument("--context_size", type=int, default=128)
    parser.add_argument("--target_size", type=int, default=64)
    args = parser.parse_args()

    # Load SAE
    print(f"Loading SAE from {args.checkpoint}...")
    sae, meta = load_sae(args.checkpoint, device=args.device)
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    print(f"  scaling_factor={meta['scaling_factor']:.4f}")

    # Load T5 model
    print("Loading T5-large...")
    model = HookedEncoderDecoder.from_pretrained("google-t5/t5-large")
    model.eval()

    # Example: analyze a few sentences
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny with clear skies.",
    ]
    print(f"\nCollecting activations from {len(texts)} texts...")
    activations = collect_activations(
        texts, model, args.hook_name,
        context_size=args.context_size,
        target_size=args.target_size,
    )
    print(f"  Collected {activations.shape[0]} tokens, dim={activations.shape[1]}")

    # Analyze
    print("\nFeature analysis:")
    stats = analyze_features(sae, activations, meta["scaling_factor"], meta["sparsity"])
    print(f"  Explained Variance: {stats['explained_variance']:.4f}")
    print(f"  L0 (avg active features): {stats['l0']:.1f}")
    print(f"  Dead features: {stats['dead_features']}/{stats['total_features']} ({stats['dead_ratio']:.1%})")

    # Logit Lens
    print("\nLogit Lens (top tokens per feature):")
    lens_results = logit_lens(sae, model, n_features=10, top_k=5)
    for feat_idx, tokens_scores in lens_results.items():
        token_strs = [f"'{t}' ({s:.2f})" for t, s in tokens_scores]
        print(f"  Feature {feat_idx:5d}: {', '.join(token_strs)}")

    # Hook Intervention Demo
    print("\n" + "="*60)
    print("Hook Intervention Demo")
    print("="*60)

    # Get top active features from first text for demo
    demo_text = texts[0]
    demo_acts = collect_activations([demo_text], model, args.hook_name,
                                     args.context_size, args.target_size)
    feature_acts = get_feature_activations(sae, demo_acts, meta["scaling_factor"])
    top_features = feature_acts.sum(dim=0).topk(5).indices.tolist()
    print(f"\nTop 5 active features for demo: {top_features}")

    # 1. Ablation: zero out top features
    print("\n[1] Ablating top 5 features...")
    ablation_hook = make_ablation_hook(sae, meta["scaling_factor"], top_features)
    orig, ablated = run_with_intervention(
        demo_text, model, args.hook_name, ablation_hook, max_new_tokens=30
    )
    print(f"  Original:  {orig}")
    print(f"  Ablated:   {ablated}")

    # 2. Keep only top features
    print("\n[2] Keeping only top 5 features...")
    keep_hook = make_patching_hook(sae, meta["scaling_factor"], top_features, mode="keep_only")
    orig, kept = run_with_intervention(
        demo_text, model, args.hook_name, keep_hook, max_new_tokens=30
    )
    print(f"  Original:  {orig}")
    print(f"  Keep only: {kept}")

    # 3. Full SAE reconstruction (no intervention, just pass through SAE)
    print("\n[3] Full SAE reconstruction (all features)...")
    recon_hook = make_patching_hook(sae, meta["scaling_factor"], mode="keep_only")
    orig, reconstructed = run_with_intervention(
        demo_text, model, args.hook_name, recon_hook, max_new_tokens=30
    )
    print(f"  Original:    {orig}")
    print(f"  Reconstructed: {reconstructed}")


if __name__ == "__main__":
    main()
