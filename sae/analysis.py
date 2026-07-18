"""Layer-wise activation statistics for DSI decoder SAEs.

For every layer, this script loads cached decoder activations, encodes them
with the corresponding SAE, and reports:

* Average Active Latents: mean number of active SAE latents per token.
* Average Activation: mean absolute activation over active latents only.
* Activation Sparsity: fraction of zero/inactive entries in the latent matrix.
* Activation Variance: mean, across latents, of each latent's variance over
  tokens (zeros included).

It also writes a Layer-vs-Active-Latents curve and a simple trend summary for
checking whether later layers become increasingly sparse. By default, the
JumpReLU inference SAE is used so that the learned threshold determines the
number of active latents. The BatchTopK training SAE can force the average
active count toward k and is therefore less suitable for this comparison.

Example:
    uv run python sae/analysis.py \
        --cache-dir data/activation_cache_dev \
        --checkpoint-root out/sae_train_8x \
        --output-dir results/layer_activation_statistics
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm


LAYER_CACHE_PATTERN = re.compile(r"layer_(\d+)\.safetensors$")
LAYER_DIR_PATTERN = re.compile(r"layer_(\d+)$")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def discover_layers(cache_dir: Path, checkpoint_root: Path) -> list[int]:
    """Return sorted layers that have both activations and an SAE checkpoint."""
    cache_layers = {
        int(match.group(1))
        for path in cache_dir.glob("layer_*.safetensors")
        if (match := LAYER_CACHE_PATTERN.fullmatch(path.name))
    }
    checkpoint_layers = {
        int(match.group(1))
        for path in checkpoint_root.glob("layer_*")
        if path.is_dir() and (match := LAYER_DIR_PATTERN.fullmatch(path.name))
        and (path / "sae_config.json").is_file()
        and (path / "sae_weights.safetensors").is_file()
    }

    layers = sorted(cache_layers & checkpoint_layers)
    if not layers:
        raise FileNotFoundError(
            "No matching layers were found. Expected activation files such as "
            f"'{cache_dir / 'layer_0.safetensors'}' and SAE directories such as "
            f"'{checkpoint_root / 'layer_0'}'."
        )
    return layers


def load_training_sae(checkpoint_dir: Path, device: torch.device):
    """Load a folded BatchTopK training checkpoint produced by train_sae.py."""
    from sae_lens.saes.batchtopk_sae import (
        BatchTopKTrainingSAE,
        BatchTopKTrainingSAEConfig,
    )

    with open(checkpoint_dir / "sae_config.json", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=saved_cfg["d_in"],
        d_sae=saved_cfg["d_sae"],
        k=saved_cfg["k"],
        aux_loss_coefficient=1.0,
        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.1,
        apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in",
        decoder_init_norm=0.01,
        device=str(device),
        dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    weights = load_file(str(checkpoint_dir / "sae_weights.safetensors"))
    sae.W_enc.data.copy_(weights["W_enc"].to(device))
    sae.W_dec.data.copy_(weights["W_dec"].to(device))
    sae.b_enc.data.copy_(weights["b_enc"].to(device))
    sae.b_dec.data.copy_(weights["b_dec"].to(device))
    sae.topk_threshold.data.copy_(weights["topk_threshold"].to(device))
    sae.eval()
    return sae, saved_cfg, "training"


def load_inference_sae(checkpoint_dir: Path, device: torch.device):
    """Load the JumpReLU inference checkpoint produced by train_sae.py."""
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

    inference_dir = checkpoint_dir / "inference"
    with open(inference_dir / "cfg.json", encoding="utf-8") as f:
        inference_cfg = json.load(f)

    sae_cfg = JumpReLUSAEConfig(
        d_in=inference_cfg["d_in"],
        d_sae=inference_cfg["d_sae"],
        apply_b_dec_to_input=inference_cfg.get("apply_b_dec_to_input", False),
        device=str(device),
        dtype="float32",
    )
    sae = JumpReLUSAE(sae_cfg).to(device)

    weights = load_file(str(inference_dir / "sae_weights.safetensors"))
    sae.W_enc.data.copy_(weights["W_enc"].to(device))
    sae.W_dec.data.copy_(weights["W_dec"].to(device))
    sae.b_enc.data.copy_(weights["b_enc"].to(device))
    sae.b_dec.data.copy_(weights["b_dec"].to(device))
    sae.threshold.data.copy_(weights["threshold"].to(device))
    sae.eval()
    return sae, inference_cfg, "inference"


def load_sae(
    checkpoint_dir: Path, device: torch.device, sae_format: str
):
    """Load the requested SAE format, preferring JumpReLU for layer trends."""
    inference_files_exist = (
        (checkpoint_dir / "inference" / "cfg.json").is_file()
        and (checkpoint_dir / "inference" / "sae_weights.safetensors").is_file()
    )
    if sae_format in {"auto", "inference"} and inference_files_exist:
        return load_inference_sae(checkpoint_dir, device)
    if sae_format == "inference":
        raise FileNotFoundError(
            f"Missing JumpReLU inference checkpoint under {checkpoint_dir / 'inference'}"
        )
    return load_training_sae(checkpoint_dir, device)


@torch.inference_mode()
def compute_layer_statistics(
    layer: int,
    cache_dir: Path,
    checkpoint_root: Path,
    device: torch.device,
    batch_size: int,
    activation_threshold: float,
    max_tokens: int | None,
    sae_format: str,
) -> dict[str, int | float | str]:
    """Compute exact streaming latent statistics for one decoder layer."""
    cache_path = cache_dir / f"layer_{layer}.safetensors"
    checkpoint_dir = checkpoint_root / f"layer_{layer}"

    cached = load_file(str(cache_path))
    activations = cached["activations"]
    if activations.ndim != 2:
        raise ValueError(
            f"Layer {layer}: expected [tokens, d_in], got {tuple(activations.shape)}"
        )

    sae, sae_cfg, loaded_format = load_sae(checkpoint_dir, device, sae_format)
    d_in = int(sae_cfg["d_in"])
    d_sae = int(sae_cfg["d_sae"])
    if activations.shape[1] != d_in:
        raise ValueError(
            f"Layer {layer}: cache d_in={activations.shape[1]} but SAE d_in={d_in}"
        )

    n_tokens = int(activations.shape[0])
    if max_tokens is not None:
        n_tokens = min(n_tokens, max_tokens)
    if n_tokens == 0:
        raise ValueError(f"Layer {layer}: no activation tokens to analyze")

    total_active = 0
    total_active_magnitude = 0.0
    # Per-feature moments remain on CPU in float64 for stable variance estimates.
    feature_sum = torch.zeros(d_sae, dtype=torch.float64)
    feature_sum_sq = torch.zeros(d_sae, dtype=torch.float64)

    starts = range(0, n_tokens, batch_size)
    for start in tqdm(
        starts,
        total=(n_tokens + batch_size - 1) // batch_size,
        desc=f"Layer {layer:02d}",
    ):
        end = min(start + batch_size, n_tokens)
        batch = activations[start:end].to(device, non_blocking=True)
        latent_acts = sae.encode(batch).float()
        if latent_acts.ndim != 2 or latent_acts.shape[1] != d_sae:
            raise ValueError(
                f"Layer {layer}: expected SAE output [tokens, {d_sae}], "
                f"got {tuple(latent_acts.shape)}"
            )

        active_mask = latent_acts.abs() > activation_threshold
        batch_active = int(active_mask.sum().item())
        total_active += batch_active
        if batch_active:
            total_active_magnitude += float(
                latent_acts.abs().masked_select(active_mask).sum().item()
            )

        # Sum over tokens first, then transfer only two d_sae-sized vectors.
        feature_sum += latent_acts.sum(dim=0).double().cpu()
        feature_sum_sq += latent_acts.square().sum(dim=0).double().cpu()

    total_latent_entries = n_tokens * d_sae
    average_active_latents = total_active / n_tokens
    average_activation = (
        total_active_magnitude / total_active if total_active else 0.0
    )
    activation_sparsity = 1.0 - (total_active / total_latent_entries)

    feature_mean = feature_sum / n_tokens
    feature_variance = feature_sum_sq / n_tokens - feature_mean.square()
    # Small negative values can arise from floating-point roundoff.
    feature_variance.clamp_(min=0.0)
    activation_variance = float(feature_variance.mean().item())

    result: dict[str, int | float | str] = {
        "layer": layer,
        "n_tokens": n_tokens,
        "d_in": d_in,
        "d_sae": d_sae,
        "sae_format": loaded_format,
        "average_active_latents": average_active_latents,
        "average_activation": average_activation,
        "activation_sparsity": activation_sparsity,
        "activation_variance": activation_variance,
    }

    del sae, activations, cached
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def linear_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return least-squares slope and Pearson correlation."""
    if len(x) < 2:
        return 0.0, 0.0
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    x_ss = float(np.dot(centered_x, centered_x))
    y_ss = float(np.dot(centered_y, centered_y))
    slope = float(np.dot(centered_x, centered_y) / x_ss) if x_ss else 0.0
    correlation = (
        float(np.dot(centered_x, centered_y) / np.sqrt(x_ss * y_ss))
        if x_ss and y_ss
        else 0.0
    )
    return slope, correlation


def summarize_sparsity_trend(
    results: list[dict[str, int | float | str]],
) -> dict:
    """Summarize whether later layers tend to use fewer active latents."""
    layers = np.asarray([row["layer"] for row in results], dtype=np.float64)
    active = np.asarray(
        [row["average_active_latents"] for row in results], dtype=np.float64
    )
    sparsity = np.asarray(
        [row["activation_sparsity"] for row in results], dtype=np.float64
    )

    active_slope, active_correlation = linear_trend(layers, active)
    sparsity_slope, sparsity_correlation = linear_trend(layers, sparsity)
    adjacent_decrease_fraction = (
        float(np.mean(np.diff(active) <= 0.0)) if len(active) > 1 else 0.0
    )
    increasingly_sparse = active_slope < 0.0 and sparsity_slope > 0.0

    return {
        "increasingly_sparse": increasingly_sparse,
        "interpretation": (
            "Later layers tend to be sparser."
            if increasingly_sparse
            else "Later layers do not show a consistent increase in sparsity."
        ),
        "active_latents_slope_per_layer": active_slope,
        "active_latents_layer_correlation": active_correlation,
        "sparsity_slope_per_layer": sparsity_slope,
        "sparsity_layer_correlation": sparsity_correlation,
        "adjacent_active_latents_decrease_fraction": adjacent_decrease_fraction,
    }


def save_results(
    results: list[dict[str, int | float | str]], trend: dict, output_dir: Path
) -> None:
    """Write machine-readable statistics and the requested curve."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer",
        "n_tokens",
        "d_in",
        "d_sae",
        "sae_format",
        "average_active_latents",
        "average_activation",
        "activation_sparsity",
        "activation_variance",
    ]
    with open(
        output_dir / "layer_activation_statistics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(
        output_dir / "layer_activation_statistics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "sparsity_trend.json", "w", encoding="utf-8") as f:
        json.dump(trend, f, indent=2, ensure_ascii=False)

    # Import lazily so statistics can still be produced in environments where
    # plotting dependencies are installed separately.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate Layer vs Active Latents curve"
        ) from exc

    layers = [int(row["layer"]) for row in results]
    active_latents = [float(row["average_active_latents"]) for row in results]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(layers, active_latents, marker="o", linewidth=2, markersize=5)
    ax.set_title("Layer vs Average Active Latents")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Average Active Latents per Token")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_vs_active_latents.png", dpi=200)
    fig.savefig(output_dir / "layer_vs_active_latents.pdf")
    plt.close(fig)


def print_table(results: list[dict[str, int | float | str]], trend: dict) -> None:
    header = (
        f"{'Layer':>5} {'Active Latents':>15} {'Avg Activation':>15} "
        f"{'Sparsity':>12} {'Variance':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        print(
            f"{int(row['layer']):>5d} "
            f"{float(row['average_active_latents']):>15.4f} "
            f"{float(row['average_activation']):>15.6f} "
            f"{float(row['activation_sparsity']):>12.6f} "
            f"{float(row['activation_variance']):>14.6f}"
        )
    print(f"\nSparsity trend: {trend['interpretation']}")
    print(
        "Active-latent slope per layer: "
        f"{trend['active_latents_slope_per_layer']:.6f}; "
        "correlation: "
        f"{trend['active_latents_layer_correlation']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute layer-wise SAE activation statistics and sparsity trend"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/activation_cache_dev"),
        help="Directory containing layer_N.safetensors activation caches",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("out/sae_train_8x"),
        help="Directory containing layer_N SAE checkpoint directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/layer_activation_statistics"),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional layer subset, e.g. --layers 0 6 12 18 23",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="A latent is active when abs(value) is greater than this threshold",
    )
    parser.add_argument(
        "--sae-format",
        choices=("inference", "training", "auto"),
        default="inference",
        help=(
            "Use JumpReLU inference checkpoints by default; training uses "
            "BatchTopK and may keep active counts close to k"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional deterministic prefix limit per layer for quick analysis",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:N"
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.activation_threshold < 0:
        parser.error("--activation-threshold must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    available_layers = discover_layers(args.cache_dir, args.checkpoint_root)

    if args.layers is None:
        layers = available_layers
    else:
        missing = sorted(set(args.layers) - set(available_layers))
        if missing:
            raise FileNotFoundError(
                f"Requested layers are missing cache or SAE checkpoints: {missing}"
            )
        layers = sorted(set(args.layers))

    print(f"Device: {device}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Checkpoint root: {args.checkpoint_root}")
    print(f"Layers: {layers}")

    results = [
        compute_layer_statistics(
            layer=layer,
            cache_dir=args.cache_dir,
            checkpoint_root=args.checkpoint_root,
            device=device,
            batch_size=args.batch_size,
            activation_threshold=args.activation_threshold,
            max_tokens=args.max_tokens,
            sae_format=args.sae_format,
        )
        for layer in layers
    ]

    trend = summarize_sparsity_trend(results)
    save_results(results, trend, args.output_dir)
    print_table(results, trend)
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
