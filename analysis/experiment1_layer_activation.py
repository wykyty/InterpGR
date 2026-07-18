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
    uv run python analysis/experiment1_layer_activation.py \
        --cache-dir data/activation_cache \
        --checkpoint-root out/sae_train_8x \
        --output-dir results/layer_activation_statistics
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import (  # noqa: E402
    discover_layers,
    linear_trend,
    load_sae,
    resolve_device,
)


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
        default=Path("data/activation_cache"),
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
