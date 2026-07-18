"""Shared utilities for the layer-wise SAE analysis experiments."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


LAYER_CACHE_PATTERN = re.compile(r"layer_(\d+)\.safetensors$")
LAYER_DIR_PATTERN = re.compile(r"layer_(\d+)$")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def discover_layers(cache_dir: Path, checkpoint_root: Path) -> list[int]:
    """Return sorted layers with both an activation cache and SAE checkpoint."""
    cache_layers = {
        int(match.group(1))
        for path in cache_dir.glob("layer_*.safetensors")
        if (match := LAYER_CACHE_PATTERN.fullmatch(path.name))
    }
    checkpoint_layers = {
        int(match.group(1))
        for path in checkpoint_root.glob("layer_*")
        if path.is_dir()
        and (match := LAYER_DIR_PATTERN.fullmatch(path.name))
        and (path / "sae_config.json").is_file()
        and (path / "sae_weights.safetensors").is_file()
    }
    layers = sorted(cache_layers & checkpoint_layers)
    if not layers:
        raise FileNotFoundError(
            "No matching cache/checkpoint layers were found under "
            f"{cache_dir} and {checkpoint_root}"
        )
    return layers


def load_training_sae(checkpoint_dir: Path, device: torch.device):
    """Load a folded BatchTopK training checkpoint."""
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
    """Load a JumpReLU inference checkpoint."""
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


def load_sae(checkpoint_dir: Path, device: torch.device, sae_format: str):
    """Load the requested SAE format, preferring JumpReLU in auto mode."""
    inference_files_exist = (
        (checkpoint_dir / "inference" / "cfg.json").is_file()
        and (checkpoint_dir / "inference" / "sae_weights.safetensors").is_file()
    )
    if sae_format in {"auto", "inference"} and inference_files_exist:
        return load_inference_sae(checkpoint_dir, device)
    if sae_format == "inference":
        raise FileNotFoundError(
            f"Missing JumpReLU checkpoint under {checkpoint_dir / 'inference'}"
        )
    return load_training_sae(checkpoint_dir, device)


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


def summarize_values(values: list[float]) -> tuple[float, float]:
    """Return mean and standard error of the mean."""
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if len(array) else 0.0
    sem = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, sem


def weighted_purity(labels: np.ndarray, weights: np.ndarray) -> tuple[float, int]:
    """Return dominant weighted-label share and its label."""
    mass: dict[int, float] = defaultdict(float)
    for label, weight in zip(labels, weights):
        mass[int(label)] += float(weight)
    if not mass:
        return 0.0, -1
    dominant_label, dominant_mass = max(mass.items(), key=lambda item: item[1])
    total_mass = sum(mass.values())
    return (dominant_mass / total_mass if total_mass > 0 else 0.0), dominant_label


def sample_top_latents(
    activation_mass: torch.Tensor,
    active_token_count: torch.Tensor,
    n_latents: int,
    candidate_pool_size: int,
    min_active_tokens: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Randomly sample latents from a high-activation candidate pool."""
    eligible = torch.where(active_token_count >= min_active_tokens)[0]
    if len(eligible) < n_latents:
        raise RuntimeError(
            f"Only {len(eligible)} latents have at least {min_active_tokens} active "
            f"tokens; cannot sample {n_latents}"
        )
    pool_size = min(candidate_pool_size, len(eligible))
    pool = eligible[torch.topk(activation_mass[eligible], pool_size).indices].numpy()
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(pool, size=n_latents, replace=False).astype(np.int64))
    return sampled, pool_size
