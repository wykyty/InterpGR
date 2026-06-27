"""
Save Latent Concepts from Trained SAE

Extracts feature activations from a trained SAE and saves them as JSONL
for subsequent interpretability analysis.

Usage:
    uv run python sae/save_latent_concepts.py \
        --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
        --cache_dir data/activation_cache_train \
        --output_path results/latent_concepts/layer_12.jsonl \
        --threshold 0.01
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig


class ActivationDataset(Dataset):
    """Dataset for loading cached activations."""
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


def load_sae(checkpoint_dir: str, device: str) -> BatchTopKTrainingSAE:
    """Load trained SAE model from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)

    # Load config
    with open(checkpoint_path / "sae_config.json", "r") as f:
        config = json.load(f)

    # Create SAE
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=config["d_in"],
        d_sae=config["d_sae"],
        k=config["k"],
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

    # Load weights
    weights = load_file(str(checkpoint_path / "sae_weights.safetensors"))
    sae.W_enc.data = weights["W_enc"].to(device)
    sae.W_dec.data = weights["W_dec"].to(device)
    sae.b_enc.data = weights["b_enc"].to(device)
    sae.b_dec.data = weights["b_dec"].to(device)
    sae.topk_threshold = weights["topk_threshold"].to(device)
    sae.eval()

    return sae


@torch.no_grad()
def extract_latent_concepts(
    sae: BatchTopKTrainingSAE,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.01,
    max_batches: int = None,
) -> list[dict]:
    """
    Extract latent concepts (active features) from SAE.

    Args:
        sae: Trained SAE model
        dataloader: DataLoader for activation data
        device: Device to run on
        threshold: Minimum activation value to consider a feature active
        max_batches: Maximum number of batches to process (None for all)

    Returns:
        List of dicts with 'docid', 'ids', 'weight' for each sample
    """
    sae.eval()
    all_results = []
    docid = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting latent concepts")):
        if max_batches and batch_idx >= max_batches:
            break

        batch = batch.to(device)

        # Get feature activations
        feature_acts, hidden_pre = sae.encode_with_hidden_pre(batch)

        # Process each sample in the batch
        for i in range(batch.shape[0]):
            acts = feature_acts[i]  # (d_sae,)

            # Find active features (above threshold)
            active_mask = acts > threshold
            active_indices = torch.where(active_mask)[0]
            active_weights = acts[active_mask]

            if len(active_indices) > 0:
                all_results.append({
                    "docid": docid,
                    "ids": active_indices.cpu().tolist(),
                    "weight": active_weights.cpu().tolist(),
                })
            else:
                # No active features
                all_results.append({
                    "docid": docid,
                    "ids": [],
                    "weight": [],
                })

            docid += 1

    return all_results


@torch.no_grad()
def extract_latent_concepts_with_metadata(
    sae: BatchTopKTrainingSAE,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.01,
    max_batches: int = None,
) -> dict:
    """
    Extract latent concepts with additional metadata for interpretability.

    Returns:
        Dict with 'concepts' (list of concept dicts) and 'statistics' (summary stats)
    """
    sae.eval()
    concepts = []
    docid = 0

    # Statistics
    total_features = sae.cfg.d_sae
    feature_activation_counts = torch.zeros(total_features, device=device)
    total_tokens = 0
    total_active_features = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting latent concepts")):
        if max_batches and batch_idx >= max_batches:
            break

        batch = batch.to(device)
        feature_acts, hidden_pre = sae.encode_with_hidden_pre(batch)

        for i in range(batch.shape[0]):
            acts = feature_acts[i]
            active_mask = acts > threshold
            active_indices = torch.where(active_mask)[0]
            active_weights = acts[active_mask]

            # Update statistics
            feature_activation_counts[active_mask] += 1
            total_tokens += 1
            total_active_features += len(active_indices)

            # Store concept info
            concepts.append({
                "docid": docid,
                "ids": active_indices.cpu().tolist(),
                "weight": active_weights.cpu().tolist(),
                "num_active": len(active_indices),
                "total_activation": acts.sum().item(),
            })

            docid += 1

    # Compute statistics
    feature_freq = feature_activation_counts / total_tokens
    dead_features = (feature_freq == 0).sum().item()
    most_active_features = torch.topk(feature_freq, 10)
    least_active_features = torch.topk(feature_freq, 10, largest=False)

    statistics = {
        "total_tokens": total_tokens,
        "total_features": total_features,
        "dead_features": dead_features,
        "dead_ratio": dead_features / total_features,
        "avg_active_features": total_active_features / total_tokens,
        "most_active_features": [
            {"index": idx.item(), "frequency": freq.item()}
            for idx, freq in zip(most_active_features.indices, most_active_features.values)
        ],
        "least_active_features": [
            {"index": idx.item(), "frequency": freq.item()}
            for idx, freq in zip(least_active_features.indices, least_active_features.values)
        ],
    }

    return {
        "concepts": concepts,
        "statistics": statistics,
    }


def main():
    parser = argparse.ArgumentParser(description="Save latent concepts from trained SAE")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to SAE checkpoint directory (e.g., checkpoints/dsi_sae_semantic/layer_12)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory containing cached activations")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save latent concepts (JSONL format)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum activation value to consider a feature active")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for processing")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum number of batches to process (None for all)")
    parser.add_argument("--save_metadata", action="store_true",
                        help="Save additional metadata and statistics")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load SAE
    print(f"Loading SAE from {args.checkpoint_dir}...")
    sae = load_sae(args.checkpoint_dir, args.device)
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={sae.cfg.k}")

    # Load activation dataset
    # Extract layer from checkpoint directory name
    layer = int(Path(args.checkpoint_dir).name.split("_")[-1])
    print(f"Loading activations for layer {layer}...")
    dataset = ActivationDataset(args.cache_dir, layer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Extract latent concepts
    print("Extracting latent concepts...")
    if args.save_metadata:
        results = extract_latent_concepts_with_metadata(
            sae, dataloader, args.device, args.threshold, args.max_batches
        )
        concepts = results["concepts"]
        statistics = results["statistics"]

        # Save statistics separately
        stats_path = args.output_path.replace(".jsonl", "_statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_path}")
    else:
        concepts = extract_latent_concepts(
            sae, dataloader, args.device, args.threshold, args.max_batches
        )

    # Save concepts as JSONL
    print(f"Saving {len(concepts)} concepts to {args.output_path}...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for concept in concepts:
            f.write(json.dumps(concept, ensure_ascii=False) + "\n")

    print(f"✅ Latent concepts saved to {args.output_path}")

    # Print summary
    if args.save_metadata:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total tokens processed: {statistics['total_tokens']}")
        print(f"Total features: {statistics['total_features']}")
        print(f"Dead features: {statistics['dead_features']} ({statistics['dead_ratio']:.2%})")
        print(f"Average active features per token: {statistics['avg_active_features']:.1f}")
        print(f"\nTop 5 most active features:")
        for feat in statistics["most_active_features"][:5]:
            print(f"  Feature {feat['index']}: {feat['frequency']:.4f}")
        print(f"\nTop 5 least active features:")
        for feat in statistics["least_active_features"][:5]:
            print(f"  Feature {feat['index']}: {feat['frequency']:.6f}")


if __name__ == "__main__":
    main()