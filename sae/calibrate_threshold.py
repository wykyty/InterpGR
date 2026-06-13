"""
校准 JumpReLU SAE 的 threshold，使其平均 L0 ≈ 目标 k。

用法:
    python sae/calibrate_threshold.py \
        --inference_dir out/sae_train_4x/layer_14/inference \
        --cache_dir data/activation_cache_train \
        --target_l0 100 \
        --num_samples 50000
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="data/activation_cache_train")
    parser.add_argument("--target_l0", type=float, default=100.0,
                        help="Target average L0 (should match training k)")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--save", action="store_true", help="Overwrite threshold in sae_weights.safetensors")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_path = Path(args.inference_dir)

    # Load config
    with open(inference_path / "cfg.json") as f:
        cfg = json.load(f)
    d_in = cfg["d_in"]
    d_sae = cfg["d_sae"]
    layer = int(inference_path.parent.name.split("_")[-1])

    # Load weights
    weights = load_file(str(inference_path / "sae_weights.safetensors"))
    W_enc = weights["W_enc"].to(device)   # [d_in, d_sae]
    b_enc = weights["b_enc"].to(device)   # [d_sae]

    # Load activations
    cache_path = Path(args.cache_dir) / f"layer_{layer}.safetensors"
    print(f"Loading activations from {cache_path}...")
    data = load_file(str(cache_path))
    acts = data["activations"][:args.num_samples].float().to(device)
    print(f"Using {acts.shape[0]} samples, d_in={acts.shape[1]}")

    # Compute hidden_pre
    print("Computing hidden_pre...")
    hidden_pre = acts @ W_enc + b_enc  # [N, d_sae]

    # Flatten and sort to find the threshold
    flat = hidden_pre.flatten().sort(descending=True)[0]
    n_total = flat.numel()

    # Target: each token activates target_l0 features on average
    # Total active = target_l0 * N
    target_total_active = int(args.target_l0 * acts.shape[0])
    target_total_active = min(target_total_active, n_total - 1)

    # Threshold = value at position target_total_active
    new_threshold = flat[target_total_active].item()

    # Verify
    active = (hidden_pre > new_threshold).float()
    l0_per_token = active.sum(dim=-1)
    actual_l0 = l0_per_token.mean().item()
    dead = (active.sum(dim=0) == 0).sum().item()

    print(f"\n{'='*50}")
    print(f"  Old threshold : {weights['threshold'][0].item():.6f}")
    print(f"  New threshold : {new_threshold:.6f}")
    print(f"  Target L0     : {args.target_l0:.1f}")
    print(f"  Actual L0     : {actual_l0:.1f}")
    print(f"  Dead features : {dead}/{d_sae} ({dead/d_sae*100:.2f}%)")
    print(f"{'='*50}")

    if args.save:
        weights["threshold"] = torch.full_like(weights["threshold"], new_threshold)
        save_file(weights, str(inference_path / "sae_weights.safetensors"))
        print(f"\nSaved updated weights to {inference_path / 'sae_weights.safetensors'}")
    else:
        print("\nDry run. Use --save to overwrite the weights.")


if __name__ == "__main__":
    main()
