"""
Train BatchTopK SAE on cached decoder MLP activations.

Supports single-GPU, CUDA_VISIBLE_DEVICES parallel, and torchrun DDP.

Prerequisites:
    Run sae/cache_activations.py first to pre-collect activations.

Usage (single GPU):
    python sae/train_sae.py --cache_dir data/activation_cache --layer 12

Usage (torchrun DDP, 8 GPUs for one layer):
    uv run torchrun --nproc_per_node=8 sae/train_sae.py \
        --cache_dir data/activation_cache_dev \
        --lr 3e-4 \
        --layer 12 --save_dir out/sae_semantic_dev
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput


class CachedActivationLoader:
    """Load pre-collected activations from disk."""

    def __init__(self, cache_dir: str, layer: int):
        cache_path = Path(cache_dir) / f"layer_{layer}.safetensors"
        print(f"Loading cached activations from {cache_path}...")
        data = load_file(str(cache_path))
        self.activations = data["activations"]  # (total_tokens, d_in)
        self.total_tokens = self.activations.shape[0]
        self.d_in = self.activations.shape[1]
        print(f"  {self.total_tokens} tokens, d_in={self.d_in}")

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.total_tokens, (batch_size,))
        return self.activations[indices].to(device)

class ActivationLoader:
    """仿 SAELens 逻辑的高效、无放回激活值加载器"""
    def __init__(self, cache_dir: str, layer: int, batch_size: int):
        cache_path = Path(cache_dir) / f"layer_{layer}.safetensors"
        print(f"Loading cached activations from {cache_path}...")
        data = load_file(str(cache_path))
        self.activations = data["activations"]  # (total_tokens, d_in)
        self.total_tokens = self.activations.shape[0]
        self.batch_size = batch_size
        self.d_in = self.activations.shape[1]
        self.reset_stream()

    def reset_stream(self):
        """当所有数据被看满一遍（一个等效 Epoch 结束）时，重新打散指针"""
        print("Dataset boundary reached. Shuffling indices...")
        self.indices = torch.randperm(self.total_tokens) # 生成无重复的乱序索引
        self.current_ptr = 0

    def sample(self, device: torch.device) -> torch.Tensor:
        # 如果剩余的 Token 不够凑满一个 Batch，说明全覆盖看了一遍，重置并重新打散
        if self.current_ptr + self.batch_size > self.total_tokens:
            self.reset_stream()
            
        # 取出当前段的无重复索引
        batch_indices = self.indices[self.current_ptr : self.current_ptr + self.batch_size]
        self.current_ptr += self.batch_size
        
        # 这样抽取可以确保在整个全量数据看完前，没有任何一个 Token 会被重复看
        return self.activations[batch_indices].to(device)

def train(args):
    # Distributed setup
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # Load cached activations
    loader = ActivationLoader(args.cache_dir, args.layer)
    d_in = loader.d_in

    # Create SAE
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in, d_sae=args.d_sae, k=args.k,
        aux_loss_coefficient=1.0, rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.01, apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in", decoder_init_norm=0.1,
        device=str(device), dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    if use_ddp:
        sae_ddp = nn.parallel.DistributedDataParallel(sae, device_ids=[local_rank])
    else:
        sae_ddp = sae

    # Estimate scaling factor
    if is_main:
        print("Estimating activation scaling factor...")
    norms = []
    for _ in range(args.n_batches_for_norm):
        batch = loader.sample(args.batch_size, device)
        norms.append(batch.norm(dim=-1).mean().item())

    mean_norm = np.mean(norms)
    scaling_factor = (d_in ** 0.5) / mean_norm
    sae.scaling_factor = torch.tensor(scaling_factor, device=device)
    if is_main:
        print(f"Scaling factor: {scaling_factor:.4f}")

    # Wandb (main rank only)
    if is_main and not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"layer_{args.layer}",
            config={
                "layer": args.layer, "d_in": d_in, "d_sae": args.d_sae,
                "k": args.k, "lr": args.lr, "batch_size": args.batch_size,
                "total_steps": args.total_steps, "scaling_factor": scaling_factor,
                "world_size": world_size,
            },
            reinit=True,
        )

    # Optimizer
    optimizer = torch.optim.Adam(sae_ddp.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sae_ddp.train()

    pbar = tqdm(range(args.total_steps), desc=f"Layer {args.layer}", disable=not is_main)
    feature_act_freq = torch.zeros(args.d_sae, device=device)
    total_tokens_seen = 0
    metrics = {
        "step": [], "total_loss": [], "mse_loss": [], "aux_loss": [],
        "explained_variance": [], "l0": [], "topk_threshold": [],
    }

    for step in pbar:
        sae_in = loader.sample(args.batch_size, device) * sae.scaling_factor

        base_sae = sae_ddp.module if use_ddp else sae_ddp

        step_output = base_sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=sae_in, coefficients={},
                dead_neuron_mask=None, n_training_steps=step,
                is_logging_step=(step % args.log_every == 0),
            )
        )

        

        optimizer.zero_grad()
        step_output.loss.backward()

        # Sync gradients across ranks
        if use_ddp:
            for param in sae_ddp.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(sae_ddp.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            l0 = step_output.feature_acts.bool().float().sum(-1).mean().item()
            per_token_l2 = (step_output.sae_out - sae_in).pow(2).sum(dim=-1)
            total_var = (sae_in - sae_in.mean(0)).pow(2).sum(dim=-1)
            ev = 1 - per_token_l2.mean() / (total_var.mean() + 1e-8)
            n_tokens = sae_in.shape[0]
            feature_act_freq += (step_output.feature_acts > 0).float().sum(dim=0)
            total_tokens_seen += n_tokens

        if is_main:
            metrics["step"].append(step)
            metrics["total_loss"].append(step_output.loss.item())
            metrics["mse_loss"].append(step_output.losses["mse_loss"].item())
            metrics["aux_loss"].append(step_output.losses["auxiliary_reconstruction_loss"].item())
            metrics["explained_variance"].append(ev.item())
            metrics["l0"].append(l0)
            metrics["topk_threshold"].append(sae.topk_threshold.item())

            if not args.no_wandb:
                import wandb
                wandb.log({
                    "train/total_loss": step_output.loss.item(),
                    "train/mse_loss": step_output.losses["mse_loss"].item(),
                    "train/aux_loss": step_output.losses["auxiliary_reconstruction_loss"].item(),
                    "train/explained_variance": ev, "train/l0": l0,
                    "train/topk_threshold": sae.topk_threshold.item(),
                }, step=step)

            if step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{step_output.loss.item():.4f}",
                    "ev": f"{ev.item():.4f}", "l0": f"{l0:.1f}",
                })

    if is_main:
        pbar.close()
        if not args.no_wandb:
            import wandb
            wandb.finish()

    # Evaluation & Save (main rank only)
    if is_main:
        print("Evaluating...")
        sae.eval()
        all_feature_acts, all_reconstructions, all_inputs = [], [], []
        with torch.no_grad():
            for _ in range(10):
                batch = loader.sample(args.batch_size, device) * sae.scaling_factor
                feature_acts, _ = sae.encode_with_hidden_pre(batch)
                all_feature_acts.append(feature_acts)
                all_reconstructions.append(sae.decode(feature_acts))
                all_inputs.append(batch)

        feature_acts = torch.cat(all_feature_acts, dim=0)
        reconstructions = torch.cat(all_reconstructions, dim=0)
        inputs = torch.cat(all_inputs, dim=0)

        per_token_mse = (reconstructions - inputs).pow(2).sum(dim=-1)
        total_var = (inputs - inputs.mean(0)).pow(2).sum(dim=-1)
        explained_variance = 1 - per_token_mse.mean() / (total_var.mean() + 1e-8)
        nmse = (per_token_mse / (inputs.pow(2).sum(dim=-1) + 1e-8)).mean()
        active = feature_acts.bool().float()
        l0 = active.sum(-1).mean()
        dead_features = (active.mean(0) < 1e-6).sum().item()

        print(f"  EV={explained_variance:.4f}  NMSE={nmse:.4f}  L0={l0:.1f}  "
              f"Dead={dead_features}/{feature_acts.shape[-1]}")

        # Save
        save_dir = Path(args.save_dir) / f"layer_{args.layer}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log_feature_sparsity = torch.log10(feature_act_freq / total_tokens_seen + 1e-10)
        hook_name = f"decoder.{args.layer}.hook_mlp_out"

        save_file({
            "W_enc": sae.W_enc.data, "W_dec": sae.W_dec.data,
            "b_enc": sae.b_enc.data, "b_dec": sae.b_dec.data,
            "scaling_factor": sae.scaling_factor, "topk_threshold": sae.topk_threshold,
        }, str(save_dir / "sae_weights.safetensors"))

        with open(save_dir / "sae_config.json", "w") as f:
            json.dump({
                "d_in": d_in, "d_sae": args.d_sae, "k": args.k,
                "hook_name": hook_name, "layer": args.layer,
                "model_name": "google-t5/t5-large (DSI semantic-docid)",
                "sae_type": "batch_topk_training_sae",
            }, f, indent=2)

        with open(save_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f)

        save_file({"sparsity": log_feature_sparsity},
                  str(save_dir / "sparsity.safetensors"))

        with open(save_dir / "runner_cfg.json", "w") as f:
            json.dump({
                "layer": args.layer, "d_in": d_in, "d_sae": args.d_sae,
                "k": args.k, "lr": args.lr, "batch_size": args.batch_size,
                "total_steps": args.total_steps, "scaling_factor": scaling_factor,
                "hook_name": hook_name, "world_size": world_size,
            }, f, indent=2)

        inference_dir = save_dir / "inference"
        sae.save_inference_model(inference_dir)
        save_file({"scaling_factor": torch.tensor(scaling_factor)},
                  str(inference_dir / "scaling_factor.safetensors"))
        save_file({"sparsity": log_feature_sparsity},
                  str(inference_dir / "sparsity.safetensors"))

        print(f"\n  Saved to {save_dir}")
        print(f"    - sae_weights.safetensors        (training weights)")
        print(f"    - sae_config.json                (training config)")
        print(f"    - training_metrics.json")
        print(f"    - sparsity.safetensors")
        print(f"    - runner_cfg.json")
        print(f"  Inference checkpoint: {inference_dir}")
        print(f"    - sae_weights.safetensors        (JumpReLU format, load via SAE.load_from_disk)")
        print(f"    - cfg.json                       (SAELens compatible)")
        print(f"    - scaling_factor.safetensors     (must apply before encode)")
        print(f"    - sparsity.safetensors")

    if use_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train SAE on cached activations")
    parser.add_argument("--cache_dir", type=str, required=True, help="Pre-collected activation cache dir")
    parser.add_argument("--layer", type=int, required=True, help="Decoder layer to train SAE on")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dsi_sae_semantic")
    parser.add_argument("--d_sae", type=int, default=16384)
    parser.add_argument("--k", type=float, default=100.0)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1024)  # 1024 * 8 = 8196
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--n_batches_for_norm", type=int, default=50)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="sae-semantic-2")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    train(args)


if __name__ == "__main__":
    main()
