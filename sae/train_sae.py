import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput


class ActivationLoader:
    """仿 SAELens 逻辑的高效、无放回激活值加载器"""
    def __init__(self, cache_dir: str, layer: int):
        cache_path = Path(cache_dir) / f"layer_{layer}.safetensors"
        print(f"Loading cached activations from {cache_path}...")
        data = load_file(str(cache_path))
        self.activations = data["activations"]  # (total_tokens, d_in)
        self.total_tokens = self.activations.shape[0]
        self.d_in = self.activations.shape[1]
        self.reset_stream()
        print(f"Loaded {self.total_tokens} tokens, d_in={self.d_in}")

    def reset_stream(self):
        print("Dataset boundary reached. Shuffling indices...")
        self.indices = torch.randperm(self.total_tokens)
        self.current_ptr = 0

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.current_ptr + batch_size > self.total_tokens:
            self.reset_stream()
        batch_indices = self.indices[self.current_ptr : self.current_ptr + batch_size]
        self.current_ptr += batch_size
        return self.activations[batch_indices].to(device)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load cached activations
    loader = ActivationLoader(args.cache_dir, args.layer)
    d_in = loader.d_in

    # Create SAE
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in, 
        d_sae=args.d_sae, 
        k=args.k,
        aux_loss_coefficient=1.0, 
        
        # use_error_term_for_dead_neurons=True, 
        # dead_feature_threshold=1e-6,

        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.01, 
        apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in", 
        decoder_init_norm=0.1,
        device=str(device), dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    # Estimate scaling factor
    print("Estimating activation scaling factor...")
    norms = []
    for _ in range(args.n_batches_for_norm):
        batch = loader.sample(args.batch_size, device)
        norms.append(batch.norm(dim=-1).mean().item())

    mean_norm = np.mean(norms)
    scaling_factor = (d_in ** 0.5) / mean_norm
    sae.scaling_factor = torch.tensor(scaling_factor, device=device)
    print(f"Scaling factor: {scaling_factor:.4f}")

    # Wandb
    if not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"layer_{args.layer}_single_gpu",
            config={
                "layer": args.layer, "d_in": d_in, "d_sae": args.d_sae,
                "k": args.k, "lr": args.lr, "batch_size": args.batch_size,
                "total_steps": args.total_steps, "scaling_factor": scaling_factor,
                "world_size": 1,
            },
            reinit=True,
        )

    # Optimizer & LR Scheduler (引入余弦退火衰减)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-6)
    
    # 增加学习率预热
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1500,        # 预热步数
        num_training_steps=args.total_steps # 总训练步数
    )

    sae.train()

    pbar = tqdm(range(args.total_steps), desc=f"Layer {args.layer}")
    feature_act_freq = torch.zeros(args.d_sae, device=device)
    total_tokens_seen = 0
    metrics = {
        "step": [], "total_loss": [], "mse_loss": [], "aux_loss": [],
        "explained_variance": [], "l0": [], "topk_threshold": [],
    }

    for step in pbar:
        sae_in = loader.sample(args.batch_size, device) * sae.scaling_factor

        step_output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=sae_in, coefficients={},
                dead_neuron_mask=None, n_training_steps=step,
                is_logging_step=(step % args.log_every == 0),
            )
        )

        optimizer.zero_grad()
        step_output.loss.backward()
        
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step() # 更新学习率

        with torch.no_grad():
            l0 = step_output.feature_acts.bool().float().sum(-1).mean().item()
            per_token_l2 = (step_output.sae_out - sae_in).pow(2).sum(dim=-1)
            total_var = (sae_in - sae_in.mean(0)).pow(2).sum(dim=-1)
            ev = 1 - per_token_l2.mean() / (total_var.mean() + 1e-8)
            n_tokens = sae_in.shape[0]
            feature_act_freq += (step_output.feature_acts > 0).float().sum(dim=0)
            total_tokens_seen += n_tokens

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
                "train/explained_variance": ev.item(), "train/l0": l0,
                "train/topk_threshold": sae.topk_threshold.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=step)

        if step % args.log_every == 0:
            pbar.set_postfix({
                "loss": f"{step_output.loss.item():.4f}",
                "ev": f"{ev.item():.4f}", "l0": f"{l0:.1f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

    pbar.close()

    # Fold scaling factor into weights (与 SAELens SAETrainer 保持一致)
    # 之后 SAE 直接接受 raw activation，无需手动乘 scaling_factor
    sae.fold_activation_norm_scaling_factor(scaling_factor)

    if not args.no_wandb:
        import wandb
        wandb.finish()

    # Save
    save_dir = Path(args.save_dir) / f"layer_{args.layer}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_feature_sparsity = torch.log10(feature_act_freq / total_tokens_seen + 1e-10)
    hook_name = f"decoder.{args.layer}.hook_mlp_out"

    save_file({
        "W_enc": sae.W_enc.data, "W_dec": sae.W_dec.data,
        "b_enc": sae.b_enc.data, "b_dec": sae.b_dec.data,
        "topk_threshold": sae.topk_threshold,
    }, str(save_dir / "sae_weights.safetensors"))

    with open(save_dir / "sae_config.json", "w") as f:
        json.dump({"d_in": d_in, "d_sae": args.d_sae, "k": args.k, "hook_name": hook_name, "layer": args.layer, "model_name": "google-t5/t5-large (DSI semantic-docid)", "sae_type": "batch_topk_training_sae"}, f, indent=2)

    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f)

    save_file({"sparsity": log_feature_sparsity}, str(save_dir / "sparsity.safetensors"))
    
    inference_dir = save_dir / "inference"
    sae.save_inference_model(inference_dir)
    save_file({"sparsity": log_feature_sparsity}, str(inference_dir / "sparsity.safetensors"))
    print(f"Saved successfully to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train SAE on cached activations")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/dsi_sae_semantic")
    parser.add_argument("--d_sae", type=int, default=16384)
    parser.add_argument("--k", type=float, default=100.0)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32768)  # 单卡大 Batch 模拟 8 卡总和
    parser.add_argument("--lr", type=float, default=3e-4)        # 对应大 Batch 的标准 LR
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--n_batches_for_norm", type=int, default=50)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sae-semantic-2")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    train(args)

if __name__ == "__main__":
    main()