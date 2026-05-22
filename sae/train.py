"""
Training a BatchTopK SAE on T5-large Decoder with SAELens

BatchTopK differs from standard (ReLU + L1) SAEs:
- Uses global top-k selection across the entire batch (not per-sample)
- Sparsity is controlled by k (average active features) instead of L1 coefficient
- Uses auxiliary loss to recover dead neurons instead of L1 warm-up
- At inference time, converts to JumpReLU via a learned threshold (EMA)

Since SAELens's SAETrainingRunner does not support HookedEncoderDecoder, we:
1. Load T5-large via TransformerLens's HookedEncoderDecoder
2. Manually collect activations from a decoder hook point
3. Train BatchTopKTrainingSAE with a manual training loop
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import wandb
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformer_lens import HookedEncoderDecoder

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput

@dataclass
class Config:
    # SAE architecture
    d_in: int = 1024          # T5-large d_model
    d_sae: int = 16384        # SAE feature count (16x expansion)
    # Hook point
    hook_name: str = "decoder.12.hook_mlp_out"
    # Data
    context_size: int = 128   # max encoder token length
    target_size: int = 64     # max decoder token length
    batch_size: int = 4096    # number of activation tokens per training step
    # BatchTopK specific
    k: float = 100.0          # avg active features per sample (float!)
    aux_loss_coefficient: float = 1.0
    rescale_acts_by_decoder_norm: bool = True
    # Training
    total_steps: int = 10_000
    lr: float = 5e-5  # BatchTopK 常用值
    log_every: int = 100
    n_batches_for_norm_estimate: int = 50
    # Checkpoint
    save_dir: str = "checkpoints/batchtopk_sae_t5_large_decoder"
    # Wandb
    wandb_project: str = "batchtopk-sae"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None



@torch.no_grad()
def collect_activations_batch(dataset_iter, model, tokenizer, cfg, device, target_tokens=4096):
    """Collect decoder activations from XSum document-summary pairs."""
    collected = []
    total_tokens = 0

    while total_tokens < target_tokens:
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break

        source = sample["document"]
        target = sample["summary"]

        if not source.strip() or not target.strip():
            continue

        enc_tokens = tokenizer(
            source, return_tensors="pt", truncation=True,
            max_length=cfg.context_size, padding=False,
        )
        dec_tokens = tokenizer(
            target, return_tensors="pt", truncation=True,
            max_length=cfg.target_size, padding=False,
        )

        _, cache = model.run_with_cache(
            enc_tokens.input_ids,
            attention_mask=enc_tokens.attention_mask,
            decoder_input=dec_tokens.input_ids,
            names_filter=lambda name: name == cfg.hook_name,
        )

        acts = cache[cfg.hook_name].squeeze(0).float()
        collected.append(acts)
        total_tokens += acts.shape[0]

    if not collected:
        return torch.empty(0, cfg.d_in, device=device)

    all_acts = torch.cat(collected, dim=0)
    if all_acts.shape[0] > target_tokens:
        indices = torch.randperm(all_acts.shape[0])[:target_tokens]
        all_acts = all_acts[indices]
    return all_acts

def main():
    cfg = Config()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("Loading T5-large via TransformerLens...")
    model = HookedEncoderDecoder.from_pretrained("google-t5/t5-large")
    model.eval()
    tokenizer = model.tokenizer

    print(f"Model: google-t5/t5-large")
    print(f"  d_model = {model.cfg.d_model}")
    print(f"  d_mlp   = {model.cfg.d_mlp}")
    print(f"  n_heads = {model.cfg.n_heads}")
    print(f"  n_layers = {model.cfg.n_layers}")

    # ------------------------------------------------------------------
    # 2. Load dataset (streaming)
    # ------------------------------------------------------------------
    print("\nLoading XSum dataset (streaming)...")
    dataset = load_dataset("EdinburghNLP/xsum", split="train", streaming=True)
    dataset_iter = iter(dataset)

    # ------------------------------------------------------------------
    # 3. Create SAE
    # ------------------------------------------------------------------
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        k=cfg.k,
        aux_loss_coefficient=cfg.aux_loss_coefficient,
        rescale_acts_by_decoder_norm=cfg.rescale_acts_by_decoder_norm,
        topk_threshold_lr=0.01,
        apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in",
        decoder_init_norm=0.1,
        device=device,
        dtype="float32",
    )

    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    n_params = sum(p.numel() for p in sae.parameters())
    print(f"\nBatchTopK SAE initialized: {n_params:,} parameters")
    print(f"  W_enc: {sae.W_enc.shape}")
    print(f"  W_dec: {sae.W_dec.shape}")
    print(f"  b_enc: {sae.b_enc.shape}")
    print(f"  b_dec: {sae.b_dec.shape}")
    print(f"  topk_threshold: {sae.topk_threshold}")
    print(f"  k={cfg.k}, d_in={cfg.d_in}, d_sae={cfg.d_sae}")

    # ------------------------------------------------------------------
    # 4. Estimate activation scaling factor
    # ------------------------------------------------------------------
    print("\nEstimating activation scaling factor...")
    norms = []
    for i in range(cfg.n_batches_for_norm_estimate):
        batch = collect_activations_batch(
            dataset_iter, model, tokenizer, cfg, device, target_tokens=cfg.batch_size
        )
        if batch.shape[0] > 0:
            norms.append(batch.norm(dim=-1).mean().item())
        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1}/{cfg.n_batches_for_norm_estimate}")

    mean_norm = np.mean(norms)
    scaling_factor = (cfg.d_in ** 0.5) / mean_norm
    sae.scaling_factor = torch.tensor(scaling_factor, device=device)

    print(f"Mean activation L2 norm: {mean_norm:.4f}")
    print(f"Target norm (sqrt(d_in)): {cfg.d_in ** 0.5:.4f}")
    print(f"Scaling factor: {scaling_factor:.4f}")

    # ------------------------------------------------------------------
    # 5. Wandb
    # ------------------------------------------------------------------
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name,
        config={
            "d_in": cfg.d_in,
            "d_sae": cfg.d_sae,
            "k": cfg.k,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "total_steps": cfg.total_steps,
            "hook_name": cfg.hook_name,
            "model_name": "google-t5/t5-large",
            "scaling_factor": scaling_factor,
        },
    )

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    metrics = {
        "step": [], "total_loss": [], "mse_loss": [], "aux_loss": [],
        "explained_variance": [], "l0": [], "topk_threshold": [],
    }

    print(f"\nStarting BatchTopK training for {cfg.total_steps} steps...")
    print(f"  Hook: {cfg.hook_name}")
    print(f"  Dataset: XSum (streaming)\n")

    sae.train()
    pbar = tqdm(range(cfg.total_steps), desc="Training BatchTopK SAE")
    feature_act_freq = torch.zeros(cfg.d_sae, device=device)
    total_tokens_seen = 0

    for step in pbar:
        sae_in = collect_activations_batch(
            dataset_iter, model, tokenizer, cfg, device, target_tokens=cfg.batch_size
        )
        if sae_in.shape[0] == 0:
            continue

        sae_in = sae_in * sae.scaling_factor

        step_output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=sae_in,
                coefficients={},
                dead_neuron_mask=None,
                n_training_steps=step,
                is_logging_step=(step % cfg.log_every == 0),
            )
        )

        optimizer.zero_grad()
        step_output.loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            l0 = step_output.feature_acts.bool().float().sum(-1).mean().item()
            per_token_l2 = (step_output.sae_out - sae_in).pow(2).sum(dim=-1)
            total_var = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
            ev = 1 - per_token_l2.mean() / (total_var.mean() + 1e-8)

            # Track per-feature activation frequency
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

        # Wandb logging
        wandb.log({
            "train/total_loss": step_output.loss.item(),
            "train/mse_loss": step_output.losses["mse_loss"].item(),
            "train/aux_loss": step_output.losses["auxiliary_reconstruction_loss"].item(),
            "train/explained_variance": ev,
            "train/l0": l0,
            "train/topk_threshold": sae.topk_threshold.item(),
        }, step=step)

        if step % cfg.log_every == 0:
            pbar.set_postfix({
                "loss": f"{step_output.loss.item():.4f}",
                "mse": f"{step_output.losses['mse_loss'].item():.4f}",
                "aux": f"{step_output.losses['auxiliary_reconstruction_loss'].item():.4f}",
                "ev": f"{ev.item():.4f}",
                "l0": f"{l0:.1f}",
            })

    pbar.close()
    print("\nTraining complete!")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating...")
    sae.eval()

    all_feature_acts = []
    all_reconstructions = []
    all_inputs = []

    with torch.no_grad():
        for _ in range(10):
            batch = collect_activations_batch(
                dataset_iter, model, tokenizer, cfg, device, target_tokens=cfg.batch_size
            )
            if batch.shape[0] == 0:
                continue
            scaled = batch * sae.scaling_factor
            feature_acts, hidden_pre = sae.encode_with_hidden_pre(scaled)
            reconstruction = sae.decode(feature_acts)

            all_feature_acts.append(feature_acts)
            all_reconstructions.append(reconstruction)
            all_inputs.append(scaled)

    feature_acts = torch.cat(all_feature_acts, dim=0)
    reconstructions = torch.cat(all_reconstructions, dim=0)
    inputs = torch.cat(all_inputs, dim=0)

    per_token_mse = (reconstructions - inputs).pow(2).sum(dim=-1)
    total_variance = (inputs - inputs.mean(0)).pow(2).sum(dim=-1)
    explained_variance = 1 - per_token_mse.mean() / (total_variance.mean() + 1e-8)

    active = feature_acts.bool().float()
    l0 = active.sum(-1).mean()
    feature_density = active.mean(0)
    dead_features = (feature_density < 1e-6).sum().item()

    print(f"BatchTopK SAE Evaluation Results:")
    print(f"  k (target avg active): {cfg.k}")
    print(f"  L0 (actual avg active): {l0.item():.1f}")
    print(f"  Explained Variance: {explained_variance.item():.4f}")
    print(f"  Dead features: {dead_features}/{feature_acts.shape[-1]} ({dead_features / feature_acts.shape[-1] * 100:.1f}%)")
    print(f"  Mean feature density: {feature_density.mean().item():.6f}")
    print(f"  TopK threshold: {sae.topk_threshold.item():.6f}")

    # ------------------------------------------------------------------
    # 8. Logit Lens
    # ------------------------------------------------------------------
    print("\nLogit Lens: top tokens for random features")
    with torch.no_grad():
        embed = model.W_dec
        projection = sae.W_dec @ embed.T  # (d_sae, d_vocab)

        top_k = 5
        vals, inds = torch.topk(projection, top_k, dim=1)

        random_indices = torch.randint(0, projection.shape[0], (10,))
        for idx in random_indices:
            feat = idx.item()
            tokens = [model.to_string(i) for i in inds[feat]]
            scores = [f"{v:.2f}" for v in vals[feat]]
            token_strs = [f"'{t}' ({s})" for t, s in zip(tokens, scores)]
            print(f"  Feature {feat:5d}: {', '.join(token_strs)}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "W_enc": sae.W_enc.data,
            "W_dec": sae.W_dec.data,
            "b_enc": sae.b_enc.data,
            "b_dec": sae.b_dec.data,
            "scaling_factor": sae.scaling_factor,
            "topk_threshold": sae.topk_threshold,
        },
        str(save_dir / "sae_weights.safetensors"),
    )

    with open(save_dir / "sae_config.json", "w") as f:
        json.dump({
            "d_in": cfg.d_in,
            "d_sae": cfg.d_sae,
            "k": cfg.k,
            "hook_name": cfg.hook_name,
            "aux_loss_coefficient": cfg.aux_loss_coefficient,
            "rescale_acts_by_decoder_norm": cfg.rescale_acts_by_decoder_norm,
            "model_name": "google-t5/t5-large",
            "sae_type": "batch_topk_training_sae",
        }, f, indent=2)

    with open(save_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f)

    # Sparsity (per-feature activation frequency, log10, compatible with SAELens)
    log_feature_sparsity = torch.log10(feature_act_freq / total_tokens_seen + 1e-10)
    save_file({"sparsity": log_feature_sparsity}, str(save_dir / "sparsity.safetensors"))

    # Runner config (compatible with SAELens)
    with open(save_dir / "runner_cfg.json", "w") as f:
        json.dump({
            "model_name": "google-t5/t5-large",
            "d_in": cfg.d_in,
            "d_sae": cfg.d_sae,
            "k": cfg.k,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "total_steps": cfg.total_steps,
            "hook_name": cfg.hook_name,
            "context_size": cfg.context_size,
            "target_size": cfg.target_size,
            "aux_loss_coefficient": cfg.aux_loss_coefficient,
            "rescale_acts_by_decoder_norm": cfg.rescale_acts_by_decoder_norm,
            "scaling_factor": scaling_factor,
        }, f, indent=2)

    # Inference format (JumpReLU, loadable via SAE.load_from_disk)
    inference_dir = save_dir / "inference"
    sae.save_inference_model(inference_dir)
    # scaling_factor is not part of SAE state_dict, save separately
    save_file(
        {"scaling_factor": torch.tensor(scaling_factor)},
        str(inference_dir / "scaling_factor.safetensors"),
    )
    # Copy sparsity to inference dir for SAELens compatibility
    save_file({"sparsity": log_feature_sparsity}, str(inference_dir / "sparsity.safetensors"))

    wandb.finish()

    print(f"\nSaved to {save_dir}")
    print(f"  - sae_weights.safetensors        (training weights)")
    print(f"  - sae_config.json                (training config)")
    print(f"  - training_metrics.json")
    print(f"  - sparsity.safetensors")
    print(f"  - runner_cfg.json")
    print(f"\nInference checkpoint: {inference_dir}")
    print(f"  - sae_weights.safetensors        (JumpReLU format, load via SAE.load_from_disk)")
    print(f"  - cfg.json                       (SAELens compatible)")
    print(f"  - scaling_factor.safetensors     (must apply before encode)")
    print(f"  - sparsity.safetensors")


if __name__ == "__main__":
    main()
