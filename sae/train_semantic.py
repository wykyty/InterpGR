"""
Train BatchTopK SAE on decoder MLP layers of a fine-tuned DSI Semantic-Docid model.

Supports multi-GPU parallel activation collection: all GPUs run T5-large forward
passes concurrently, feeding activations to the main GPU for SAE training.

Usage (single layer, 8 GPUs):
    python sae/train_semantic.py \
        --checkpoint out/dsi-semantic/49.pt \
        --data_path dataset/nq320k/dev.json \
        --save_dir checkpoints/dsi_sae_semantic \
        --layers 0 --n_gpus 8
"""

import argparse
import json
import os
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformer_lens import HookedEncoderDecoder
from transformers import AutoTokenizer, T5ForConditionalGeneration

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput


# ---------------------------------------------------------------------------
# Model loading (same for all GPUs)
# ---------------------------------------------------------------------------

def load_dsi_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """Load a DSI semantic checkpoint into HookedEncoderDecoder."""
    import transformer_lens.loading_from_pretrained as loading
    from transformer_lens.pretrained.weight_conversions.t5 import convert_t5_weights

    print(f"Loading DSI semantic checkpoint from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
    num_new_tokens = 30
    tokenizer.add_tokens([f'${i}$' for i in range(num_new_tokens)])
    target_vocab = len(tokenizer)

    hf_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")
    hf_model.resize_token_embeddings(target_vocab)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.eval()

    cfg = loading.get_pretrained_model_config(
        "google-t5/t5-large", fold_ln=False, device=None, n_devices=1,
    )
    cfg.d_vocab = target_vocab
    cfg.d_vocab_out = target_vocab

    tl_state_dict = convert_t5_weights(hf_model, cfg)

    model = HookedEncoderDecoder(cfg, tokenizer=tokenizer, move_to_device=False)
    model.load_state_dict(tl_state_dict, strict=False)

    if device:
        model.to(device)
    model.eval()

    print(f"  d_model={model.cfg.d_model}  n_layers={model.cfg.n_layers}  vocab={target_vocab}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_queries(data_path: str) -> list[str]:
    """Load queries from nq320k/dev.json format: [[query, doc_id], ...]"""
    data = json.load(open(data_path))
    queries = []
    for item in data:
        q = item[0]
        if isinstance(q, list):
            q = q[0]
        if q and str(q).strip():
            queries.append(str(q))
    print(f"Loaded {len(queries)} queries from {data_path}")
    return queries


# ---------------------------------------------------------------------------
# Multi-GPU activation collection
# ---------------------------------------------------------------------------

SENTINEL = None  # poison pill to stop workers


def _gpu_worker(
    gpu_id: int,
    checkpoint_path: str,
    queries: list[str],
    hook_name: str,
    d_in: int,
    context_size: int,
    batch_size: int,
    queue: Queue,
):
    """Worker process: loads T5 on its own GPU, continuously collects activations."""
    device = f"cuda:{gpu_id}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model, tokenizer = load_dsi_model(checkpoint_path, device)
    idx = len(queries) // 8 * gpu_id  # stagger starting points

    while True:
        msg = queue.get()
        if msg is SENTINEL:
            break

        collected = []
        total_tokens = 0
        while total_tokens < batch_size:
            if idx >= len(queries):
                idx = 0
            query = queries[idx]
            idx += 1

            enc_tokens = tokenizer(
                query, return_tensors="pt", truncation=True,
                max_length=context_size, padding=False,
            )
            dec_input = torch.full(
                (1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device,
            )

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    enc_tokens.input_ids,
                    one_zero_attention_mask=enc_tokens.attention_mask,
                    decoder_input=dec_input,
                    names_filter=lambda name: name == hook_name,
                )

            acts = cache[hook_name].squeeze(0).float()
            collected.append(acts)
            total_tokens += acts.shape[0]

        all_acts = torch.cat(collected, dim=0)
        if all_acts.shape[0] > batch_size:
            indices = torch.randperm(all_acts.shape[0])[:batch_size]
            all_acts = all_acts[indices]
        queue.put(all_acts.cpu())  # send to main process


def start_gpu_workers(n_gpus, checkpoint_path, queries, hook_name, d_in, context_size, batch_size):
    """Launch GPU workers and return (queue, processes)."""
    queue = mp.Queue(maxsize=n_gpus * 2)
    procs = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=_gpu_worker,
            args=(gpu_id, checkpoint_path, queries, hook_name, d_in, context_size, batch_size, queue),
            daemon=True,
        )
        p.start()
        procs.append(p)
    return queue, procs


def request_batch(queue, n_gpus):
    """Signal all workers to produce one batch each."""
    for _ in range(n_gpus):
        queue.put("go")


def collect_batch(queue, device, d_in, target_tokens):
    """Receive one batch from queue, move to device."""
    acts = queue.get()
    if isinstance(acts, str):  # shouldn't happen, but safety
        return torch.empty(0, d_in, device=device)
    return acts.to(device)


def stop_workers(queue, procs, n_gpus):
    """Send poison pill to all workers and join."""
    for _ in range(n_gpus):
        queue.put(SENTINEL)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()


# ---------------------------------------------------------------------------
# Single layer training
# ---------------------------------------------------------------------------

def train_layer(
    layer: int,
    checkpoint_path: str,
    queries: list[str],
    d_in: int,
    device: str,
    save_dir: Path,
    n_gpus: int = 8,
    d_sae: int = 16384,
    k: float = 100.0,
    total_steps: int = 10_000,
    batch_size: int = 4096,
    lr: float = 5e-5,
    context_size: int = 32,
    log_every: int = 100,
    n_batches_for_norm_estimate: int = 50,
    wandb_project: str = "batchtopk-sae-semantic",
    wandb_entity: str | None = None,
) -> dict:
    """Train a BatchTopK SAE on one decoder MLP layer using multi-GPU activation collection."""
    hook_name = f"decoder.{layer}.hook_mlp_out"
    layer_dir = save_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Layer {layer}: {hook_name}  ({n_gpus} GPUs for activation collection)")
    print(f"{'='*60}")

    # Start GPU workers
    queue, procs = start_gpu_workers(
        n_gpus, checkpoint_path, queries, hook_name, d_in, context_size, batch_size,
    )

    # Create SAE on main device
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in, d_sae=d_sae, k=k,
        aux_loss_coefficient=1.0,
        rescale_acts_by_decoder_norm=True,
        topk_threshold_lr=0.01,
        apply_b_dec_to_input=False,
        normalize_activations="expected_average_only_in",
        decoder_init_norm=0.1,
        device=device, dtype="float32",
    )
    sae = BatchTopKTrainingSAE(sae_cfg).to(device)

    # Estimate scaling factor
    print("Estimating activation scaling factor...")
    norms = []
    for i in range(n_batches_for_norm_estimate):
        request_batch(queue, n_gpus)
        for _ in range(n_gpus):
            batch = collect_batch(queue, device, d_in, batch_size)
            if batch.shape[0] > 0:
                norms.append(batch.norm(dim=-1).mean().item())
        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1}/{n_batches_for_norm_estimate}")

    mean_norm = np.mean(norms)
    scaling_factor = (d_in ** 0.5) / mean_norm
    sae.scaling_factor = torch.tensor(scaling_factor, device=device)
    print(f"Scaling factor: {scaling_factor:.4f}")

    # Wandb
    wandb.init(
        project=wandb_project, entity=wandb_entity,
        name=f"layer_{layer}",
        config={
            "layer": layer, "hook_name": hook_name,
            "d_in": d_in, "d_sae": d_sae, "k": k,
            "lr": lr, "batch_size": batch_size,
            "total_steps": total_steps, "scaling_factor": scaling_factor,
            "n_gpus": n_gpus,
        },
        reinit=True,
    )

    # Training loop
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sae.train()
    pbar = tqdm(range(total_steps), desc=f"Layer {layer}")
    feature_act_freq = torch.zeros(d_sae, device=device)
    total_tokens_seen = 0
    metrics = {
        "step": [], "total_loss": [], "mse_loss": [], "aux_loss": [],
        "explained_variance": [], "l0": [], "topk_threshold": [],
    }

    for step in pbar:
        # Request activations from all GPUs, then collect
        request_batch(queue, n_gpus)
        batches = []
        for _ in range(n_gpus):
            batch = collect_batch(queue, device, d_in, batch_size)
            if batch.shape[0] > 0:
                batches.append(batch)

        if not batches:
            continue
        sae_in = torch.cat(batches, dim=0) * sae.scaling_factor

        step_output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=sae_in, coefficients={},
                dead_neuron_mask=None,
                n_training_steps=step,
                is_logging_step=(step % log_every == 0),
            )
        )

        optimizer.zero_grad()
        step_output.loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

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

        wandb.log({
            "train/total_loss": step_output.loss.item(),
            "train/mse_loss": step_output.losses["mse_loss"].item(),
            "train/aux_loss": step_output.losses["auxiliary_reconstruction_loss"].item(),
            "train/explained_variance": ev,
            "train/l0": l0,
            "train/topk_threshold": sae.topk_threshold.item(),
        }, step=step)

        if step % log_every == 0:
            pbar.set_postfix({
                "loss": f"{step_output.loss.item():.4f}",
                "ev": f"{ev.item():.4f}",
                "l0": f"{l0:.1f}",
            })

    pbar.close()
    wandb.finish()

    # Evaluation
    print("Evaluating...")
    sae.eval()
    all_feature_acts, all_reconstructions, all_inputs = [], [], []
    with torch.no_grad():
        for _ in range(10):
            request_batch(queue, n_gpus)
            batches = []
            for _ in range(n_gpus):
                batch = collect_batch(queue, device, d_in, batch_size)
                if batch.shape[0] > 0:
                    batches.append(batch)
            if not batches:
                continue
            scaled = torch.cat(batches, dim=0) * sae.scaling_factor
            feature_acts, _ = sae.encode_with_hidden_pre(scaled)
            reconstruction = sae.decode(feature_acts)
            all_feature_acts.append(feature_acts)
            all_reconstructions.append(reconstruction)
            all_inputs.append(scaled)

    feature_acts = torch.cat(all_feature_acts, dim=0)
    reconstructions = torch.cat(all_reconstructions, dim=0)
    inputs = torch.cat(all_inputs, dim=0)

    per_token_mse = (reconstructions - inputs).pow(2).sum(dim=-1)
    total_var = (inputs - inputs.mean(0)).pow(2).sum(dim=-1)
    explained_variance = 1 - per_token_mse.mean() / (total_var.mean() + 1e-8)
    active = feature_acts.bool().float()
    l0 = active.sum(-1).mean()
    feature_density = active.mean(0)
    dead_features = (feature_density < 1e-6).sum().item()

    eval_result = {
        "layer": layer,
        "explained_variance": explained_variance.item(),
        "l0": l0.item(),
        "dead_features": dead_features,
        "dead_ratio": dead_features / feature_acts.shape[-1],
        "topk_threshold": sae.topk_threshold.item(),
    }

    print(f"  EV={eval_result['explained_variance']:.4f}  "
          f"L0={eval_result['l0']:.1f}  "
          f"Dead={dead_features}/{feature_acts.shape[-1]}")

    # Stop workers
    stop_workers(queue, procs, n_gpus)

    # Save
    log_feature_sparsity = torch.log10(feature_act_freq / total_tokens_seen + 1e-10)

    save_file({
        "W_enc": sae.W_enc.data, "W_dec": sae.W_dec.data,
        "b_enc": sae.b_enc.data, "b_dec": sae.b_dec.data,
        "scaling_factor": sae.scaling_factor, "topk_threshold": sae.topk_threshold,
    }, str(layer_dir / "sae_weights.safetensors"))

    with open(layer_dir / "sae_config.json", "w") as f:
        json.dump({
            "d_in": d_in, "d_sae": d_sae, "k": k,
            "hook_name": hook_name, "layer": layer,
            "model_name": "google-t5/t5-large (DSI semantic-docid)",
            "sae_type": "batch_topk_training_sae",
        }, f, indent=2)

    with open(layer_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f)

    save_file({"sparsity": log_feature_sparsity}, str(layer_dir / "sparsity.safetensors"))

    with open(layer_dir / "runner_cfg.json", "w") as f:
        json.dump({
            "layer": layer, "hook_name": hook_name,
            "d_in": d_in, "d_sae": d_sae, "k": k,
            "lr": lr, "batch_size": batch_size, "total_steps": total_steps,
            "scaling_factor": scaling_factor, "n_gpus": n_gpus,
        }, f, indent=2)

    inference_dir = layer_dir / "inference"
    sae.save_inference_model(inference_dir)
    save_file({"scaling_factor": torch.tensor(scaling_factor)},
              str(inference_dir / "scaling_factor.safetensors"))
    save_file({"sparsity": log_feature_sparsity},
              str(inference_dir / "sparsity.safetensors"))

    print(f"  Saved to {layer_dir}")
    return eval_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train SAE on DSI semantic decoder MLP layers")
    parser.add_argument("--checkpoint", type=str, required=True, help="DSI semantic checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, help="nq320k/dev.json path")
    parser.add_argument("--save_dir", type=str, default="checkpoints/dsi_sae_semantic")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer numbers to train (default: all 0-23)")
    parser.add_argument("--layer_range", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Layer range [START, END) to train")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs for activation collection")
    parser.add_argument("--d_sae", type=int, default=16384)
    parser.add_argument("--k", type=float, default=100.0)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--context_size", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="batchtopk-sae-semantic")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    queries = load_queries(args.data_path)
    d_in = 1024  # T5-large d_model

    if args.layers:
        layers = args.layers
    elif args.layer_range:
        layers = list(range(args.layer_range[0], args.layer_range[1]))
    else:
        layers = list(range(24))
    save_dir = Path(args.save_dir)

    print(f"\nWill train SAE on {len(layers)} layers: {layers}")
    print(f"  d_in={d_in}, d_sae={args.d_sae}, k={args.k}")
    print(f"  total_steps={args.total_steps}, batch_size={args.batch_size}")
    print(f"  n_gpus={args.n_gpus} (parallel activation collection)")

    all_results = []
    for layer in layers:
        result = train_layer(
            layer=layer, checkpoint_path=args.checkpoint,
            queries=queries, d_in=d_in, device=device,
            save_dir=save_dir, n_gpus=args.n_gpus,
            d_sae=args.d_sae, k=args.k,
            total_steps=args.total_steps, batch_size=args.batch_size,
            lr=args.lr, context_size=args.context_size,
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'EV':>8} {'L0':>8} {'Dead':>8} {'Dead%':>8}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['layer']:<8} {r['explained_variance']:>8.4f} {r['l0']:>8.1f} "
              f"{r['dead_features']:>8} {r['dead_ratio']:>7.1%}")

    best = max(all_results, key=lambda x: x["explained_variance"])
    print(f"\nBest layer: {best['layer']} (EV={best['explained_variance']:.4f})")


if __name__ == "__main__":
    main()
