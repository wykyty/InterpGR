"""Free-running SAE logit contribution and activation-patching analysis.

This experiment deliberately does *not* read the teacher-forced activation
cache.  For every matched S_r/F_r pair, the decoder input is reconstructed
from the tokens produced by the original free-running generation.  At the
first-error position the generated prefix must equal the gold prefix; this is
checked before any activation is accepted.

The analysis has two stages:

1. Select stable Success-/Failure-selective layer-latents on a discovery split
   using sign consistency across grouped folds.
2. On held-out pairs, report each selected latent's direct gold-vs-wrong logit
   attribution and causally patch paired Success latent values into the Failure
   run.  A target-token-stratified shuffled Success source is the negative
   control.

Only the current generation position is patched.  The patched token and all
subsequent tokens are then greedily generated under the corpus DocID trie.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.common import load_sae, resolve_device
from data_process.build_generation_failure_sets import load_state_dict


GROUP_NAMES = tuple(f"S{position}_vs_F{position}" for position in range(1, 5))


@dataclass
class PairGroup:
    name: str
    position: int
    pair_rows: list[dict[str, Any]]
    successes: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    success_ids: np.ndarray
    failure_ids: np.ndarray
    discovery_indices: np.ndarray
    evaluation_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint", type=Path, default=Path("out/dsi-semantic-bert/99.pt")
    )
    parser.add_argument("--base-model", default="google-t5/t5-large")
    parser.add_argument("--sae-root", type=Path, default=Path("out/sae_train_8x"))
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("dataset/nq320k_id/id.semantic.bert.json"),
    )
    parser.add_argument(
        "--outcome-root",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path(
            "dataset/nq320k/generation_outcomes/matched_success_controls"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/success_failure_logit_patching"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--evaluation-fraction", type=float, default=0.30)
    parser.add_argument("--stability-folds", type=int, default=5)
    parser.add_argument("--min-fold-consistency", type=float, default=0.80)
    parser.add_argument("--top-per-direction", type=int, default=5)
    parser.add_argument(
        "--patch-counts", type=int, nargs="+", default=[1, 3, 5, 10]
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--active-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sae-format",
        choices=("auto", "inference", "training"),
        default="inference",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=None,
        help="Optional deterministic prefix for smoke tests only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_id(record: dict[str, Any]) -> int:
    value = record.get("sample_index", record.get("query_id"))
    if value is None:
        raise KeyError("Outcome record has no sample_index/query_id")
    return int(value)


def load_pair_groups(args: argparse.Namespace) -> list[PairGroup]:
    success_records = read_json(args.outcome_root / "success.json")
    success_by_id = {record_id(row): row for row in success_records}
    rng = np.random.default_rng(args.seed)
    groups: list[PairGroup] = []
    for position in range(1, 5):
        failure_records = read_json(args.outcome_root / f"failure_{position}.json")
        failure_by_id = {record_id(row): row for row in failure_records}
        pair_rows = read_jsonl(args.matched_root / f"S{position}_pairs.jsonl")
        if args.max_pairs_per_group is not None:
            pair_rows = pair_rows[: args.max_pairs_per_group]
        successes = [success_by_id[int(row["success_query_id"])] for row in pair_rows]
        failures = [
            failure_by_id[int(row["matched_failure_query_id"])] for row in pair_rows
        ]
        success_ids = np.asarray([record_id(row) for row in successes], dtype=np.int64)
        failure_ids = np.asarray([record_id(row) for row in failures], dtype=np.int64)
        if len(pair_rows) < max(20, args.stability_folds * 2):
            raise RuntimeError(f"S{position}/F{position} has too few pairs")

        # Group split prevents a reused Success query from leaking from latent
        # selection into causal evaluation.
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.evaluation_fraction,
            random_state=args.seed + position,
        )
        discovery, evaluation = next(
            splitter.split(np.zeros(len(pair_rows)), groups=success_ids)
        )
        # Randomize row order within each subset without changing membership.
        discovery = rng.permutation(discovery)
        evaluation = rng.permutation(evaluation)
        groups.append(
            PairGroup(
                name=f"S{position}_vs_F{position}",
                position=position,
                pair_rows=pair_rows,
                successes=successes,
                failures=failures,
                success_ids=success_ids,
                failure_ids=failure_ids,
                discovery_indices=np.asarray(discovery, dtype=np.int64),
                evaluation_indices=np.asarray(evaluation, dtype=np.int64),
            )
        )
    return groups


def load_model(
    checkpoint: Path, base_model: str, device: torch.device
) -> tuple[T5ForConditionalGeneration, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_tokens([f"${index}$" for index in range(30)])
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    missing, unexpected = model.load_state_dict(
        load_state_dict(checkpoint), strict=False
    )
    if missing or unexpected:
        print(
            f"Checkpoint load: {len(missing)} missing, "
            f"{len(unexpected)} unexpected keys"
        )
    model.to(device).eval()
    return model, tokenizer


def value_to_token_map(tokenizer: Any) -> dict[int, int]:
    result = {}
    for value in range(30):
        token_id = int(tokenizer.convert_tokens_to_ids(f"${value}$"))
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer does not contain ${value}$")
        result[value] = token_id
    return result


def build_allowed_tokens(
    semantic_ids: Sequence[Sequence[int]],
    value_to_token: dict[int, int],
    eos_token_id: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    allowed: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for docid in semantic_ids:
        tokens = tuple(value_to_token[int(value)] for value in docid)
        for position, token in enumerate(tokens):
            allowed[tokens[:position]].add(token)
        allowed[tokens].add(int(eos_token_id))
    return {
        prefix: tuple(sorted(tokens)) for prefix, tokens in allowed.items()
    }


def gold_token_ids(
    record: dict[str, Any], value_to_token: dict[int, int]
) -> list[int]:
    return [value_to_token[int(value)] for value in record["gold_docid"]]


def generated_prefix(
    record: dict[str, Any],
    position: int,
    value_to_token: dict[int, int],
) -> list[int]:
    generated = [int(token) for token in record["generated_token_ids"]]
    prefix = generated[: position - 1]
    gold_prefix = gold_token_ids(record, value_to_token)[: position - 1]
    if prefix != gold_prefix:
        raise AssertionError(
            f"query {record_id(record)} at first-error position {position}: "
            "free-running prefix does not equal gold prefix"
        )
    return prefix


def validate_pair(
    success: dict[str, Any],
    failure: dict[str, Any],
    position: int,
    value_to_token: dict[int, int],
    eos_token_id: int,
) -> None:
    success_prefix = generated_prefix(success, position, value_to_token)
    failure_prefix = generated_prefix(failure, position, value_to_token)
    success_gold = gold_token_ids(success, value_to_token)
    failure_gold = gold_token_ids(failure, value_to_token)
    if success_gold[position - 1] != failure_gold[position - 1]:
        raise AssertionError("Matched Success/Failure target tokens differ")
    failure_generated = [int(token) for token in failure["generated_token_ids"]]
    failure_token = (
        failure_generated[position - 1]
        if position - 1 < len(failure_generated)
        else int(eos_token_id)
    )
    if failure_token == failure_gold[position - 1]:
        raise AssertionError("Failure record is not wrong at its first-error position")


def pair_has_exact_prefix(
    success: dict[str, Any],
    failure: dict[str, Any],
    position: int,
    value_to_token: dict[int, int],
) -> bool:
    return (
        generated_prefix(success, position, value_to_token)
        == generated_prefix(failure, position, value_to_token)
    )


def make_decoder_input(
    records: Sequence[dict[str, Any]],
    position: int,
    tokenizer: Any,
    value_to_token: dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    pad = int(tokenizer.pad_token_id)
    rows = [
        [pad, *generated_prefix(record, position, value_to_token)]
        for record in records
    ]
    expected = position
    if any(len(row) != expected for row in rows):
        raise RuntimeError("Unexpected decoder prefix length")
    return torch.tensor(rows, dtype=torch.long, device=device)


def encode_queries(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    context_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [str(record["query"]) for record in records],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=context_size,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@contextmanager
def capture_mlp_outputs(
    model: T5ForConditionalGeneration, layers: Iterable[int]
) -> Iterator[dict[int, torch.Tensor]]:
    cache: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        module = model.decoder.block[layer].layer[2].DenseReluDense

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            layer_id: int = layer,
        ) -> None:
            cache[layer_id] = output[:, -1, :]

        handles.append(module.register_forward_hook(hook))
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def assign_grouped_folds(
    success_ids: np.ndarray, indices: np.ndarray, folds: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(success_ids[indices])
    if len(unique) < folds:
        raise RuntimeError("Too few unique Success IDs for stability folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    mapping = {int(query_id): rank % folds for rank, query_id in enumerate(shuffled)}
    assignment = np.asarray(
        [mapping[int(success_ids[index])] for index in indices], dtype=np.int64
    )
    sizes = np.bincount(assignment, minlength=folds)
    if np.any(sizes == 0):
        raise RuntimeError("Empty stability fold")
    return assignment, sizes


@torch.inference_mode()
def select_stable_latents(
    group: PairGroup,
    model: T5ForConditionalGeneration,
    tokenizer: Any,
    saes: dict[int, torch.nn.Module],
    layers: Sequence[int],
    value_to_token: dict[int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices = group.discovery_indices
    fold_assignment, fold_sizes = assign_grouped_folds(
        group.success_ids,
        indices,
        args.stability_folds,
        args.seed + group.position * 100,
    )
    d_sae = {layer: int(saes[layer].W_dec.shape[0]) for layer in layers}
    success_counts = {
        layer: torch.zeros(
            (args.stability_folds, d_sae[layer]),
            dtype=torch.int32,
            device=device,
        )
        for layer in layers
    }
    failure_counts = {
        layer: torch.zeros_like(success_counts[layer]) for layer in layers
    }

    starts = range(0, len(indices), args.batch_size)
    for start in tqdm(
        starts,
        desc=f"Free-running stability [{group.name}]",
        unit="batch",
    ):
        batch_indices = indices[start : start + args.batch_size]
        successes = [group.successes[index] for index in batch_indices]
        failures = [group.failures[index] for index in batch_indices]
        for success, failure in zip(successes, failures):
            validate_pair(
                success,
                failure,
                group.position,
                value_to_token,
                int(tokenizer.eos_token_id),
            )
        records = [*successes, *failures]
        encoded = encode_queries(records, tokenizer, args.context_size, device)
        decoder_input = make_decoder_input(
            records, group.position, tokenizer, value_to_token, device
        )
        with capture_mlp_outputs(model, layers) as cache:
            model(
                **encoded,
                decoder_input_ids=decoder_input,
                use_cache=False,
                return_dict=True,
            )
        batch_folds = fold_assignment[start : start + len(batch_indices)]
        batch_size = len(batch_indices)
        for layer in layers:
            latent = saes[layer].encode(cache[layer].float())
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(
                    f"{group.name} layer {layer}: non-finite discovery latent"
                )
            active = latent > args.active_epsilon
            for fold in np.unique(batch_folds):
                mask = torch.from_numpy(batch_folds == fold).to(device)
                success_counts[layer][fold] += active[:batch_size][mask].sum(
                    dim=0, dtype=torch.int32
                )
                failure_counts[layer][fold] += active[batch_size:][mask].sum(
                    dim=0, dtype=torch.int32
                )
        del cache, encoded, decoder_input

    candidates: list[dict[str, Any]] = []
    n_discovery = len(indices)
    support_threshold = max(20, math.ceil(0.01 * n_discovery))
    for layer in layers:
        count_s = success_counts[layer].cpu().numpy().astype(np.float64)
        count_f = failure_counts[layer].cpu().numpy().astype(np.float64)
        full_s = count_s.sum(axis=0)
        full_f = count_f.sum(axis=0)
        separation = (full_s - full_f) / n_discovery
        fold_separation = (count_s - count_f) / fold_sizes[:, None]
        sign = np.sign(separation)
        consistency = np.mean(np.sign(fold_separation) == sign[None, :], axis=0)
        min_fold_abs = np.min(np.abs(fold_separation), axis=0)
        support = np.maximum(full_s, full_f)
        eligible = (
            (support >= support_threshold)
            & (sign != 0)
            & (consistency >= args.min_fold_consistency)
        )
        stable_score = np.abs(separation) * consistency
        for latent_id in np.flatnonzero(eligible):
            candidates.append(
                {
                    "group": group.name,
                    "position": group.position,
                    "layer": int(layer),
                    "latent_id": int(latent_id),
                    "direction": (
                        "success" if separation[latent_id] > 0 else "failure"
                    ),
                    "success_activation_rate": float(
                        full_s[latent_id] / n_discovery
                    ),
                    "failure_activation_rate": float(
                        full_f[latent_id] / n_discovery
                    ),
                    "separation_score": float(separation[latent_id]),
                    "fold_sign_consistency": float(consistency[latent_id]),
                    "minimum_fold_absolute_separation": float(
                        min_fold_abs[latent_id]
                    ),
                    "support": int(support[latent_id]),
                    "support_threshold": support_threshold,
                    "stability_score": float(stable_score[latent_id]),
                    "n_discovery_pairs": n_discovery,
                }
            )

    selected = []
    for direction in ("success", "failure"):
        directional = [row for row in candidates if row["direction"] == direction]
        directional.sort(
            key=lambda row: (
                row["stability_score"],
                row["minimum_fold_absolute_separation"],
                row["support"],
            ),
            reverse=True,
        )
        selected.extend(directional[: args.top_per_direction])
    selected.sort(
        key=lambda row: (
            row["stability_score"],
            row["minimum_fold_absolute_separation"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    if not selected:
        raise RuntimeError(f"No stable latents selected for {group.name}")
    diagnostics = {
        "n_pairs": len(group.pair_rows),
        "n_discovery_pairs": n_discovery,
        "n_evaluation_pairs": len(group.evaluation_indices),
        "n_unique_discovery_success": int(
            len(np.unique(group.success_ids[group.discovery_indices]))
        ),
        "n_unique_evaluation_success": int(
            len(np.unique(group.success_ids[group.evaluation_indices]))
        ),
        "fold_sizes": fold_sizes.tolist(),
        "support_threshold": support_threshold,
        "eligible_candidates": len(candidates),
        "selected_latents": len(selected),
        "n_exact_prefix_pairs": int(
            sum(
                pair_has_exact_prefix(
                    group.successes[index],
                    group.failures[index],
                    group.position,
                    value_to_token,
                )
                for index in range(len(group.pair_rows))
            )
        ),
        "n_exact_prefix_discovery_pairs": int(
            sum(
                pair_has_exact_prefix(
                    group.successes[index],
                    group.failures[index],
                    group.position,
                    value_to_token,
                )
                for index in group.discovery_indices
            )
        ),
        "n_exact_prefix_evaluation_pairs": int(
            sum(
                pair_has_exact_prefix(
                    group.successes[index],
                    group.failures[index],
                    group.position,
                    value_to_token,
                )
                for index in group.evaluation_indices
            )
        ),
    }
    return selected, diagnostics


def selected_by_layer(
    selected: Sequence[dict[str, Any]], count: int | None = None
) -> dict[int, list[tuple[int, int]]]:
    limit = len(selected) if count is None else min(count, len(selected))
    result: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for column, row in enumerate(selected[:limit]):
        result[int(row["layer"])].append((column, int(row["latent_id"])))
    return dict(result)


@torch.inference_mode()
def extract_selected_states(
    group: PairGroup,
    model: T5ForConditionalGeneration,
    tokenizer: Any,
    saes: dict[int, torch.nn.Module],
    selected: Sequence[dict[str, Any]],
    value_to_token: dict[int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    indices = group.evaluation_indices
    n = len(indices)
    k = len(selected)
    success_values = np.zeros((n, k), dtype=np.float32)
    failure_values = np.zeros((n, k), dtype=np.float32)
    baseline_margin = np.zeros(n, dtype=np.float32)
    gold_ids = np.zeros(n, dtype=np.int64)
    wrong_ids = np.zeros(n, dtype=np.int64)
    layer_map = selected_by_layer(selected)
    layers = sorted(layer_map)

    starts = range(0, n, args.batch_size)
    for start in tqdm(
        starts,
        desc=f"Free-running held-out states [{group.name}]",
        unit="batch",
    ):
        batch_indices = indices[start : start + args.batch_size]
        successes = [group.successes[index] for index in batch_indices]
        failures = [group.failures[index] for index in batch_indices]
        records = [*successes, *failures]
        encoded = encode_queries(records, tokenizer, args.context_size, device)
        decoder_input = make_decoder_input(
            records, group.position, tokenizer, value_to_token, device
        )
        with capture_mlp_outputs(model, layers) as cache:
            output = model(
                **encoded,
                decoder_input_ids=decoder_input,
                use_cache=False,
                return_dict=True,
            )
        batch_size = len(batch_indices)
        target_slice = slice(start, start + batch_size)
        for layer, items in layer_map.items():
            latent = saes[layer].encode(cache[layer].float())
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(
                    f"{group.name} layer {layer}: non-finite held-out latent"
                )
            for column, latent_id in items:
                success_values[target_slice, column] = (
                    latent[:batch_size, latent_id].float().cpu().numpy()
                )
                failure_values[target_slice, column] = (
                    latent[batch_size:, latent_id].float().cpu().numpy()
                )

        logits = output.logits[batch_size:, -1, :]
        for local, (index, failure) in enumerate(zip(batch_indices, failures)):
            gold = gold_token_ids(failure, value_to_token)[group.position - 1]
            generated = [int(token) for token in failure["generated_token_ids"]]
            wrong = (
                generated[group.position - 1]
                if group.position - 1 < len(generated)
                else int(tokenizer.eos_token_id)
            )
            gold_ids[start + local] = gold
            wrong_ids[start + local] = wrong
            baseline_margin[start + local] = float(
                (logits[local, gold] - logits[local, wrong]).item()
            )
        del cache, output, encoded, decoder_input

    return {
        "success_values": success_values,
        "failure_values": failure_values,
        "baseline_margin": baseline_margin,
        "gold_token_ids": gold_ids,
        "wrong_token_ids": wrong_ids,
    }


def cluster_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0 or replicates <= 0:
        return float("nan"), float("nan")
    unique, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_sums = np.bincount(
        inverse, weights=values.astype(np.float64), minlength=len(unique)
    )
    cluster_sizes = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(replicates, len(unique)))
    estimates = (
        cluster_sums[sampled].sum(axis=1)
        / cluster_sizes[sampled].sum(axis=1)
    )
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def logit_contribution_rows(
    group: PairGroup,
    selected: Sequence[dict[str, Any]],
    states: dict[str, np.ndarray],
    saes: dict[int, torch.nn.Module],
    model: T5ForConditionalGeneration,
    value_to_token: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    gold_ids = torch.from_numpy(states["gold_token_ids"]).long().to(model.device)
    wrong_ids = torch.from_numpy(states["wrong_token_ids"]).long().to(model.device)
    embedding = model.lm_head.weight
    scale = float(model.model_dim ** -0.5) if model.config.tie_word_embeddings else 1.0
    cluster_ids = group.success_ids[group.evaluation_indices]
    exact_prefix = np.asarray(
        [
            pair_has_exact_prefix(
                group.successes[index],
                group.failures[index],
                group.position,
                value_to_token,
            )
            for index in group.evaluation_indices
        ],
        dtype=bool,
    )
    rows = []
    with torch.inference_mode():
        token_directions = (embedding[gold_ids] - embedding[wrong_ids]) * scale
        for column, latent in enumerate(selected):
            direction = saes[int(latent["layer"])].W_dec[int(latent["latent_id"])]
            coefficient = (token_directions @ direction).float().cpu().numpy()
            success_contribution = states["success_values"][:, column] * coefficient
            failure_contribution = states["failure_values"][:, column] * coefficient
            paired_shift = success_contribution - failure_contribution
            ci_low, ci_high = cluster_bootstrap_ci(
                paired_shift,
                cluster_ids,
                args.bootstrap_replicates,
                args.seed + group.position * 10_000 + column,
            )
            if exact_prefix.any():
                exact_shift = paired_shift[exact_prefix]
                exact_clusters = cluster_ids[exact_prefix]
                exact_ci_low, exact_ci_high = cluster_bootstrap_ci(
                    exact_shift,
                    exact_clusters,
                    args.bootstrap_replicates,
                    args.seed + group.position * 20_000 + column,
                )
                exact_mean = float(exact_shift.mean())
            else:
                exact_ci_low = exact_ci_high = exact_mean = float("nan")
            rows.append(
                {
                    **latent,
                    "n_evaluation_pairs": len(group.evaluation_indices),
                    "mean_decoder_direction_gold_wrong_projection": float(
                        coefficient.mean()
                    ),
                    "mean_success_direct_logit_contribution": float(
                        success_contribution.mean()
                    ),
                    "mean_failure_direct_logit_contribution": float(
                        failure_contribution.mean()
                    ),
                    "mean_paired_direct_logit_shift": float(paired_shift.mean()),
                    "median_paired_direct_logit_shift": float(
                        np.median(paired_shift)
                    ),
                    "paired_direct_logit_shift_ci_low": ci_low,
                    "paired_direct_logit_shift_ci_high": ci_high,
                    "gold_favoring_shift_fraction": float(np.mean(paired_shift > 0)),
                    "n_exact_prefix_evaluation_pairs": int(exact_prefix.sum()),
                    "exact_prefix_mean_paired_direct_logit_shift": exact_mean,
                    "exact_prefix_paired_shift_ci_low": exact_ci_low,
                    "exact_prefix_paired_shift_ci_high": exact_ci_high,
                    "attribution_method": (
                        "z_i * <W_dec_i, W_U(gold)-W_U(wrong)>; "
                        "T5 tied-embedding scale included; final-LN/downstream "
                        "nonlinearity excluded"
                    ),
                }
            )
    return rows


def constrained_argmax(
    logits: torch.Tensor,
    decoder_input: torch.Tensor,
    allowed_tokens: dict[tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    result = []
    prefixes = decoder_input[:, 1:].detach().cpu().tolist()
    for row, prefix in enumerate(prefixes):
        allowed = allowed_tokens.get(tuple(int(token) for token in prefix))
        if not allowed:
            raise KeyError(f"No DocID continuation for prefix {prefix}")
        ids = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        winner = ids[torch.argmax(logits[row].index_select(0, ids))]
        result.append(winner)
    return torch.stack(result)


@contextmanager
def patch_hooks(
    model: T5ForConditionalGeneration,
    saes: dict[int, torch.nn.Module],
    selected: Sequence[dict[str, Any]],
    count: int,
    target_values: torch.Tensor,
) -> Iterator[None]:
    layer_map = selected_by_layer(selected, count)
    handles = []
    for layer, items in layer_map.items():
        columns = torch.tensor(
            [column for column, _ in items],
            dtype=torch.long,
            device=target_values.device,
        )
        latent_ids = torch.tensor(
            [latent_id for _, latent_id in items],
            dtype=torch.long,
            device=target_values.device,
        )
        module = model.decoder.block[layer].layer[2].DenseReluDense
        sae = saes[layer]

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            source_columns: torch.Tensor = columns,
            feature_ids: torch.Tensor = latent_ids,
            layer_sae: torch.nn.Module = sae,
        ) -> torch.Tensor:
            current_hidden = output[:, -1, :].float()
            current_latent = layer_sae.encode(current_hidden)
            desired = target_values.index_select(1, source_columns)
            if not bool(torch.isfinite(current_latent).all()) or not bool(
                torch.isfinite(desired).all()
            ):
                raise FloatingPointError("Non-finite value encountered during patching")
            delta = desired - current_latent.index_select(1, feature_ids)
            decoder_directions = layer_sae.W_dec.index_select(0, feature_ids)
            patched = output.clone()
            patched[:, -1, :] = (
                patched[:, -1, :]
                + (delta @ decoder_directions).to(patched.dtype)
            )
            return patched

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def shuffled_within_target(
    source_values: np.ndarray, target_ids: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(source_values))
    for target in np.unique(target_ids):
        rows = np.flatnonzero(target_ids == target)
        permutation[rows] = rng.permutation(rows)
    return source_values[permutation]


@torch.inference_mode()
def continue_generation(
    model: T5ForConditionalGeneration,
    encoder_outputs: Any,
    attention_mask: torch.Tensor,
    decoder_input: torch.Tensor,
    next_token: torch.Tensor,
    eos_token_id: int,
    max_docid_length: int,
    allowed_tokens: dict[tuple[int, ...], tuple[int, ...]],
) -> list[list[int]]:
    sequence = torch.cat([decoder_input, next_token[:, None]], dim=1)
    finished = next_token == eos_token_id
    # One extra decoder step is required to emit EOS after a maximum-length ID.
    while sequence.shape[1] - 1 <= max_docid_length and not bool(finished.all()):
        output = model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=sequence,
            use_cache=False,
            return_dict=True,
        )
        token = torch.full_like(next_token, eos_token_id)
        active_rows = torch.nonzero(~finished, as_tuple=False).flatten()
        active_token = constrained_argmax(
            output.logits.index_select(0, active_rows)[:, -1, :],
            sequence.index_select(0, active_rows),
            allowed_tokens,
        )
        token[active_rows] = active_token
        sequence = torch.cat([sequence, token[:, None]], dim=1)
        finished |= token == eos_token_id
    completed = []
    for row in sequence[:, 1:].detach().cpu().tolist():
        docid = []
        for token in row:
            if int(token) == eos_token_id:
                break
            docid.append(int(token))
        completed.append(docid)
    return completed


@torch.inference_mode()
def run_activation_patching(
    group: PairGroup,
    selected: Sequence[dict[str, Any]],
    states: dict[str, np.ndarray],
    model: T5ForConditionalGeneration,
    tokenizer: Any,
    saes: dict[int, torch.nn.Module],
    value_to_token: dict[int, int],
    allowed_tokens: dict[tuple[int, ...], tuple[int, ...]],
    max_docid_length: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    indices = group.evaluation_indices
    source_sets = {
        "paired_success": states["success_values"],
        "shuffled_success_control": shuffled_within_target(
            states["success_values"],
            states["gold_token_ids"],
            args.seed + group.position * 1000,
        ),
    }
    patch_counts = sorted(
        {
            min(int(count), len(selected))
            for count in args.patch_counts
            if int(count) > 0
        }
    )
    rows: list[dict[str, Any]] = []
    starts = range(0, len(indices), args.batch_size)
    for start in tqdm(
        starts,
        desc=f"Free-running activation patching [{group.name}]",
        unit="batch",
    ):
        batch_indices = indices[start : start + args.batch_size]
        failures = [group.failures[index] for index in batch_indices]
        encoded = encode_queries(failures, tokenizer, args.context_size, device)
        decoder_input = make_decoder_input(
            failures, group.position, tokenizer, value_to_token, device
        )
        encoder_outputs = model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        baseline = model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input,
            use_cache=False,
            return_dict=True,
        )
        baseline_logits = baseline.logits[:, -1, :]
        baseline_token = constrained_argmax(
            baseline_logits, decoder_input, allowed_tokens
        )
        gold = torch.tensor(
            [
                gold_token_ids(record, value_to_token)[group.position - 1]
                for record in failures
            ],
            dtype=torch.long,
            device=device,
        )
        recorded_wrong = torch.tensor(
            [
                (
                    int(record["generated_token_ids"][group.position - 1])
                    if group.position - 1 < len(record["generated_token_ids"])
                    else int(tokenizer.eos_token_id)
                )
                for record in failures
            ],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(baseline_token, recorded_wrong):
            mismatches = int((baseline_token != recorded_wrong).sum().item())
            raise AssertionError(
                f"{group.name}: constrained free-running replay disagrees with "
                f"recorded generation for {mismatches} held-out examples"
            )
        row_index = torch.arange(len(failures), device=device)
        baseline_margin = (
            baseline_logits[row_index, gold]
            - baseline_logits[row_index, recorded_wrong]
        )

        for source_name, source_values in source_sets.items():
            batch_source = torch.from_numpy(
                source_values[start : start + len(batch_indices)]
            ).to(device)
            for count in patch_counts:
                with patch_hooks(
                    model, saes, selected, count, batch_source
                ):
                    patched = model(
                        encoder_outputs=encoder_outputs,
                        attention_mask=encoded["attention_mask"],
                        decoder_input_ids=decoder_input,
                        use_cache=False,
                        return_dict=True,
                    )
                patched_logits = patched.logits[:, -1, :]
                patched_token = constrained_argmax(
                    patched_logits, decoder_input, allowed_tokens
                )
                patched_margin = (
                    patched_logits[row_index, gold]
                    - patched_logits[row_index, recorded_wrong]
                )
                completed = continue_generation(
                    model,
                    encoder_outputs,
                    encoded["attention_mask"],
                    decoder_input,
                    patched_token,
                    int(tokenizer.eos_token_id),
                    max_docid_length,
                    allowed_tokens,
                )
                for local, pair_index in enumerate(batch_indices):
                    gold_full = torch.tensor(
                        gold_token_ids(failures[local], value_to_token),
                        dtype=torch.long,
                        device=device,
                    )
                    rows.append(
                        {
                            "group": group.name,
                            "position": group.position,
                            "pair_index": int(pair_index),
                            "success_query_id": int(group.success_ids[pair_index]),
                            "failure_query_id": int(group.failure_ids[pair_index]),
                            "exact_prior_prefix_match": pair_has_exact_prefix(
                                group.successes[pair_index],
                                group.failures[pair_index],
                                group.position,
                                value_to_token,
                            ),
                            "source_condition": source_name,
                            "patched_latent_count": count,
                            "baseline_gold_wrong_margin": float(
                                baseline_margin[local].item()
                            ),
                            "patched_gold_wrong_margin": float(
                                patched_margin[local].item()
                            ),
                            "margin_delta": float(
                                (patched_margin[local] - baseline_margin[local]).item()
                            ),
                            "gold_token_id": int(gold[local].item()),
                            "original_wrong_token_id": int(
                                recorded_wrong[local].item()
                            ),
                            "patched_token_id": int(patched_token[local].item()),
                            "current_token_recovered": bool(
                                patched_token[local] == gold[local]
                            ),
                            "full_docid_recovered": bool(
                                completed[local] == gold_full.detach().cpu().tolist()
                            ),
                        }
                    )
        del baseline, encoded, decoder_input, encoder_outputs
    return rows


def summarize_patching(
    rows: Sequence[dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        base_key = (
            str(row["group"]),
            "all_pairs",
            str(row["source_condition"]),
            int(row["patched_latent_count"]),
        )
        grouped[base_key].append(row)
        if bool(row["exact_prior_prefix_match"]):
            exact_key = (
                str(row["group"]),
                "exact_prefix_pairs",
                str(row["source_condition"]),
                int(row["patched_latent_count"]),
            )
            grouped[exact_key].append(row)
    result = []
    for offset, (
        (group, evaluation_scope, condition, count),
        items,
    ) in enumerate(sorted(grouped.items())):
        clusters = np.asarray(
            [item["success_query_id"] for item in items], dtype=np.int64
        )
        margin_delta = np.asarray(
            [item["margin_delta"] for item in items], dtype=np.float64
        )
        current = np.asarray(
            [item["current_token_recovered"] for item in items], dtype=np.float64
        )
        full = np.asarray(
            [item["full_docid_recovered"] for item in items], dtype=np.float64
        )
        margin_ci = cluster_bootstrap_ci(
            margin_delta, clusters, bootstrap_replicates, seed + offset * 3
        )
        current_ci = cluster_bootstrap_ci(
            current, clusters, bootstrap_replicates, seed + offset * 3 + 1
        )
        full_ci = cluster_bootstrap_ci(
            full, clusters, bootstrap_replicates, seed + offset * 3 + 2
        )
        result.append(
            {
                "group": group,
                "position": GROUP_NAMES.index(group) + 1,
                "evaluation_scope": evaluation_scope,
                "source_condition": condition,
                "patched_latent_count": count,
                "n_evaluation_pairs": len(items),
                "n_unique_success_sources": int(len(np.unique(clusters))),
                "baseline_margin_mean": float(
                    np.mean([item["baseline_gold_wrong_margin"] for item in items])
                ),
                "patched_margin_mean": float(
                    np.mean([item["patched_gold_wrong_margin"] for item in items])
                ),
                "margin_delta_mean": float(margin_delta.mean()),
                "margin_delta_ci_low": margin_ci[0],
                "margin_delta_ci_high": margin_ci[1],
                "current_token_recovery_rate": float(current.mean()),
                "current_token_recovery_ci_low": current_ci[0],
                "current_token_recovery_ci_high": current_ci[1],
                "full_docid_recovery_rate": float(full.mean()),
                "full_docid_recovery_ci_low": full_ci[0],
                "full_docid_recovery_ci_high": full_ci[1],
            }
        )
    return result


def summarize_paired_control_contrast(
    rows: Sequence[dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_key: dict[
        tuple[str, int, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["group"]),
            int(row["patched_latent_count"]),
            int(row["pair_index"]),
        )
        by_key[key][str(row["source_condition"])] = row

    grouped: dict[
        tuple[str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for (group, count, _pair_index), conditions in by_key.items():
        if set(conditions) != {"paired_success", "shuffled_success_control"}:
            raise RuntimeError("Incomplete paired-vs-shuffled patching contrast")
        pair = (
            conditions["paired_success"],
            conditions["shuffled_success_control"],
        )
        grouped[(group, "all_pairs", count)].append(pair)
        if bool(pair[0]["exact_prior_prefix_match"]):
            grouped[(group, "exact_prefix_pairs", count)].append(pair)

    result = []
    for offset, ((group, scope, count), items) in enumerate(sorted(grouped.items())):
        clusters = np.asarray(
            [paired["success_query_id"] for paired, _ in items], dtype=np.int64
        )
        margin = np.asarray(
            [
                float(paired["margin_delta"]) - float(shuffled["margin_delta"])
                for paired, shuffled in items
            ],
            dtype=np.float64,
        )
        current = np.asarray(
            [
                float(paired["current_token_recovered"])
                - float(shuffled["current_token_recovered"])
                for paired, shuffled in items
            ],
            dtype=np.float64,
        )
        full = np.asarray(
            [
                float(paired["full_docid_recovered"])
                - float(shuffled["full_docid_recovered"])
                for paired, shuffled in items
            ],
            dtype=np.float64,
        )
        margin_ci = cluster_bootstrap_ci(
            margin, clusters, bootstrap_replicates, seed + offset * 3
        )
        current_ci = cluster_bootstrap_ci(
            current, clusters, bootstrap_replicates, seed + offset * 3 + 1
        )
        full_ci = cluster_bootstrap_ci(
            full, clusters, bootstrap_replicates, seed + offset * 3 + 2
        )
        result.append(
            {
                "group": group,
                "position": GROUP_NAMES.index(group) + 1,
                "evaluation_scope": scope,
                "patched_latent_count": count,
                "n_evaluation_pairs": len(items),
                "paired_minus_shuffled_margin_delta": float(margin.mean()),
                "margin_contrast_ci_low": margin_ci[0],
                "margin_contrast_ci_high": margin_ci[1],
                "paired_minus_shuffled_current_recovery_rate": float(
                    current.mean()
                ),
                "current_recovery_contrast_ci_low": current_ci[0],
                "current_recovery_contrast_ci_high": current_ci[1],
                "paired_minus_shuffled_full_docid_recovery_rate": float(
                    full.mean()
                ),
                "full_recovery_contrast_ci_low": full_ci[0],
                "full_recovery_contrast_ci_high": full_ci[1],
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_logit_contributions(
    rows: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    maximum = max(
        abs(float(row["mean_paired_direct_logit_shift"])) for row in rows
    )
    for axis, group in zip(axes.flat, GROUP_NAMES):
        selected = [row for row in rows if row["group"] == group]
        selected.sort(key=lambda row: int(row["rank"]), reverse=True)
        values = [float(row["mean_paired_direct_logit_shift"]) for row in selected]
        lows = [
            max(0.0, value - float(row["paired_direct_logit_shift_ci_low"]))
            for value, row in zip(values, selected)
        ]
        highs = [
            max(0.0, float(row["paired_direct_logit_shift_ci_high"]) - value)
            for value, row in zip(values, selected)
        ]
        labels = [
            f"L{row['layer']}:{row['latent_id']} ({row['direction'][0].upper()})"
            for row in selected
        ]
        colors = ["#cb181d" if value >= 0 else "#08519c" for value in values]
        y = np.arange(len(selected))
        axis.barh(y, values, color=colors, alpha=0.82)
        axis.errorbar(
            values, y, xerr=np.asarray([lows, highs]), fmt="none",
            ecolor="#333333", capsize=2, linewidth=0.8,
        )
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set_yticks(y, labels)
        limit = max(maximum * 1.2, 1e-3)
        axis.set_xlim(-limit, limit)
        axis.grid(axis="x", alpha=0.2)
        axis.set_title(f"{group} — position {GROUP_NAMES.index(group) + 1}")
        axis.set_xlabel("Paired Success→Failure direct gold−wrong logit shift")
    fig.suptitle("Top Stable SAE Latents: Gold-vs-Wrong Direct Logit Contribution")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "top_latent_gold_wrong_contribution.png", dpi=320)
    fig.savefig(output_dir / "top_latent_gold_wrong_contribution.pdf")
    plt.close(fig)


def plot_patching(
    rows: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    all_zero_recovery = all(
        float(row["current_token_recovery_rate"]) == 0
        and float(row["full_docid_recovery_rate"]) == 0
        for row in rows
        if row["evaluation_scope"] == "all_pairs"
    )
    for axis, group in zip(axes.flat, GROUP_NAMES):
        selected = [
            row
            for row in rows
            if row["group"] == group and row["evaluation_scope"] == "all_pairs"
        ]
        for condition, color, linestyle in (
            ("paired_success", "#cb181d", "-"),
            ("shuffled_success_control", "#6b6b6b", "--"),
        ):
            condition_rows = sorted(
                [row for row in selected if row["source_condition"] == condition],
                key=lambda row: int(row["patched_latent_count"]),
            )
            x = [int(row["patched_latent_count"]) for row in condition_rows]
            current = [
                100 * float(row["current_token_recovery_rate"])
                for row in condition_rows
            ]
            full = [
                100 * float(row["full_docid_recovery_rate"])
                for row in condition_rows
            ]
            label = "Paired source" if condition == "paired_success" else "Shuffled source"
            axis.plot(
                x, current, color=color, linestyle=linestyle, marker="o",
                label=f"{label}: current token",
            )
            axis.plot(
                x, full, color=color, linestyle=":" if condition == "paired_success" else "-.",
                marker="s", label=f"{label}: full DocID",
            )
        axis.set_title(f"{group} — position {GROUP_NAMES.index(group) + 1}")
        axis.set_xlabel("Number of patched stable latents")
        axis.set_ylabel("Recovery rate (%)")
        axis.set_xticks(
            sorted(
                {
                    int(row["patched_latent_count"])
                    for row in selected
                }
            )
        )
        if all_zero_recovery:
            n_pairs = int(selected[0]["n_evaluation_pairs"]) if selected else 0
            axis.set_ylim(-0.25, 5)
            axis.text(
                0.5,
                0.55,
                f"No recovery in {n_pairs} held-out pairs",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Free-running Activation Patching: Failure Recovery")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_dir / "activation_patching_recovery.png", dpi=320)
    fig.savefig(output_dir / "activation_patching_recovery.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, GROUP_NAMES):
        selected = [
            row
            for row in rows
            if row["group"] == group and row["evaluation_scope"] == "all_pairs"
        ]
        for condition, color, linestyle, label in (
            ("paired_success", "#cb181d", "-", "Paired Success source"),
            (
                "shuffled_success_control",
                "#6b6b6b",
                "--",
                "Target-stratified shuffled source",
            ),
        ):
            condition_rows = sorted(
                [row for row in selected if row["source_condition"] == condition],
                key=lambda row: int(row["patched_latent_count"]),
            )
            x = [int(row["patched_latent_count"]) for row in condition_rows]
            y = [float(row["margin_delta_mean"]) for row in condition_rows]
            low = [float(row["margin_delta_ci_low"]) for row in condition_rows]
            high = [float(row["margin_delta_ci_high"]) for row in condition_rows]
            axis.plot(x, y, color=color, linestyle=linestyle, marker="o", label=label)
            axis.fill_between(x, low, high, color=color, alpha=0.14)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(f"{group} — position {GROUP_NAMES.index(group) + 1}")
        axis.set_xlabel("Number of patched stable latents")
        axis.set_ylabel("Δ gold−wrong logit margin")
        axis.set_xticks(
            sorted(
                {
                    int(row["patched_latent_count"])
                    for row in selected
                }
            )
        )
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Free-running Activation Patching: Gold-vs-Wrong Margin")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output_dir / "activation_patching_margin_delta.png", dpi=320)
    fig.savefig(output_dir / "activation_patching_margin_delta.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 < args.evaluation_fraction < 1:
        raise ValueError("--evaluation-fraction must be between zero and one")
    if not 0.5 <= args.min_fold_consistency <= 1:
        raise ValueError("--min-fold-consistency must be in [0.5, 1]")
    if args.batch_size < 1 or args.top_per_direction < 1:
        raise ValueError("Batch size and top-per-direction must be positive")
    layers = sorted(set(args.layers or range(24)))
    if not layers or any(layer < 0 or layer >= 24 for layer in layers):
        raise ValueError("Layers must be in 0..23")
    for path in (
        args.model_checkpoint,
        args.semantic_id_path,
        args.outcome_root / "success.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on {device}")
    model, tokenizer = load_model(args.model_checkpoint, args.base_model, device)
    value_to_token = value_to_token_map(tokenizer)
    semantic_ids = read_json(args.semantic_id_path)
    max_docid_length = max(len(docid) for docid in semantic_ids)
    allowed_tokens = build_allowed_tokens(
        semantic_ids, value_to_token, int(tokenizer.eos_token_id)
    )
    groups = load_pair_groups(args)

    print(f"Loading {len(layers)} JumpReLU SAEs")
    saes: dict[int, torch.nn.Module] = {}
    sae_formats: dict[int, str] = {}
    for layer in tqdm(layers, desc="Loading SAEs", unit="layer"):
        sae, _config, loaded_format = load_sae(
            args.sae_root / f"layer_{layer}", device, args.sae_format
        )
        saes[layer] = sae
        sae_formats[layer] = loaded_format

    all_selected: list[dict[str, Any]] = []
    all_contributions: list[dict[str, Any]] = []
    all_patching_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for group in groups:
        selected, group_diagnostics = select_stable_latents(
            group,
            model,
            tokenizer,
            saes,
            layers,
            value_to_token,
            args,
            device,
        )
        diagnostics[group.name] = group_diagnostics
        all_selected.extend(selected)
        states = extract_selected_states(
            group,
            model,
            tokenizer,
            saes,
            selected,
            value_to_token,
            args,
            device,
        )
        all_contributions.extend(
            logit_contribution_rows(
                group, selected, states, saes, model, value_to_token, args
            )
        )
        all_patching_rows.extend(
            run_activation_patching(
                group,
                selected,
                states,
                model,
                tokenizer,
                saes,
                value_to_token,
                allowed_tokens,
                max_docid_length,
                args,
                device,
            )
        )

    patching_summary = summarize_patching(
        all_patching_rows, args.bootstrap_replicates, args.seed
    )
    patching_contrast = summarize_paired_control_contrast(
        all_patching_rows, args.bootstrap_replicates, args.seed + 500_000
    )
    write_csv(args.output_dir / "stable_latents.csv", all_selected)
    write_csv(args.output_dir / "top_latent_logit_contributions.csv", all_contributions)
    write_csv(args.output_dir / "activation_patching_summary.csv", patching_summary)
    write_csv(
        args.output_dir / "activation_patching_paired_vs_shuffled.csv",
        patching_contrast,
    )
    with gzip.open(
        args.output_dir / "activation_patching_per_example.csv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_patching_rows[0]))
        writer.writeheader()
        writer.writerows(all_patching_rows)
    plot_logit_contributions(all_contributions, args.output_dir)
    plot_patching(patching_summary, args.output_dir)

    manifest = {
        "activation_source": (
            "free-running generated-prefix replay at the actual decision position; "
            "teacher-forced activation caches are not read"
        ),
        "prefix_invariant": (
            "for every accepted first-error example, recorded generated prefix "
            "strictly before the decision equals the gold prefix"
        ),
        "selection_evaluation_separation": (
            "stable latents selected on grouped discovery pairs; logit attribution "
            "and patching evaluated only on held-out Success-ID groups"
        ),
        "stability_definition": (
            "activation-frequency separation with matching sign in at least "
            f"{args.min_fold_consistency:.0%} of {args.stability_folds} grouped folds"
        ),
        "patch_definition": (
            "at the current free-running position, h <- h + "
            "(z_source[J]-z_failure[J]) @ W_dec[J], preserving all unpatched "
            "components and then regenerating the remaining DocID tokens"
        ),
        "negative_control": (
            "Success sources shuffled within the current gold-token stratum"
        ),
        "layers": layers,
        "sae_formats": sae_formats,
        "patch_counts": sorted(
            set(min(count, args.top_per_direction * 2) for count in args.patch_counts)
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "groups": diagnostics,
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Saved free-running contribution and patching results to {args.output_dir}")


if __name__ == "__main__":
    main()
