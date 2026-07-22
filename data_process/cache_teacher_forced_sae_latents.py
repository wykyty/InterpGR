"""Cache layer-by-position SAE latents from teacher-forced DocID decoding.

The input is not the raw dev set.  It consumes the outcome splits produced by
``build_generation_failure_sets.py``: Success and the four first-error groups
F1--F4.  Each split is independently teacher-forced, encoded, and saved.

The main output is a Parquet table with this schema::

    query_id: int32
    layer: int8
    position: int8
    gold_prefix: list<int16>
    target_token: int16
    sae_activation: map<int32, float32>

``position`` is one-based.  ``gold_prefix`` contains tokens strictly before
the target, so at position 1 it is empty.  ``sae_activation`` is an exact
sparse representation of the latent vector: keys are latent IDs and values
are activations whose absolute value exceeds ``--activation-threshold``.

Example, from the repository root::

    uv run python data_process/cache_teacher_forced_sae_latents.py \
        --model-checkpoint out/dsi-semantic-bert/99.pt \
        --sae-root out/sae_train_8x \
        --split-root dataset/nq320k/generation_outcomes \
        --output-dir data/teacher_forced_sae_latents
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

# Offline flags must precede imports that initialize Hugging Face libraries.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.common import load_sae  # noqa: E402
from sae.cache_activations import load_dsi_model  # noqa: E402

N_DECODER_LAYERS = 24
DEFAULT_SPLIT_FILES = {
    "success": "success.json",
    "F1": "failure_1.json",
    "F2": "failure_2.json",
    "F3": "failure_3.json",
    "F4": "failure_4.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher-forced decoding and cache sparse SAE latents."
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=Path("out/dsi-semantic-bert/99.pt"),
    )
    parser.add_argument(
        "--sae-root", type=Path, default=Path("out/sae_train_8x")
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
        help="Directory containing success.json and failure_1.json ... failure_4.json.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(DEFAULT_SPLIT_FILES),
        default=list(DEFAULT_SPLIT_FILES),
        help="Outcome groups to process independently.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/teacher_forced_sae_latents"),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Decoder layers to process (default: all 0-23).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.0,
        help="Store latents for which abs(activation) exceeds this value.",
    )
    parser.add_argument(
        "--sae-format",
        choices=("auto", "inference", "training"),
        default="inference",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-queries-per-split",
        type=int,
        default=None,
        help="Optional per-split prefix, useful only for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def load_split_examples(
    split_name: str, split_path: Path, max_queries: int | None
) -> list[dict[str, Any]]:
    """Load and validate one generated-outcome split."""
    with split_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"{split_path} must contain a JSON array")

    examples: list[dict[str, Any]] = []
    expected_position = None if split_name == "success" else int(split_name[1:])
    for row_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{split_path} row {row_index} is not an object")
        query = str(record.get("query", "")).strip()
        query_id = record.get("sample_index", record.get("query_id"))
        doc_id = record.get("gold_doc_id", record.get("doc_id"))
        gold_docid = record.get("gold_docid")
        if not query or query_id is None or doc_id is None or not gold_docid:
            raise ValueError(f"{split_path} row {row_index} has incomplete fields")
        is_success = bool(record.get("success"))
        error_position = record.get("first_error_position")
        if split_name == "success":
            if not is_success or error_position is not None:
                raise ValueError(f"Invalid success record at {split_path}:{row_index}")
        elif is_success or int(error_position) != expected_position:
            raise ValueError(
                f"Invalid {split_name} record at {split_path}:{row_index}; "
                f"first_error_position={error_position}"
            )
        examples.append(
            {
                "query_id": int(query_id),
                "query": query,
                "doc_id": int(doc_id),
                "gold_docid": [int(token) for token in gold_docid],
            }
        )
        if max_queries is not None and len(examples) >= max_queries:
            break
    return examples


def validate_paths(args: argparse.Namespace, layers: Sequence[int]) -> None:
    required_files = [args.model_checkpoint]
    required_files.extend(
        args.split_root / DEFAULT_SPLIT_FILES[split_name]
        for split_name in args.splits
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for layer in layers:
        checkpoint_dir = args.sae_root / f"layer_{layer}"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(checkpoint_dir)


def make_decoder_inputs(
    batch: Sequence[dict[str, Any]], tokenizer: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Create shifted gold decoder inputs and their valid-position mask."""
    value_to_token: dict[int, int] = {}
    for example in batch:
        for value in example["gold_docid"]:
            if value not in value_to_token:
                token_id = int(tokenizer.convert_tokens_to_ids(f"${value}$"))
                if token_id == tokenizer.unk_token_id:
                    raise ValueError(f"Tokenizer is missing DocID token ${value}$")
                value_to_token[value] = token_id

    target_ids = [
        [value_to_token[value] for value in example["gold_docid"]]
        for example in batch
    ]
    lengths = [len(tokens) for tokens in target_ids]
    max_length = max(lengths)
    pad_id = int(tokenizer.pad_token_id)
    decoder_input = torch.full(
        (len(batch), max_length), pad_id, dtype=torch.long, device=device
    )
    valid_mask = torch.zeros(
        (len(batch), max_length), dtype=torch.bool, device=device
    )
    for row, tokens in enumerate(target_ids):
        valid_mask[row, : len(tokens)] = True
        # decoder_input[:, 0] is PAD, T5's decoder start token.  At position t,
        # the remaining input is exactly gold tokens [0:t].
        if len(tokens) > 1:
            decoder_input[row, 1 : len(tokens)] = torch.tensor(
                tokens[:-1], dtype=torch.long, device=device
            )
    return decoder_input, valid_mask, target_ids


def make_batch_metadata(
    batch: Sequence[dict[str, Any]],
) -> tuple[list[int], list[int], list[list[int]], list[int]]:
    query_ids: list[int] = []
    positions: list[int] = []
    prefixes: list[list[int]] = []
    targets: list[int] = []
    for example in batch:
        gold = example["gold_docid"]
        for position_index, target in enumerate(gold):
            query_ids.append(int(example["query_id"]))
            positions.append(position_index + 1)
            prefixes.append(gold[:position_index])
            targets.append(int(target))
    return query_ids, positions, prefixes, targets


def sparse_activation_maps(
    latent_acts: torch.Tensor, threshold: float
) -> tuple[pa.MapArray, int]:
    """Convert a dense SAE batch to a vectorized Arrow sparse-map array."""
    latent_acts = latent_acts.detach().float()
    active_mask = latent_acts.abs() > threshold
    counts = active_mask.sum(dim=1, dtype=torch.int64)
    active_total = int(counts.sum().item())
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=counts.device),
            counts.cumsum(dim=0),
        ]
    )
    # torch.nonzero is row-major, so latent IDs remain sorted within each map.
    active_ids = torch.nonzero(active_mask, as_tuple=False)[:, 1]
    active_values = latent_acts[active_mask]
    map_type = LATENT_SCHEMA.field("sae_activation").type
    maps = pa.MapArray.from_arrays(
        pa.array(offsets.cpu().numpy(), type=pa.int32()),
        pa.array(active_ids.to(torch.int32).cpu().numpy(), type=pa.int32()),
        pa.array(active_values.cpu().numpy(), type=pa.float32()),
        type=map_type,
    )
    return maps, active_total


LATENT_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.int32(), nullable=False),
        pa.field("layer", pa.int8(), nullable=False),
        pa.field("position", pa.int8(), nullable=False),
        pa.field("gold_prefix", pa.list_(pa.int16()), nullable=False),
        pa.field("target_token", pa.int16(), nullable=False),
        pa.field(
            "sae_activation",
            pa.map_(pa.int32(), pa.float32(), keys_sorted=True),
            nullable=False,
        ),
    ]
)


def make_arrow_table(
    query_ids: Sequence[int],
    layer: int,
    positions: Sequence[int],
    prefixes: Sequence[Sequence[int]],
    targets: Sequence[int],
    sparse_maps: pa.MapArray,
) -> pa.Table:
    row_count = len(query_ids)
    if not all(
        len(column) == row_count
        for column in (positions, prefixes, targets, sparse_maps)
    ):
        raise RuntimeError("Latent rows and metadata rows have different lengths")
    return pa.Table.from_arrays(
        [
            pa.array(query_ids, type=pa.int32()),
            pa.array(np.full(row_count, layer, dtype=np.int8), type=pa.int8()),
            pa.array(positions, type=pa.int8()),
            pa.array(prefixes, type=pa.list_(pa.int16())),
            pa.array(targets, type=pa.int16()),
            sparse_maps,
        ],
        schema=LATENT_SCHEMA,
    )


def write_query_table(examples: Sequence[dict[str, Any]], path: Path) -> None:
    table = pa.table(
        {
            "query_id": pa.array(
                [example["query_id"] for example in examples], type=pa.int32()
            ),
            "query": pa.array([example["query"] for example in examples], type=pa.string()),
            "doc_id": pa.array(
                [example["doc_id"] for example in examples], type=pa.int32()
            ),
            "gold_docid": pa.array(
                [example["gold_docid"] for example in examples],
                type=pa.list_(pa.int16()),
            ),
        }
    )
    pq.write_table(table, path, compression="zstd")


def process_split(
    split_name: str,
    source_path: Path,
    examples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    layers: Sequence[int],
    model: torch.nn.Module,
    tokenizer: Any,
    saes: dict[int, torch.nn.Module],
    sae_formats: dict[int, str],
    d_sae_by_layer: dict[int, int],
    hook_names: set[str],
    device: torch.device,
) -> dict[str, Any]:
    """Forward and save one outcome group without mixing it with another."""
    if not examples:
        raise RuntimeError(f"Split {split_name} contains no examples")
    split_output_dir = args.output_dir / split_name
    final_latent_path = split_output_dir / "sae_latents.parquet"
    final_query_path = split_output_dir / "queries.parquet"
    final_manifest_path = split_output_dir / "manifest.json"
    existing = [
        path
        for path in (final_latent_path, final_query_path, final_manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists ({existing[0]}). Pass --overwrite to replace it."
        )
    split_output_dir.mkdir(parents=True, exist_ok=True)
    temporary_latent_path = split_output_dir / "sae_latents.in_progress.parquet"
    if temporary_latent_path.exists():
        temporary_latent_path.unlink()

    token_rows_per_layer = sum(len(example["gold_docid"]) for example in examples)
    expected_rows = token_rows_per_layer * len(layers)
    print(
        f"[{split_name}] {len(examples):,} queries; "
        f"{token_rows_per_layer:,} DocID positions "
        f"x {len(layers)} layers = {expected_rows:,} table rows"
    )
    per_layer_rows: Counter[int] = Counter()
    per_layer_active: Counter[int] = Counter()
    total_rows = 0
    writer = pq.ParquetWriter(
        temporary_latent_path,
        LATENT_SCHEMA,
        compression="zstd",
        use_dictionary=["layer", "position", "target_token"],
        write_statistics=True,
    )
    try:
        starts = range(0, len(examples), args.batch_size)
        for start in tqdm(
            starts, desc=f"Teacher-forced SAE encoding [{split_name}]", unit="batch"
        ):
            batch = examples[start : start + args.batch_size]
            encoder = tokenizer(
                [example["query"] for example in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.context_size,
            ).to(device)
            decoder_input, valid_mask, _ = make_decoder_inputs(batch, tokenizer, device)
            query_ids, positions, prefixes, targets = make_batch_metadata(batch)

            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    encoder.input_ids,
                    one_zero_attention_mask=encoder.attention_mask,
                    decoder_input=decoder_input,
                    names_filter=lambda name: name in hook_names,
                )
                for layer in layers:
                    hook_name = f"decoder.{layer}.hook_mlp_out"
                    model_activations = cache[hook_name][valid_mask].float()
                    if model_activations.shape[0] != len(query_ids):
                        raise RuntimeError(
                            f"Layer {layer}: got {model_activations.shape[0]} positions, "
                            f"expected {len(query_ids)}"
                        )
                    latent_acts = saes[layer].encode(model_activations)
                    if latent_acts.shape != (
                        len(query_ids),
                        d_sae_by_layer[layer],
                    ):
                        raise RuntimeError(
                            f"Layer {layer}: unexpected SAE output shape "
                            f"{tuple(latent_acts.shape)}"
                        )
                    sparse_maps, active_count = sparse_activation_maps(
                        latent_acts, args.activation_threshold
                    )
                    writer.write_table(
                        make_arrow_table(
                            query_ids,
                            layer,
                            positions,
                            prefixes,
                            targets,
                            sparse_maps,
                        )
                    )
                    row_count = len(query_ids)
                    per_layer_rows[layer] += row_count
                    per_layer_active[layer] += active_count
                    total_rows += row_count
                    del model_activations, latent_acts, sparse_maps
            del cache, encoder, decoder_input, valid_mask
    finally:
        writer.close()

    if total_rows != expected_rows:
        raise RuntimeError(f"Wrote {total_rows} rows, expected {expected_rows}")
    for layer in layers:
        if per_layer_rows[layer] != token_rows_per_layer:
            raise RuntimeError(
                f"Layer {layer}: wrote {per_layer_rows[layer]} rows, "
                f"expected {token_rows_per_layer}"
            )

    # Only publish final names after the complete Parquet file has closed and
    # all row-count invariants pass.
    os.replace(temporary_latent_path, final_latent_path)
    write_query_table(examples, final_query_path)
    manifest = {
        "n_queries": len(examples),
        "n_docid_positions_per_layer": token_rows_per_layer,
        "n_layers": len(layers),
        "n_rows": total_rows,
        "layers": layers,
        "d_sae_by_layer": {str(layer): d_sae_by_layer[layer] for layer in layers},
        "average_active_latents_by_layer": {
            str(layer): per_layer_active[layer] / per_layer_rows[layer]
            for layer in layers
        },
        "docid_length_distribution": dict(
            sorted(Counter(len(example["gold_docid"]) for example in examples).items())
        ),
        "split": split_name,
        "source_split_file": str(source_path),
        "model_checkpoint": str(args.model_checkpoint),
        "sae_root": str(args.sae_root),
        "sae_formats": {str(layer): sae_formats[layer] for layer in layers},
        "context_size": args.context_size,
        "activation_threshold": args.activation_threshold,
        "position_definition": "one-based DocID generation position",
        "gold_prefix_definition": "gold DocID tokens strictly before target_token",
        "sae_activation_representation": (
            "Arrow map<latent_id:int32, activation:float32>; entries with "
            "abs(activation) <= activation_threshold are omitted"
        ),
        "files": {
            "latent_table": final_latent_path.name,
            "query_table": final_query_path.name,
        },
    }
    with final_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"[{split_name}] Saved {total_rows:,} rows to {final_latent_path}")
    return manifest


def main() -> None:
    args = parse_args()
    layers = sorted(set(args.layers or range(N_DECODER_LAYERS)))
    if not layers or any(layer < 0 or layer >= N_DECODER_LAYERS for layer in layers):
        raise ValueError("--layers must contain decoder layer IDs from 0 through 23")
    if args.batch_size < 1 or args.context_size < 1:
        raise ValueError("--batch-size and --context-size must be positive")
    if args.activation_threshold < 0:
        raise ValueError("--activation-threshold must be non-negative")
    if args.max_queries_per_split is not None and args.max_queries_per_split < 1:
        raise ValueError("--max-queries-per-split must be positive")
    validate_paths(args, layers)

    split_examples: dict[str, list[dict[str, Any]]] = {}
    split_paths: dict[str, Path] = {}
    for split_name in args.splits:
        split_path = args.split_root / DEFAULT_SPLIT_FILES[split_name]
        split_paths[split_name] = split_path
        split_examples[split_name] = load_split_examples(
            split_name, split_path, args.max_queries_per_split
        )
    total_queries = sum(len(examples) for examples in split_examples.values())
    total_positions = sum(
        len(example["gold_docid"])
        for examples in split_examples.values()
        for example in examples
    )
    print(
        "Loaded outcome splits: "
        + ", ".join(
            f"{split_name}={len(split_examples[split_name]):,}"
            for split_name in args.splits
        )
    )
    print(
        f"Total: {total_queries:,} queries, {total_positions:,} DocID positions, "
        f"{total_positions * len(layers):,} layer-position rows"
    )

    device = torch.device(args.device)
    print(f"Loading GenIR model from {args.model_checkpoint} on {device}")
    model, tokenizer = load_dsi_model(str(args.model_checkpoint), str(device))
    saes: dict[int, torch.nn.Module] = {}
    sae_formats: dict[int, str] = {}
    d_sae_by_layer: dict[int, int] = {}
    print(f"Loading {len(layers)} SAE checkpoints from {args.sae_root}")
    for layer in tqdm(layers, desc="Loading SAEs", unit="layer"):
        sae, config, loaded_format = load_sae(
            args.sae_root / f"layer_{layer}", device, args.sae_format
        )
        saes[layer] = sae
        sae_formats[layer] = loaded_format
        d_sae_by_layer[layer] = int(config["d_sae"])
    hook_names = {f"decoder.{layer}.hook_mlp_out" for layer in layers}

    manifests: dict[str, dict[str, Any]] = {}
    for split_name in args.splits:
        manifests[split_name] = process_split(
            split_name=split_name,
            source_path=split_paths[split_name],
            examples=split_examples[split_name],
            args=args,
            layers=layers,
            model=model,
            tokenizer=tokenizer,
            saes=saes,
            sae_formats=sae_formats,
            d_sae_by_layer=d_sae_by_layer,
            hook_names=hook_names,
            device=device,
        )

    run_manifest = {
        "splits": args.splits,
        "split_query_counts": {
            split_name: manifests[split_name]["n_queries"] for split_name in args.splits
        },
        "split_row_counts": {
            split_name: manifests[split_name]["n_rows"] for split_name in args.splits
        },
        "total_queries": total_queries,
        "total_docid_positions": total_positions,
        "total_rows": sum(manifest["n_rows"] for manifest in manifests.values()),
        "layers": layers,
        "separation": "Each outcome split was forwarded and saved independently.",
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)

    del saes, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Finished independent splits under {args.output_dir}")


if __name__ == "__main__":
    main()
