"""Build success/failure dev splits from free autoregressive DocID generation.

The activation-cache experiments use teacher-forced DocID tokens.  This script
instead lets the model generate a DocID from each dev query, compares the
complete generated DocID with its gold semantic DocID, and writes datasets for
successful examples, all failed examples, and each first-error position.

Decoding is *autoregressive but constrained* to the trie of valid corpus
DocIDs.  It never reads the gold DocID while decoding; the constraint only
prevents malformed identifiers.

Example (run from the repository root)::

    python data_process/build_generation_failure_sets.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --output-dir dataset/nq320k/generation_outcomes

Outputs
-------
``all_generation_outcomes.jsonl`` contains every valid dev example.  The
remaining files are JSON arrays: ``success.json``, ``failure.json`` and
``failure_<N>.json``.  ``N`` is the one-based *absolute* DocID token position
of the first discrepancy (so an error at the third DocID token is in
``failure_3.json``).  Files are created dynamically for every observed
position; for the usual four-token DocIDs this gives failure_1 ... failure_4.
``summary.json`` records the exact configuration and split counts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

# Keep the same offline-friendly behaviour as sae/cache_activations.py.  These
# must be set before transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: E402


class DocIDTrie:
    """Prefix constraint over complete decoder token sequences."""

    def __init__(self) -> None:
        self.root: dict[int, dict] = {}

    def insert(self, sequence: Iterable[int]) -> None:
        node = self.root
        for token_id in sequence:
            node = node.setdefault(int(token_id), {})

    def allowed_tokens(self, _batch_id: int, prefix: torch.Tensor) -> list[int]:
        node = self.root
        for token_id in prefix.tolist():
            child = node.get(int(token_id))
            if child is None:
                # This should not occur for a valid constrained generation.
                # EOS makes the failure explicit instead of allowing arbitrary
                # vocabulary tokens after an invalid prefix.
                return [1]
            node = child
        return list(node.keys()) or [1]


@dataclass
class ParsedGeneration:
    predicted_docid: list[Optional[int]]
    generated_token_ids: list[int]
    invalid_generated_tokens: list[dict[str, Any]]
    finished_with_eos: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dev DocIDs and construct success/failure splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/nq320k"),
        help="Directory containing dev.json and the semantic-id JSON file.",
    )
    parser.add_argument(
        "--dev-file", type=Path, default=None, help="Defaults to DATA_DIR/dev.json."
    )
    parser.add_argument(
        "--semantic-id-file",
        type=Path,
        default=None,
        help="Defaults to DATA_DIR/docid_semantic_bert.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned DSI/GenIR T5 checkpoint (.pt).",
    )
    parser.add_argument("--base-model", default="google-t5/t5-large")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=6,
        help="Includes only newly generated tokens (DocID plus EOS).",
    )
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unwrap_scalar(value: Any) -> Any:
    """Unwrap legacy NQ fields that may use a one-element list."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def get_query(record: Any) -> str:
    if isinstance(record, dict):
        query = record.get("query", "")
    elif isinstance(record, (list, tuple)) and record:
        query = record[0]
    else:
        raise TypeError(f"Unsupported dev record format: {type(record)!r}")
    query = unwrap_scalar(query)
    if not isinstance(query, str):
        query = str(query)
    return query.strip()


def get_doc_id(record: Any) -> int:
    if isinstance(record, dict):
        doc_id = record.get("doc_id")
    elif isinstance(record, (list, tuple)) and len(record) >= 2:
        doc_id = record[1]
    else:
        raise TypeError(f"Unsupported dev record format: {type(record)!r}")
    doc_id = unwrap_scalar(doc_id)
    if doc_id is None:
        raise KeyError("dev record is missing doc_id")
    return int(doc_id)


def find_semantic_id_file(data_dir: Path) -> Path:
    candidates = (
        data_dir / "docid_semantic_bert.json",
        data_dir / "docid_semantic.json",
        data_dir / "docid_semantic_bert_64.json",
        data_dir.parent / f"{data_dir.name}_id" / "id.semantic.bert.json",
        data_dir.parent / f"{data_dir.name}_id" / "id.semantic.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(data_dir.glob("*semantic*.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find a semantic DocID JSON in {data_dir}. "
        "Pass --semantic-id-file explicitly."
    )


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state)!r}")

    # DataParallel checkpoints have this prefix; normal checkpoints are left
    # untouched.  Some training wrappers use model., which is handled too.
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    return normalized


def make_docid_token_map(tokenizer: AutoTokenizer, semantic_ids: Sequence[Sequence[int]]) -> dict[int, int]:
    values = sorted({int(value) for docid in semantic_ids for value in docid})
    token_map: dict[int, int] = {}
    for value in values:
        token_id = tokenizer.convert_tokens_to_ids(f"${value}$")
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"DocID token ${value}$ is missing from the tokenizer.")
        token_map[int(token_id)] = value
    return token_map


def parse_generated_sequence(
    sequence: Sequence[int],
    *,
    token_to_value: dict[int, int],
    tokenizer: AutoTokenizer,
    decoder_start_token_id: Optional[int],
) -> ParsedGeneration:
    """Keep positions intact: non-DocID tokens become ``None``, not dropped."""
    raw = [int(token_id) for token_id in sequence]
    if raw and raw[0] == decoder_start_token_id:
        raw = raw[1:]

    generated: list[int] = []
    predicted: list[Optional[int]] = []
    invalid: list[dict[str, Any]] = []
    finished = False
    for position, token_id in enumerate(raw, start=1):
        if token_id == tokenizer.eos_token_id:
            finished = True
            break
        generated.append(token_id)
        value = token_to_value.get(token_id)
        if value is None:
            predicted.append(None)
            invalid.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "token": tokenizer.convert_ids_to_tokens(token_id),
                }
            )
        else:
            predicted.append(value)
    return ParsedGeneration(predicted, generated, invalid, finished)


def first_error(
    gold: Sequence[int], predicted: Sequence[Optional[int]]
) -> tuple[Optional[int], Optional[str]]:
    """Return zero-based absolute position and the type of first discrepancy."""
    for index, gold_token in enumerate(gold):
        if index >= len(predicted):
            return index, "prediction_ended_early"
        if predicted[index] != gold_token:
            return index, "token_mismatch"
    if len(predicted) > len(gold):
        return len(gold), "extra_prediction_token"
    return None, None


def json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def build_prefix_constraint(
    semantic_ids: Sequence[Sequence[int]],
    token_to_value: dict[int, int],
    decoder_start_token_id: int,
    eos_token_id: int,
) -> Callable[[int, torch.Tensor], list[int]]:
    value_to_token = {value: token_id for token_id, value in token_to_value.items()}
    trie = DocIDTrie()
    for docid in semantic_ids:
        trie.insert([decoder_start_token_id, *(value_to_token[int(value)] for value in docid), eos_token_id])
    return trie.allowed_tokens


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_beams < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size, num-beams and max-new-tokens must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev_file = args.dev_file or args.data_dir / "dev.json"
    semantic_file = args.semantic_id_file or find_semantic_id_file(args.data_dir)
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")
    if not semantic_file.exists():
        raise FileNotFoundError(f"Semantic ID file not found: {semantic_file}")

    with dev_file.open("r", encoding="utf-8") as handle:
        raw_dev = json.load(handle)
    with semantic_file.open("r", encoding="utf-8") as handle:
        semantic_ids = json.load(handle)
    semantic_ids = [[int(value) for value in docid] for docid in semantic_ids]

    # Keep the cache_activations convention: examples with blank queries are
    # excluded, while sample_index preserves their original dev-file position.
    examples: list[dict[str, Any]] = []
    skipped_empty = 0
    for sample_index, record in enumerate(raw_dev):
        query = get_query(record)
        if not query:
            skipped_empty += 1
            continue
        doc_id = get_doc_id(record)
        if not 0 <= doc_id < len(semantic_ids):
            raise IndexError(f"doc_id {doc_id} at dev index {sample_index} is out of range")
        examples.append(
            {
                "sample_index": sample_index,
                "query_index": len(examples),
                "query": query,
                "gold_doc_id": doc_id,
                "gold_docid": semantic_ids[doc_id],
            }
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading {args.base_model} on {device}; {len(examples):,} valid dev queries")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_tokens([f"${index}$" for index in range(30)])
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    missing, unexpected = model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    if missing or unexpected:
        print(f"Checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.to(device).eval()

    token_to_value = make_docid_token_map(tokenizer, semantic_ids)
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id
    if decoder_start is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer/model must define decoder_start_token_id and eos_token_id")

    prefix_allowed_tokens_fn = build_prefix_constraint(
        semantic_ids, token_to_value, int(decoder_start), int(tokenizer.eos_token_id)
    )

    outcomes: list[dict[str, Any]] = []
    iterator = range(0, len(examples), args.batch_size)
    for start in tqdm(iterator, desc="Generating DocIDs", unit="batch"):
        batch_examples = examples[start : start + args.batch_size]
        encoded = tokenizer(
            [item["query"] for item in batch_examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "do_sample": False,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
        }
        generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)

        for example, sequence in zip(batch_examples, generated.detach().cpu().tolist()):
            parsed = parse_generated_sequence(
                sequence,
                token_to_value=token_to_value,
                tokenizer=tokenizer,
                decoder_start_token_id=int(decoder_start),
            )
            error_index, error_type = first_error(example["gold_docid"], parsed.predicted_docid)
            outcome = {
                **example,
                "predicted_docid": parsed.predicted_docid,
                "generated_token_ids": parsed.generated_token_ids,
                "invalid_generated_tokens": parsed.invalid_generated_tokens,
                "finished_with_eos": parsed.finished_with_eos,
                "decoding": "constrained_docid_trie",
                "success": error_index is None,
                "first_error_index": error_index,
                "first_error_position": None if error_index is None else error_index + 1,
                "first_error_type": error_type,
                "failure_bucket": None if error_index is None else f"failure_{error_index + 1}",
            }
            outcomes.append(outcome)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success = [outcome for outcome in outcomes if outcome["success"]]
    failure = [outcome for outcome in outcomes if not outcome["success"]]
    by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for outcome in failure:
        by_position[int(outcome["first_error_position"])].append(outcome)

    # JSONL is convenient for streaming analysis; JSON arrays make split files
    # directly consumable by standard dataset loaders.
    with (args.output_dir / "all_generation_outcomes.jsonl").open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
    json_dump(args.output_dir / "success.json", success)
    json_dump(args.output_dir / "failure.json", failure)
    for position, split in sorted(by_position.items()):
        json_dump(args.output_dir / f"failure_{position}.json", split)

    summary = {
        "num_raw_dev_examples": len(raw_dev),
        "num_valid_dev_examples": len(outcomes),
        "num_skipped_empty_queries": skipped_empty,
        "num_success": len(success),
        "num_failure": len(failure),
        "success_rate": len(success) / len(outcomes) if outcomes else 0.0,
        "failure_counts_by_first_error_position": {
            str(position): len(split) for position, split in sorted(by_position.items())
        },
        "gold_docid_length_distribution": {
            str(length): count
            for length, count in sorted(Counter(len(item["gold_docid"]) for item in outcomes).items())
        },
        "config": {
            "base_model": args.base_model,
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "dev_file": str(dev_file),
            "semantic_id_file": str(semantic_file),
            "decoding": "constrained_docid_trie",
            "num_beams": args.num_beams,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "position_definition": (
            "first_error_position is one-based absolute DocID-token position; "
            "first_error_index is the corresponding zero-based index."
        ),
    }
    json_dump(args.output_dir / "summary.json", summary)

    print(
        f"Wrote {len(outcomes):,} outcomes to {args.output_dir} | "
        f"success={len(success):,} ({summary['success_rate']:.2%}), failure={len(failure):,}"
    )
    print("Failure counts by first-error position:", summary["failure_counts_by_first_error_position"])


if __name__ == "__main__":
    main()
