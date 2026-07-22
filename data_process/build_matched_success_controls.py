"""Construct matched Success controls S1--S4 for Failure F1--F4.

Inputs are the outcome datasets created by
``build_generation_failure_sets.py``.  For each failure group Fr, matching has
the following strict priority:

1. Keep the current gold target token identical.
2. Match the complete gold route through position r (prior prefix + target).
3. If that route has no Success example, maximize the common prior-prefix
   length while retaining the same target token.
4. Within equally good structural matches, minimize query-length and document-
   frequency differences and prefer the same document cluster.

Success examples are used without replacement whenever possible.  Reuse is
allowed only when needed to preserve the target-token distribution or when an
exact route exists but has insufficient unique Success examples.  Reuse is
reported explicitly, because matched row count and effective unique sample
size are different quantities.

Outputs::

    matched_success_controls/
      S1.json ... S4.json
      S1_pairs.jsonl ... S4_pairs.jsonl
      matching_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


GROUPS = range(1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build matched Success controls S1-S4 for Failure groups."
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes/matched_success_controls"),
    )
    parser.add_argument(
        "--query-length",
        choices=("whitespace", "characters"),
        default="whitespace",
        help="Length definition used only for nearest-neighbour tie breaking.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise TypeError(f"{path} must contain a JSON array of objects")
    return value


def query_id(record: dict[str, Any]) -> int:
    value = record.get("sample_index", record.get("query_id"))
    if value is None:
        raise KeyError("Outcome record is missing sample_index/query_id")
    return int(value)


def doc_id(record: dict[str, Any]) -> int:
    value = record.get("gold_doc_id", record.get("doc_id"))
    if value is None:
        raise KeyError("Outcome record is missing gold_doc_id/doc_id")
    return int(value)


def gold_docid(record: dict[str, Any]) -> tuple[int, ...]:
    value = record.get("gold_docid")
    if not isinstance(value, list) or not value:
        raise ValueError("Outcome record has no valid gold_docid")
    return tuple(int(token) for token in value)


def validate_inputs(
    success: Sequence[dict[str, Any]], failures: dict[int, list[dict[str, Any]]]
) -> None:
    seen_success_ids: set[int] = set()
    for row in success:
        row_id = query_id(row)
        if row_id in seen_success_ids:
            raise ValueError(f"Duplicate Success query_id: {row_id}")
        seen_success_ids.add(row_id)
        if not bool(row.get("success")):
            raise ValueError(f"Success input contains failed query_id {row_id}")
        gold_docid(row)

    for position, rows in failures.items():
        seen_failure_ids: set[int] = set()
        for row in rows:
            row_id = query_id(row)
            if row_id in seen_failure_ids:
                raise ValueError(f"Duplicate F{position} query_id: {row_id}")
            seen_failure_ids.add(row_id)
            if bool(row.get("success")):
                raise ValueError(f"F{position} contains successful query_id {row_id}")
            if int(row.get("first_error_position", -1)) != position:
                raise ValueError(
                    f"F{position} query_id {row_id} has first_error_position="
                    f"{row.get('first_error_position')}"
                )
            if len(gold_docid(row)) < position:
                raise ValueError(f"F{position} query_id {row_id} has a short gold DocID")


def compute_document_frequencies(
    success: Sequence[dict[str, Any]], failure: Sequence[dict[str, Any]]
) -> Counter[int]:
    """Frequency means query occurrences of a gold document in the full dev set."""
    return Counter(doc_id(row) for row in [*success, *failure])


def get_query_length(record: dict[str, Any], mode: str) -> int:
    query = str(record.get("query", "")).strip()
    return len(query) if mode == "characters" else len(query.split())


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def structural_score(
    failure: dict[str, Any], success: dict[str, Any], position: int
) -> tuple[int, int, int]:
    """Lexicographic prefix score; lower is better."""
    failure_prior = gold_docid(failure)[: position - 1]
    success_prior = gold_docid(success)[: position - 1]
    common = common_prefix_length(failure_prior, success_prior)
    hamming = sum(
        left != right for left, right in zip(failure_prior, success_prior)
    ) + abs(len(failure_prior) - len(success_prior))
    cluster_mismatch = int(gold_docid(failure)[0] != gold_docid(success)[0])
    return -common, hamming, cluster_mismatch


def covariate_score(
    failure: dict[str, Any],
    success: dict[str, Any],
    query_lengths: dict[int, int],
    document_frequencies: Counter[int],
) -> tuple[float, float, int]:
    query_difference = abs(
        query_lengths[query_id(failure)] - query_lengths[query_id(success)]
    )
    frequency_difference = abs(
        math.log1p(document_frequencies[doc_id(failure)])
        - math.log1p(document_frequencies[doc_id(success)])
    )
    return float(query_difference), float(frequency_difference), query_id(success)


def choose_candidate(
    failure: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    position: int,
    usage: Counter[int],
    query_lengths: dict[int, int],
    document_frequencies: Counter[int],
    *,
    compare_structure: bool,
    unused_only: bool,
) -> dict[str, Any] | None:
    eligible = [
        candidate
        for candidate in candidates
        if not unused_only or usage[query_id(candidate)] == 0
    ]
    if not eligible:
        return None

    def score(candidate: dict[str, Any]) -> tuple[Any, ...]:
        structural: tuple[int, ...] = (
            structural_score(failure, candidate, position)
            if compare_structure
            else ()
        )
        # For reused controls, spread reuse within the best structural class.
        reuse = () if unused_only else (usage[query_id(candidate)],)
        return (*structural, *reuse, *covariate_score(
            failure, candidate, query_lengths, document_frequencies
        ))

    return min(eligible, key=score)


def match_group(
    position: int,
    failures: Sequence[dict[str, Any]],
    successes: Sequence[dict[str, Any]],
    query_lengths: dict[int, int],
    document_frequencies: Counter[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Match one Sr group with target-token equality as a hard constraint."""
    route_candidates: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    target_candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for success in successes:
        gold = gold_docid(success)
        if len(gold) < position:
            continue
        route_candidates[gold[:position]].append(success)
        target_candidates[gold[position - 1]].append(success)
    for candidates in [*route_candidates.values(), *target_candidates.values()]:
        candidates.sort(key=query_id)

    usage: Counter[int] = Counter()
    controls: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    strategy_counts: Counter[str] = Counter()
    query_differences: list[int] = []
    frequency_differences: list[float] = []
    prior_common_lengths: list[int] = []

    # Rare routes are matched first so common routes cannot consume their few
    # exact candidates through a relaxed target-only match.
    ordered_failures = sorted(
        failures,
        key=lambda row: (
            len(route_candidates.get(gold_docid(row)[:position], [])),
            query_id(row),
        ),
    )
    for control_index, failure in enumerate(ordered_failures):
        failure_gold = gold_docid(failure)
        route = failure_gold[:position]
        target = failure_gold[position - 1]
        exact_pool = route_candidates.get(route, [])

        matched = choose_candidate(
            failure,
            exact_pool,
            position,
            usage,
            query_lengths,
            document_frequencies,
            compare_structure=False,
            unused_only=True,
        )
        strategy = "exact_unused"
        if matched is None and exact_pool:
            matched = choose_candidate(
                failure,
                exact_pool,
                position,
                usage,
                query_lengths,
                document_frequencies,
                compare_structure=False,
                unused_only=False,
            )
            strategy = "exact_reused"
        if matched is None:
            target_pool = target_candidates.get(target, [])
            matched = choose_candidate(
                failure,
                target_pool,
                position,
                usage,
                query_lengths,
                document_frequencies,
                compare_structure=True,
                unused_only=True,
            )
            strategy = "relaxed_prefix_unused"
        if matched is None:
            target_pool = target_candidates.get(target, [])
            matched = choose_candidate(
                failure,
                target_pool,
                position,
                usage,
                query_lengths,
                document_frequencies,
                compare_structure=True,
                unused_only=False,
            )
            strategy = "relaxed_prefix_reused"
        if matched is None:
            raise RuntimeError(
                f"F{position} query_id {query_id(failure)} has target token {target}, "
                "but Success contains no candidate with that token"
            )

        matched_id = query_id(matched)
        usage_before = usage[matched_id]
        usage[matched_id] += 1
        strategy_counts[strategy] += 1
        matched_gold = gold_docid(matched)
        exact_route = matched_gold[:position] == route
        prior_common = common_prefix_length(
            failure_gold[: position - 1], matched_gold[: position - 1]
        )
        query_difference = abs(
            query_lengths[query_id(failure)] - query_lengths[matched_id]
        )
        failure_frequency = document_frequencies[doc_id(failure)]
        success_frequency = document_frequencies[doc_id(matched)]
        frequency_difference = abs(
            math.log1p(failure_frequency) - math.log1p(success_frequency)
        )
        diagnostics = {
            "control_row_id": control_index,
            "group": f"S{position}",
            "matched_failure_query_id": query_id(failure),
            "failure_group": f"F{position}",
            "matching_strategy": strategy,
            "success_usage_number": usage_before + 1,
            "exact_gold_route_through_target": exact_route,
            "prior_prefix_common_length": prior_common,
            "prior_prefix_length": position - 1,
            "target_token": target,
            "query_length_difference": query_difference,
            "failure_document_frequency": failure_frequency,
            "success_document_frequency": success_frequency,
            "log_document_frequency_difference": frequency_difference,
            "document_cluster_match": failure_gold[0] == matched_gold[0],
        }
        control = dict(matched)
        control["matching"] = diagnostics
        controls.append(control)
        pairs.append(
            {
                **diagnostics,
                "success_query_id": matched_id,
                "failure_gold_docid": list(failure_gold),
                "success_gold_docid": list(matched_gold),
            }
        )
        query_differences.append(query_difference)
        frequency_differences.append(frequency_difference)
        prior_common_lengths.append(prior_common)

    target_failure = Counter(gold_docid(row)[position - 1] for row in failures)
    target_control = Counter(gold_docid(row)[position - 1] for row in controls)
    route_supply = Counter(gold_docid(row)[:position] for row in successes)
    route_demand = Counter(gold_docid(row)[:position] for row in failures)
    exact_without_replacement_upper = sum(
        min(count, route_supply[route]) for route, count in route_demand.items()
    )
    exact_with_replacement_upper = sum(
        count for route, count in route_demand.items() if route_supply[route] > 0
    )
    exact_count = sum(pair["exact_gold_route_through_target"] for pair in pairs)
    cluster_count = sum(pair["document_cluster_match"] for pair in pairs)
    unique_successes = len(usage)
    report = {
        "success_group": f"S{position}",
        "failure_group": f"F{position}",
        "n_failure": len(failures),
        "n_matched_success_rows": len(controls),
        "equal_group_size": len(controls) == len(failures),
        "n_unique_success_queries": unique_successes,
        "n_reused_rows": len(controls) - unique_successes,
        "max_uses_of_one_success": max(usage.values(), default=0),
        "target_token_distribution_identical": target_failure == target_control,
        "target_token_match_rate": 1.0,
        "exact_gold_route_match_count": exact_count,
        "exact_gold_route_match_rate": exact_count / len(failures),
        "exact_match_upper_bound_without_replacement": (
            exact_without_replacement_upper / len(failures)
        ),
        "exact_match_upper_bound_with_replacement": (
            exact_with_replacement_upper / len(failures)
        ),
        "document_cluster_match_rate": cluster_count / len(failures),
        "mean_prior_prefix_common_length": mean(prior_common_lengths),
        "mean_query_length_difference": mean(query_differences),
        "median_query_length_difference": median(query_differences),
        "mean_log_document_frequency_difference": mean(frequency_differences),
        "matching_strategy_counts": dict(sorted(strategy_counts.items())),
        "target_token_distribution": dict(sorted(target_failure.items())),
    }
    return controls, pairs, report


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    success_path = args.split_root / "success.json"
    failure_path = args.split_root / "failure.json"
    required = [success_path, failure_path]
    required.extend(args.split_root / f"failure_{position}.json" for position in GROUPS)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_files = [args.output_dir / "matching_report.json"]
    for position in GROUPS:
        output_files.extend(
            [
                args.output_dir / f"S{position}.json",
                args.output_dir / f"S{position}_pairs.jsonl",
            ]
        )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists ({existing[0]}). Pass --overwrite to replace it."
        )

    success = load_json(success_path)
    all_failure = load_json(failure_path)
    failures = {
        position: load_json(args.split_root / f"failure_{position}.json")
        for position in GROUPS
    }
    validate_inputs(success, failures)
    document_frequencies = compute_document_frequencies(success, all_failure)
    all_records = [*success, *all_failure]
    query_lengths = {
        query_id(row): get_query_length(row, args.query_length) for row in all_records
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for position in GROUPS:
        controls, pairs, report = match_group(
            position,
            failures[position],
            success,
            query_lengths,
            document_frequencies,
        )
        write_json(args.output_dir / f"S{position}.json", controls)
        with (args.output_dir / f"S{position}_pairs.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        reports[f"S{position}_vs_F{position}"] = report
        print(
            f"S{position} vs F{position}: n={report['n_failure']:,}, "
            f"exact={report['exact_gold_route_match_rate']:.2%}, "
            f"target={report['target_token_match_rate']:.2%}, "
            f"unique_success={report['n_unique_success_queries']:,}"
        )

    full_report = {
        "matching_policy": {
            "hard_constraint": "current target token is identical",
            "priority": [
                "exact gold route through current target",
                "maximum common prior-prefix length",
                "same document cluster",
                "nearest query length",
                "nearest log document frequency",
            ],
            "replacement": (
                "prefer unused Success; reuse only to retain an available exact route "
                "or when no unused same-target Success remains"
            ),
            "query_length_definition": args.query_length,
            "document_frequency_definition": (
                "number of dev queries whose gold_doc_id is the document"
            ),
        },
        "source": {
            "success": str(success_path),
            "failure": str(failure_path),
            "n_success_pool": len(success),
            "n_all_failure": len(all_failure),
        },
        "groups": reports,
    }
    write_json(args.output_dir / "matching_report.json", full_report)
    print(f"Saved matched controls and diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
