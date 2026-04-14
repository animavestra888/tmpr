from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_POINTER_PATTERN = re.compile(r"(\d+)\s*->\s*(\d+)")


@dataclass(frozen=True)
class PointerParseResult:
    pointers: dict[str, str]
    duplicate_sources: frozenset[str]


def _canonical_line_id(value: str) -> str:
    return str(int(value))


def parse_pointer_output(text: str) -> PointerParseResult:
    """Extract pointer links from model output such as `0->1`."""

    pointers: dict[str, str] = {}
    duplicate_sources: set[str] = set()
    for source, target in _POINTER_PATTERN.findall(text):
        source_id = _canonical_line_id(source)
        target_id = _canonical_line_id(target)
        if source_id in pointers:
            duplicate_sources.add(source_id)
        pointers[source_id] = target_id
    return PointerParseResult(
        pointers=pointers,
        duplicate_sources=frozenset(duplicate_sources),
    )


def _sort_cluster_key(cluster: frozenset[str]) -> tuple[int, int]:
    return min(int(line_id) for line_id in cluster), len(cluster)


def pointers_to_clusters(
    pointers: dict[str, str],
    *,
    expected_line_ids: Iterable[str],
) -> tuple[frozenset[str], ...] | None:
    """Convert pointer links into an unordered partition of line ids.

    The grouping is evaluated from connectivity: `0->1` means lines 0 and 1
    belong to the same paragraph. Self-pointers keep singleton paragraphs.
    """

    expected_ids = frozenset(_canonical_line_id(line_id) for line_id in expected_line_ids)
    if set(pointers) != expected_ids:
        return None
    if any(target not in expected_ids for target in pointers.values()):
        return None

    parent = {line_id: line_id for line_id in expected_ids}

    def find(line_id: str) -> str:
        while parent[line_id] != line_id:
            parent[line_id] = parent[parent[line_id]]
            line_id = parent[line_id]
        return line_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, target in pointers.items():
        union(source, target)

    clusters: dict[str, set[str]] = {}
    for line_id in expected_ids:
        clusters.setdefault(find(line_id), set()).add(line_id)

    return tuple(
        sorted(
            (frozenset(cluster) for cluster in clusters.values()),
            key=_sort_cluster_key,
        )
    )


def line_accuracy(gold_text: str, prediction_text: str) -> float:
    gold = parse_pointer_output(gold_text).pointers
    prediction = parse_pointer_output(prediction_text).pointers
    if not gold:
        return 1.0 if not prediction else 0.0
    correct = sum(1 for source, target in gold.items() if prediction.get(source) == target)
    return correct / len(gold)


def global_accuracy(gold_text: str, prediction_text: str) -> float:
    gold_parse = parse_pointer_output(gold_text)
    pred_parse = parse_pointer_output(prediction_text)
    if pred_parse.duplicate_sources:
        return 0.0

    expected_ids = gold_parse.pointers.keys()
    gold_clusters = pointers_to_clusters(gold_parse.pointers, expected_line_ids=expected_ids)
    pred_clusters = pointers_to_clusters(pred_parse.pointers, expected_line_ids=expected_ids)
    return 1.0 if pred_clusters is not None and pred_clusters == gold_clusters else 0.0


def evaluate_pointer_outputs(records: Iterable[tuple[str, str]]) -> dict[str, float]:
    total_samples = 0
    total_lines = 0
    correct_lines = 0
    global_correct = 0
    valid_predictions = 0

    for gold_text, prediction_text in records:
        total_samples += 1
        gold_parse = parse_pointer_output(gold_text)
        pred_parse = parse_pointer_output(prediction_text)
        expected_ids = gold_parse.pointers.keys()

        total_lines += len(gold_parse.pointers)
        correct_lines += sum(
            1
            for source, target in gold_parse.pointers.items()
            if pred_parse.pointers.get(source) == target
        )

        pred_clusters = None
        if not pred_parse.duplicate_sources:
            pred_clusters = pointers_to_clusters(pred_parse.pointers, expected_line_ids=expected_ids)
        if pred_clusters is not None:
            valid_predictions += 1

        gold_clusters = pointers_to_clusters(gold_parse.pointers, expected_line_ids=expected_ids)
        if pred_clusters is not None and pred_clusters == gold_clusters:
            global_correct += 1

    if total_samples == 0:
        return {
            "global_accuracy": 0.0,
            "line_accuracy": 0.0,
            "valid_prediction_rate": 0.0,
            "num_samples": 0.0,
            "num_lines": 0.0,
        }

    return {
        "global_accuracy": global_correct / total_samples,
        "line_accuracy": correct_lines / total_lines if total_lines else 0.0,
        "valid_prediction_rate": valid_predictions / total_samples,
        "num_samples": float(total_samples),
        "num_lines": float(total_lines),
    }
