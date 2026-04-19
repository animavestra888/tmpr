from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_POINTER_PATTERN = re.compile(r"(\d+)\s*->\s*(\d+)")


@dataclass(frozen=True)
class PointerParseResult:
    pointers: dict[str, str]
    duplicate_sources: frozenset[str]


@dataclass(frozen=True)
class SanitizedPointerResult:
    pointers: dict[str, str]
    is_valid: bool
    repairs: tuple[str, ...]


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


def _choose_predecessor_to_keep(
    sources: list[str],
    *,
    target: str,
    id_order: dict[str, int],
) -> str:
    target_order = id_order[target]

    def key(source: str) -> tuple[int, int, int]:
        source_order = id_order[source]
        if source_order < target_order:
            return 0, target_order - source_order, source_order
        return 1, abs(source_order - target_order), source_order

    return min(sources, key=key)


def _find_cycle_start(
    pointers: dict[str, str],
    *,
    expected_line_ids: list[str],
    expected_ids: set[str],
) -> list[str] | None:
    for start in expected_line_ids:
        path: list[str] = []
        seen: dict[str, int] = {}
        current = start
        while current in expected_ids:
            if current in seen:
                return path[seen[current] :]
            seen[current] = len(path)
            path.append(current)
            target = pointers[current]
            if target == current:
                break
            current = target
    return None


def sanitize_pointer_output(
    text: str,
    *,
    expected_line_ids: Iterable[str],
) -> SanitizedPointerResult:
    """Repair malformed pointer chains into a deterministic paragraph partition.

    Repairs are intentionally conservative: missing/invalid outputs become
    self-pointers, ambiguous multiple parents keep one predecessor, and cycles
    are cut by making the latest line in the cycle point to itself.
    """

    line_ids = [_canonical_line_id(line_id) for line_id in expected_line_ids]
    expected_ids = set(line_ids)
    id_order = {line_id: index for index, line_id in enumerate(line_ids)}
    parsed = parse_pointer_output(text)
    repairs: list[str] = []
    pointers: dict[str, str] = {}

    for line_id in line_ids:
        target = parsed.pointers.get(line_id)
        if line_id in parsed.duplicate_sources:
            pointers[line_id] = line_id
            repairs.append("duplicate_source")
        elif target is None:
            pointers[line_id] = line_id
            repairs.append("missing_source")
        elif target not in expected_ids:
            pointers[line_id] = line_id
            repairs.append("invalid_target")
        else:
            pointers[line_id] = target

    for source in parsed.pointers:
        if source not in expected_ids:
            repairs.append("unexpected_source")

    incoming: dict[str, list[str]] = {line_id: [] for line_id in line_ids}
    for source, target in pointers.items():
        # Self-pointers are paragraph ends, not another parent pointing into the line.
        if source != target:
            incoming[target].append(source)

    for target, sources in incoming.items():
        if len(sources) <= 1:
            continue
        keep_source = _choose_predecessor_to_keep(sources, target=target, id_order=id_order)
        for source in sources:
            if source == keep_source:
                continue
            pointers[source] = source
            repairs.append("ambiguous_parent")

    while True:
        cycle = _find_cycle_start(
            pointers,
            expected_line_ids=line_ids,
            expected_ids=expected_ids,
        )
        if cycle is None:
            break
        end_line_id = max(cycle, key=lambda line_id: id_order[line_id])
        pointers[end_line_id] = end_line_id
        repairs.append("cycle")

    return SanitizedPointerResult(
        pointers=pointers,
        is_valid=not repairs,
        repairs=tuple(repairs),
    )


def pointers_to_pointer_text(
    pointers: dict[str, str],
    *,
    expected_line_ids: Iterable[str],
) -> str:
    line_ids = [_canonical_line_id(line_id) for line_id in expected_line_ids]
    return "\n".join(f"{line_id}->{pointers[line_id]}" for line_id in line_ids)


def line_accuracy(gt_text: str, prediction_text: str) -> float:
    gt = parse_pointer_output(gt_text).pointers
    prediction = parse_pointer_output(prediction_text).pointers
    if not gt:
        return 1.0 if not prediction else 0.0
    correct = sum(1 for source, target in gt.items() if prediction.get(source) == target)
    return correct / len(gt)


def global_accuracy(gt_text: str, prediction_text: str) -> float:
    gt_parse = parse_pointer_output(gt_text)
    pred_parse = parse_pointer_output(prediction_text)
    if pred_parse.duplicate_sources:
        return 0.0

    expected_ids = gt_parse.pointers.keys()
    gt_clusters = pointers_to_clusters(gt_parse.pointers, expected_line_ids=expected_ids)
    pred_clusters = pointers_to_clusters(pred_parse.pointers, expected_line_ids=expected_ids)
    return 1.0 if pred_clusters is not None and pred_clusters == gt_clusters else 0.0


def _line_id_from_record(line: dict[str, Any], fallback: int) -> str:
    return _canonical_line_id(str(line.get("id", fallback)))


def ocr_lines_to_pointer_text(ocr_lines: list[dict[str, Any]]) -> str:
    """Convert HierText-style OCR lines with paragraph ids into pointer text."""

    line_ids = [_line_id_from_record(line, index) for index, line in enumerate(ocr_lines)]
    id_order = {line_id: index for index, line_id in enumerate(line_ids)}
    clusters: dict[int, list[str]] = {}
    for line_id, line in zip(line_ids, ocr_lines):
        paragraph_id = int(line.get("paragraph_id", -1))
        clusters.setdefault(paragraph_id, []).append(line_id)

    pointers: dict[str, str] = {}
    ordered_clusters = sorted(
        clusters.values(),
        key=lambda cluster: min(id_order[line_id] for line_id in cluster),
    )
    for cluster in ordered_clusters:
        cluster = sorted(cluster, key=lambda line_id: id_order[line_id])
        for index, line_id in enumerate(cluster):
            next_id = cluster[index + 1] if index + 1 < len(cluster) else line_id
            pointers[line_id] = next_id

    return "\n".join(f"{line_id}->{pointers[line_id]}" for line_id in line_ids)


def evaluate_pointer_outputs(records: Iterable[tuple[str, str]]) -> dict[str, float]:
    total_samples = 0
    total_lines = 0
    correct_lines = 0
    global_correct = 0
    valid_predictions = 0

    for gt_text, prediction_text in records:
        total_samples += 1
        gt_parse = parse_pointer_output(gt_text)
        pred_parse = parse_pointer_output(prediction_text)
        expected_ids = gt_parse.pointers.keys()

        total_lines += len(gt_parse.pointers)
        correct_lines += sum(
            1
            for source, target in gt_parse.pointers.items()
            if pred_parse.pointers.get(source) == target
        )

        pred_clusters = None
        if not pred_parse.duplicate_sources:
            pred_clusters = pointers_to_clusters(pred_parse.pointers, expected_line_ids=expected_ids)
        if pred_clusters is not None:
            valid_predictions += 1

        gt_clusters = pointers_to_clusters(gt_parse.pointers, expected_line_ids=expected_ids)
        if pred_clusters is not None and pred_clusters == gt_clusters:
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
