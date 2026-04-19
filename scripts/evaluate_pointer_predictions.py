from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polygon_qwen import (
    evaluate_pointer_outputs,
    ocr_lines_to_pointer_text,
    parse_pointer_output,
    pointers_to_pointer_text,
    sanitize_pointer_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paragraph pointer predictions.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--gt-field", type=str, default="gt_answer")
    parser.add_argument("--prediction-field", type=str, default="model_answer")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_records(path: Path, *, gt_field: str, prediction_field: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload: dict[str, Any] = json.loads(line)
            if "gt_ocr_lines" in payload and "ocr_lines" in payload:
                records.append(
                    (
                        ocr_lines_to_pointer_text(payload["gt_ocr_lines"]),
                        ocr_lines_to_pointer_text(payload["ocr_lines"]),
                    )
                )
                continue

            if gt_field not in payload:
                raise KeyError(f"Missing gt field {gt_field!r} at line {line_number}.")
            if prediction_field not in payload:
                raise KeyError(f"Missing prediction field {prediction_field!r} at line {line_number}.")
            gt_text = str(payload[gt_field])
            gt_parse = parse_pointer_output(gt_text)
            expected_line_ids = list(gt_parse.pointers.keys())
            sanitized = sanitize_pointer_output(
                str(payload[prediction_field]),
                expected_line_ids=expected_line_ids,
            )
            records.append(
                (
                    gt_text,
                    pointers_to_pointer_text(sanitized.pointers, expected_line_ids=expected_line_ids),
                )
            )
    return records


def main() -> None:
    args = parse_args()
    records = load_records(
        args.predictions_jsonl,
        gt_field=args.gt_field,
        prediction_field=args.prediction_field,
    )
    metrics = evaluate_pointer_outputs(records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
