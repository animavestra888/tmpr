from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polygon_qwen import evaluate_pointer_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paragraph pointer predictions.")
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--gold-field", type=str, default="answer")
    parser.add_argument("--prediction-field", type=str, default="prediction")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_records(path: Path, *, gold_field: str, prediction_field: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload: dict[str, Any] = json.loads(line)
            if gold_field not in payload:
                raise KeyError(f"Missing gold field {gold_field!r} at line {line_number}.")
            if prediction_field not in payload:
                raise KeyError(f"Missing prediction field {prediction_field!r} at line {line_number}.")
            records.append((str(payload[gold_field]), str(payload[prediction_field])))
    return records


def main() -> None:
    args = parse_args()
    records = load_records(
        args.predictions_jsonl,
        gold_field=args.gold_field,
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
