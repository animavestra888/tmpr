from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter HierText JSONL records by maximum number of OCR lines."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/hiertext/jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hiertext/jsonl_max300"))
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    return parser.parse_args()


def line_count(record: dict[str, Any]) -> int:
    return len(record.get("ocr_lines", []))


def filter_split(*, input_path: Path, output_path: Path, max_lines: int) -> dict[str, int]:
    total = 0
    kept = 0
    dropped = 0
    max_seen = 0
    max_kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for line in input_handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            count = line_count(record)
            max_seen = max(max_seen, count)
            if count <= max_lines:
                output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
                max_kept = max(max_kept, count)
            else:
                dropped += 1

    return {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "max_seen": max_seen,
        "max_kept": max_kept,
    }


def main() -> None:
    args = parse_args()
    if args.max_lines <= 0:
        raise ValueError("--max-lines must be positive.")

    for split in args.splits:
        input_path = args.input_dir / f"{split}.jsonl"
        output_path = args.output_dir / f"{split}.jsonl"
        stats = filter_split(input_path=input_path, output_path=output_path, max_lines=args.max_lines)
        print(
            json.dumps(
                {
                    "split": split,
                    "input": input_path.as_posix(),
                    "output": output_path.as_posix(),
                    "max_lines": args.max_lines,
                    **stats,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
