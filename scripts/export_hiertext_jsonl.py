from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from polygon_qwen.hiertext import load_hiertext_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HierText GT into one-record-per-image JSONL.")
    parser.add_argument("--hiertext-root", type=Path, default=Path("data/hiertext"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/hiertext/repo/gt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/hiertext/jsonl"))
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--drop-empty-illegible", action="store_true")
    return parser.parse_args()


def to_portable_path(path: Path, *, path_root: Path) -> str:
    path = path.resolve()
    path_root = path_root.resolve()
    try:
        return path.relative_to(path_root).as_posix()
    except ValueError:
        return path.as_posix()


def extract_ocr_lines(annotation: dict[str, Any], *, include_empty: bool) -> list[dict[str, Any]]:
    ocr_lines: list[dict[str, Any]] = []
    for paragraph_id, paragraph in enumerate(annotation.get("paragraphs", [])):
        for line in paragraph.get("lines", []):
            polygon = line.get("vertices")
            if not polygon:
                continue
            text = str(line.get("text") or "")
            if not include_empty and not text and not line.get("legible", True):
                continue
            ocr_lines.append(
                {
                    "text": text,
                    "polygon": polygon,
                    "paragraph_id": paragraph_id,
                }
            )
    return ocr_lines


def export_split(
    *,
    split: str,
    hiertext_root: Path,
    gt_root: Path,
    output_dir: Path,
    path_root: Path,
    include_empty: bool,
) -> tuple[Path, int, int]:
    annotations = load_hiertext_annotations(gt_root / f"{split}.jsonl.gz")
    image_dir = hiertext_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            image_path = image_dir / f"{annotation['image_id']}.jpg"
            if not image_path.exists():
                skipped += 1
                continue
            ocr_lines = extract_ocr_lines(annotation, include_empty=include_empty)
            if not ocr_lines:
                skipped += 1
                continue
            record = {
                "img_path": to_portable_path(image_path, path_root=path_root),
                "ocr_lines": ocr_lines,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    return output_path, written, skipped


def main() -> None:
    args = parse_args()
    for split in args.splits:
        output_path, written, skipped = export_split(
            split=split,
            hiertext_root=args.hiertext_root,
            gt_root=args.gt_root,
            output_dir=args.output_dir,
            path_root=args.path_root,
            include_empty=not args.drop_empty_illegible,
        )
        print(f"{split}: wrote {written} records to {output_path.as_posix()} skipped={skipped}")


if __name__ == "__main__":
    main()
