from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polygon_qwen import EMBEDDING_GEOMETRIES, HierTextParagraphClusteringDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview HierText paragraph-clustering prompts.")
    parser.add_argument("--gt-json", type=Path, default=Path("data/hiertext/repo/gt/train.jsonl.gz"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/hiertext/train"))
    parser.add_argument("--jsonl-path", type=Path, default=None)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--coord-precision", type=int, default=4)
    parser.add_argument("--bbox-scale", type=int, default=1000)
    parser.add_argument("--exclude-illegible", action="store_true")
    parser.add_argument("--polygon-mode", choices=["text", "embedding"], default="text")
    parser.add_argument("--embedding-geometry", choices=list(EMBEDDING_GEOMETRIES), default="bbox_corners")
    parser.add_argument("--poly-token", type=str, default="<poly>")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = HierTextParagraphClusteringDataset(
        gt_json=None if args.jsonl_path else args.gt_json,
        image_dir=None if args.jsonl_path else args.image_dir,
        jsonl_path=args.jsonl_path,
        path_root=args.path_root,
        limit=args.limit,
        include_illegible=not args.exclude_illegible,
        polygon_mode=args.polygon_mode,
        embedding_geometry=args.embedding_geometry,
        poly_token=args.poly_token,
        coord_precision=args.coord_precision,
        bbox_scale=args.bbox_scale,
    )

    print(f"loaded_examples: {len(dataset)}")
    output_handle = None
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output_jsonl.open("w", encoding="utf-8")

    try:
        for index in range(len(dataset)):
            example = dataset[index]
            record = {
                "image_id": example["image_id"],
                "num_lines": example["num_lines"],
                "prompt": example["prompt"],
                "answer": example["answer"],
            }
            if output_handle is not None:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("\n" + "=" * 88)
            print(f"example_index: {index}")
            print(f"image_id: {example['image_id']}")
            print(f"num_lines: {example['num_lines']}")
            print("\nPROMPT\n")
            print(example["prompt"])
            print("\nANSWER\n")
            print(example["answer"])
    finally:
        if output_handle is not None:
            output_handle.close()
            print(f"\nwrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
