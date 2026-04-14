from __future__ import annotations

import argparse
import colorsys
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from polygon_qwen.geometry import order_box_corners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HierText OCR line polygons colored by paragraph cluster."
    )
    parser.add_argument("--jsonl-path", type=Path, default=Path("data/hiertext/jsonl/train.jsonl"))
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hiertext_visualizations"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl-line", type=int, action="append", default=[])
    parser.add_argument("--image-path", type=str, action="append", default=[])
    parser.add_argument("--max-image-side", type=int, default=2200)
    parser.add_argument("--draw-line-ids", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            record["_num_lines"] = len(record.get("ocr_lines", []))
            records.append(record)
    return records


def cluster_color(paragraph_id: int) -> tuple[int, int, int, int]:
    hue = ((paragraph_id * 0.618033988749895) % 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255), 230


def resolve_image_path(path_root: Path, img_path: str) -> Path:
    path = Path(img_path)
    if path.is_absolute():
        return path
    return path_root / path


def resize_for_view(image: Image.Image, max_image_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(1.0, max_image_side / max(width, height))
    if scale == 1.0:
        return image, scale
    return image.resize((round(width * scale), round(height * scale))), scale


def scale_polygon(polygon: list[list[float]], scale: float) -> list[tuple[float, float]]:
    return [(float(x) * scale, float(y) * scale) for x, y in polygon]


def min_area_rect(points_xy: list[list[float]]) -> list[tuple[float, float]]:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError(f"Expected at least three xy points, received shape {points.shape}.")
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float32)
    return [(float(x), float(y)) for x, y in order_box_corners(box)]


def draw_record(
    *,
    record: dict[str, Any],
    path_root: Path,
    output_dir: Path,
    split: str,
    max_image_side: int,
    draw_line_ids: bool,
) -> Path:
    image_path = resolve_image_path(path_root, str(record["img_path"]))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image, scale = resize_for_view(image, max_image_side=max_image_side)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(1, round(max(image.size) / 700))
    font = ImageFont.load_default()

    cluster_counts: dict[int, int] = {}
    cluster_points: dict[int, list[list[float]]] = {}
    drawable_lines: list[tuple[int, int, list[tuple[float, float]]]] = []
    for index, line in enumerate(record.get("ocr_lines", [])):
        polygon = line.get("polygon")
        if not polygon:
            continue
        paragraph_id = int(line.get("paragraph_id", -1))
        cluster_counts[paragraph_id] = cluster_counts.get(paragraph_id, 0) + 1
        cluster_points.setdefault(paragraph_id, []).extend(polygon)
        drawable_lines.append((index, paragraph_id, scale_polygon(polygon, scale=scale)))

    cluster_rect_width = max(line_width + 1, round(max(image.size) / 420))
    for paragraph_id, points_xy in cluster_points.items():
        color = cluster_color(paragraph_id)
        rect_points = scale_polygon(min_area_rect(points_xy), scale=scale)
        fill = color[:3] + (38,)
        outline = color[:3] + (210,)
        draw.polygon(rect_points, fill=fill)
        draw.line(rect_points + [rect_points[0]], fill=outline, width=cluster_rect_width)

    for index, paragraph_id, points in drawable_lines:
        color = cluster_color(paragraph_id)
        fill = color[:3] + (76,)
        outline = color[:3] + (255,)
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=line_width)

        if draw_line_ids:
            x = min(point[0] for point in points)
            y = min(point[1] for point in points)
            draw.text((x, y), str(index), fill=(255, 255, 255, 255), font=font)

    visual = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(visual)
    title = (
        f"{split} | line={record['_line_number']} | lines={record['_num_lines']} | "
        f"paragraphs={len(cluster_counts)} | {record['img_path']}"
    )
    draw.rectangle((0, 0, image.size[0], 28), fill=(0, 0, 0, 170))
    draw.text((8, 8), title, fill=(255, 255, 255, 255), font=font)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_id = Path(str(record["img_path"])).stem
    output_path = output_dir / f"{split}_{record['_line_number']:05d}_{record['_num_lines']:04d}lines_{image_id}.png"
    visual.convert("RGB").save(output_path)
    return output_path


def select_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    by_line = {record["_line_number"]: record for record in records}
    by_image = {str(record["img_path"]).replace("\\", "/"): record for record in records}

    for line_number in args.jsonl_line:
        if line_number in by_line:
            selected[line_number] = by_line[line_number]

    for image_path in args.image_path:
        normalized = image_path.replace("\\", "/")
        if normalized in by_image:
            record = by_image[normalized]
            selected[record["_line_number"]] = record

    remaining = [record for record in records if record["_line_number"] not in selected]
    if args.num_samples > 0 and remaining:
        rng = random.Random(args.seed)
        sample_size = min(args.num_samples, len(remaining))
        for record in rng.sample(remaining, sample_size):
            selected[record["_line_number"]] = record

    return sorted(selected.values(), key=lambda item: item["_line_number"])


def main() -> None:
    args = parse_args()
    records = load_records(args.jsonl_path)
    split = args.jsonl_path.stem
    selected = select_records(args, records)

    print(f"jsonl: {args.jsonl_path.as_posix()}")
    print(f"records: {len(records)}")
    print(f"selected: {len(selected)}")
    for record in selected:
        output_path = draw_record(
            record=record,
            path_root=args.path_root,
            output_dir=args.output_dir,
            split=split,
            max_image_side=args.max_image_side,
            draw_line_ids=args.draw_line_ids,
        )
        paragraph_ids = {int(line.get("paragraph_id", -1)) for line in record.get("ocr_lines", [])}
        print(
            json.dumps(
                {
                    "jsonl_line": record["_line_number"],
                    "img_path": record["img_path"],
                    "num_lines": record["_num_lines"],
                    "num_paragraphs": len(paragraph_ids),
                    "visualization": output_path.as_posix(),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
