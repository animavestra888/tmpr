from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import EMBEDDING_GEOMETRIES, polygon_to_bbox_2d, polygon_to_embedding_coords

BBOX_COORD_DIM = 8


def load_hiertext_annotations(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["annotations"]


def load_hiertext_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            records.append(record)
    return records


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _center_key(vertices: list[list[float]]) -> tuple[float, float, float, float]:
    points = np.asarray(vertices, dtype=np.float32)
    min_xy = points.min(axis=0)
    center_xy = points.mean(axis=0)
    return float(center_xy[1]), float(center_xy[0]), float(min_xy[1]), float(min_xy[0])


def _round_coords(coords: np.ndarray, precision: int) -> list[float]:
    return [round(float(value), precision) for value in coords.tolist()]


def _safe_text(line: dict[str, Any]) -> str:
    text = str(line.get("text") or "").strip()
    if not text:
        return "<none>"
    return text


def _line_id(index: int) -> str:
    return str(index)


def _line_sort_value(line_id: str) -> int:
    return int(line_id)


@dataclass(slots=True)
class HierTextLine:
    paragraph_index: int
    source_line_index: int
    vertices: list[list[float]]
    text: str
    legible: bool
    handwritten: bool
    vertical: bool
    line_id: str = ""
    coords: list[float] | None = None
    bbox_2d: list[int] | None = None


class HierTextParagraphClusteringDataset(Dataset):
    """Create paragraph-clustering SFT examples from HierText annotations."""

    def __init__(
        self,
        *,
        gt_json: str | Path | None = None,
        image_dir: str | Path | None = None,
        jsonl_path: str | Path | None = None,
        path_root: str | Path = ".",
        limit: int | None = None,
        include_illegible: bool = True,
        polygon_mode: str = "text",
        embedding_geometry: str = "bbox_corners",
        poly_token: str = "<poly>",
        coord_precision: int = 4,
        bbox_scale: int = 1000,
    ) -> None:
        if jsonl_path is None and (gt_json is None or image_dir is None):
            raise ValueError("Provide either jsonl_path or both gt_json and image_dir.")

        self.gt_json = Path(gt_json) if gt_json is not None else None
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.path_root = Path(path_root)
        self.include_illegible = include_illegible
        if polygon_mode not in {"text", "embedding"}:
            raise ValueError("polygon_mode must be either 'text' or 'embedding'.")
        self.polygon_mode = polygon_mode
        if embedding_geometry not in EMBEDDING_GEOMETRIES:
            choices = ", ".join(EMBEDDING_GEOMETRIES)
            raise ValueError(f"embedding_geometry must be one of: {choices}.")
        self.embedding_geometry = embedding_geometry
        self.poly_token = poly_token
        self.coord_precision = coord_precision
        self.bbox_scale = bbox_scale

        if self.jsonl_path is not None:
            self.records = self._filter_jsonl_records(load_hiertext_jsonl(self.jsonl_path), limit=limit)
        else:
            assert self.gt_json is not None
            self.records = self._filter_annotations(load_hiertext_annotations(self.gt_json), limit=limit)
        if not self.records:
            raise ValueError("No usable HierText records found.")

    def _extract_annotation_lines(self, annotation: dict[str, Any]) -> list[HierTextLine]:
        lines: list[HierTextLine] = []
        for paragraph_index, paragraph in enumerate(annotation.get("paragraphs", [])):
            for source_line_index, line in enumerate(paragraph.get("lines", [])):
                vertices = line.get("vertices")
                if not vertices:
                    continue
                legible = bool(line.get("legible", True))
                if not legible and not self.include_illegible:
                    continue
                lines.append(
                    HierTextLine(
                        paragraph_index=paragraph_index,
                        source_line_index=source_line_index,
                        vertices=vertices,
                        text=_safe_text(line),
                        legible=legible,
                        handwritten=bool(line.get("handwritten", False)),
                        vertical=bool(line.get("vertical", False)),
                    )
                )
        lines.sort(key=lambda item: _center_key(item.vertices))
        for index, line in enumerate(lines):
            line.line_id = _line_id(index)
        return lines

    def _extract_jsonl_lines(self, record: dict[str, Any]) -> list[HierTextLine]:
        lines: list[HierTextLine] = []
        for source_line_index, line in enumerate(record.get("ocr_lines", [])):
            vertices = line.get("polygon")
            if not vertices:
                continue
            paragraph_id = int(line.get("paragraph_id", -1))
            lines.append(
                HierTextLine(
                    paragraph_index=paragraph_id,
                    source_line_index=source_line_index,
                    vertices=vertices,
                    text=_safe_text(line),
                    legible=True,
                    handwritten=False,
                    vertical=False,
                )
            )
        lines.sort(key=lambda item: _center_key(item.vertices))
        for index, line in enumerate(lines):
            line.line_id = _line_id(index)
        return lines

    def _resolve_jsonl_image_path(self, record: dict[str, Any]) -> Path:
        image_path = Path(record["img_path"])
        if image_path.is_absolute():
            return image_path
        return self.path_root / image_path

    def _filter_annotations(
        self,
        annotations: list[dict[str, Any]],
        *,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for annotation in annotations:
            assert self.image_dir is not None
            image_path = self.image_dir / f"{annotation['image_id']}.jpg"
            if not image_path.exists():
                continue
            lines = self._extract_annotation_lines(annotation)
            if not self._is_usable_lines(lines):
                continue
            records.append({"source": "annotation", "payload": annotation})
            if limit is not None and len(records) >= limit:
                break
        return records

    def _filter_jsonl_records(
        self,
        jsonl_records: list[dict[str, Any]],
        *,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for record in jsonl_records:
            image_path = self._resolve_jsonl_image_path(record)
            if not image_path.exists():
                continue
            lines = self._extract_jsonl_lines(record)
            if not self._is_usable_lines(lines):
                continue
            records.append({"source": "jsonl", "payload": record})
            if limit is not None and len(records) >= limit:
                break
        return records

    def _is_usable_lines(self, lines: list[HierTextLine]) -> bool:
        return bool(lines)

    def __len__(self) -> int:
        return len(self.records)

    def _build_target(self, lines: list[HierTextLine]) -> str:
        clusters: dict[int, list[str]] = {}
        for line in lines:
            clusters.setdefault(line.paragraph_index, []).append(line.line_id)
        ordered_clusters = sorted(clusters.values(), key=lambda ids: _line_sort_value(ids[0]))

        pointers: dict[str, str] = {}
        for cluster in ordered_clusters:
            for index, line_id in enumerate(cluster):
                next_id = cluster[index + 1] if index + 1 < len(cluster) else line_id
                pointers[line_id] = next_id

        return "\n".join(
            f"{line.line_id}->{pointers[line.line_id]}"
            for line in sorted(lines, key=lambda item: _line_sort_value(item.line_id))
        )

    def _build_prompt(
        self,
        *,
        image_id: str,
        image_width: int,
        image_height: int,
        lines: list[HierTextLine],
    ) -> str:
        ocr_lines: list[dict[str, Any]] = []
        for line in lines:
            if self.polygon_mode == "embedding":
                ocr_lines.append(
                    {
                        "id": int(line.line_id),
                        "text": line.text,
                        "polygon": self.poly_token,
                    }
                )
            else:
                ocr_lines.append(
                    {
                        "id": int(line.line_id),
                        "text": line.text,
                        "bbox_2d": line.bbox_2d or [],
                    }
                )

        if self.polygon_mode == "embedding":
            geometry_description = (
                "Each line has a stable id, recognized text when present, and a polygon. "
                "The polygon placeholder is replaced by a learned embedding of that line's normalized geometry."
            )
        else:
            geometry_description = (
                "Each line has a stable id, recognized text when present, and bbox_2d. "
                f"bbox_2d is [x1,y1,x2,y2] normalized to [0,{self.bbox_scale}]."
            )

        return (
            "Task: cluster OCR text lines into paragraphs.\n"
            "You are given the image and a list of OCR line candidates. "
            f"{geometry_description}\n"
            "Answer in pointer format, one pointer per line. A line points to itself when it ends a paragraph.\n"
            "Example:\n"
            "0->1\n"
            "1->3\n"
            "2->2\n"
            "3->3\n"
            "This represents two paragraphs: [0, 1, 3] and [2].\n"
            "Do not return markdown or explanations.\n\n"
            "ocr_lines:\n"
            + json.dumps(ocr_lines, ensure_ascii=False, separators=(",", ":"))
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record["source"] == "jsonl":
            payload = record["payload"]
            image_path = self._resolve_jsonl_image_path(payload)
            image_id = image_path.stem
            lines = self._extract_jsonl_lines(payload)
        else:
            payload = record["payload"]
            assert self.image_dir is not None
            image_path = self.image_dir / f"{payload['image_id']}.jpg"
            image_id = payload["image_id"]
            lines = self._extract_annotation_lines(payload)

        image = _load_image(image_path)
        width, height = image.size

        for line in lines:
            coords = polygon_to_embedding_coords(
                line.vertices,
                image_width=width,
                image_height=height,
                embedding_geometry=self.embedding_geometry,
            )
            line.coords = _round_coords(coords, self.coord_precision)
            line.bbox_2d = polygon_to_bbox_2d(
                line.vertices,
                image_width=width,
                image_height=height,
                scale=self.bbox_scale,
            )

        polygon_coords = np.asarray([line.coords for line in lines], dtype=np.float32)
        return {
            "image": image,
            "image_id": image_id,
            "img_path": image_path.as_posix(),
            "prompt": self._build_prompt(
                image_id=image_id,
                image_width=width,
                image_height=height,
                lines=lines,
            ),
            "answer": self._build_target(lines),
            "num_lines": len(lines),
            "polygon_coords": polygon_coords,
        }


@dataclass(slots=True)
class HierTextParagraphCollator:
    processor: Any
    use_polygon_embeddings: bool = False

    def _messages(self, prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [example["image"] for example in examples]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._messages(example["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]
        full_texts = [
            self.processor.apply_chat_template(
                self._messages(example["prompt"], example["answer"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]

        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100

        mm_token_type_ids = batch.pop("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = batch.pop("token_type_ids", None)
        if mm_token_type_ids is not None:
            batch["mm_token_type_ids"] = mm_token_type_ids

        batch["labels"] = labels
        if self.use_polygon_embeddings:
            polygon_counts = torch.tensor(
                [len(example["polygon_coords"]) for example in examples],
                dtype=torch.long,
            )
            max_polygons = int(polygon_counts.max().item())
            polygon_coords = torch.zeros(
                len(examples),
                max_polygons,
                BBOX_COORD_DIM,
                dtype=torch.float32,
            )
            for row_index, example in enumerate(examples):
                coords = torch.as_tensor(example["polygon_coords"], dtype=torch.float32)
                polygon_coords[row_index, : coords.shape[0]] = coords
            batch["polygon_coords"] = polygon_coords
            batch["polygon_counts"] = polygon_counts
        return batch
