from __future__ import annotations

from typing import Sequence

import numpy as np
import cv2

EMBEDDING_GEOMETRIES = ("bbox_corners", "minrect")
EMBEDDING_GEOMETRY_COORD_FORMATS = {
    "bbox_corners": "bbox_corners_xyxy",
    "minrect": "minrect_8coords",
}


def _as_points_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Expected polygon points with shape (N, 2), received {array.shape}."
        )
    if array.shape[0] < 3:
        raise ValueError("A polygon needs at least 3 points.")
    return array


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for minimum-area-rectangle preprocessing."
        )


def embedding_geometry_to_coord_format(embedding_geometry: str) -> str:
    try:
        return EMBEDDING_GEOMETRY_COORD_FORMATS[embedding_geometry]
    except KeyError as exc:
        choices = ", ".join(EMBEDDING_GEOMETRIES)
        raise ValueError(f"Unsupported embedding geometry '{embedding_geometry}'. Choose one of: {choices}.") from exc


def coord_format_to_embedding_geometry(coord_format: str) -> str:
    for embedding_geometry, candidate_coord_format in EMBEDDING_GEOMETRY_COORD_FORMATS.items():
        if candidate_coord_format == coord_format:
            return embedding_geometry
    choices = ", ".join(EMBEDDING_GEOMETRY_COORD_FORMATS.values())
    raise ValueError(f"Unsupported coordinate format '{coord_format}'. Choose one of: {choices}.")


def order_box_corners(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return 4 box corners in a stable clockwise order."""

    box = np.asarray(points, dtype=np.float32)
    if box.shape != (4, 2):
        raise ValueError(f"Expected box corners with shape (4, 2), received {box.shape}.")

    center = box.mean(axis=0)
    angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
    ordered = box[np.argsort(angles)]
    start_index = int(np.argmin(ordered[:, 1] + ordered[:, 0]))
    return np.roll(ordered, -start_index, axis=0).astype(np.float32)


def polygon_to_normalized_bbox(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert an arbitrary polygon to normalized [x1, y1, x2, y2] bbox coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    polygon = _as_points_array(polygon_xy)
    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    bbox = np.asarray(
        [
            min_xy[0] / float(image_width),
            min_xy[1] / float(image_height),
            max_xy[0] / float(image_width),
            max_xy[1] / float(image_height),
        ],
        dtype=np.float32,
    )
    if clip:
        bbox = np.clip(bbox, 0.0, 1.0)
    return bbox.astype(np.float32)


def polygon_to_normalized_bbox_8coords(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert a polygon to normalized axis-aligned bbox corners.

    Output order is [x1,y1, x2,y1, x2,y2, x1,y2].
    """

    x1, y1, x2, y2 = polygon_to_normalized_bbox(
        polygon_xy,
        image_width=image_width,
        image_height=image_height,
        clip=clip,
    ).tolist()
    return np.asarray([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)


def polygon_to_minrect_8coords(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert an arbitrary polygon to normalized minimum-area rectangle coordinates."""

    _require_cv2()
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    polygon = _as_points_array(polygon_xy)
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect).astype(np.float32)
    coords = order_box_corners(box).reshape(8)
    coords[0::2] /= float(image_width)
    coords[1::2] /= float(image_height)
    if clip:
        coords = np.clip(coords, 0.0, 1.0)
    return coords.astype(np.float32)


def polygon_to_embedding_coords(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    embedding_geometry: str,
    clip: bool = True,
) -> np.ndarray:
    if embedding_geometry == "bbox_corners":
        return polygon_to_normalized_bbox_8coords(
            polygon_xy,
            image_width=image_width,
            image_height=image_height,
            clip=clip,
        )
    if embedding_geometry == "minrect":
        return polygon_to_minrect_8coords(
            polygon_xy,
            image_width=image_width,
            image_height=image_height,
            clip=clip,
        )
    embedding_geometry_to_coord_format(embedding_geometry)
    raise AssertionError("unreachable")


def polygon_to_bbox_2d(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    scale: int = 1000,
    clip: bool = True,
) -> list[int]:
    """Convert an arbitrary polygon to Qwen-style [x1, y1, x2, y2] coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if scale <= 0:
        raise ValueError("Coordinate scale must be positive.")

    bbox = polygon_to_normalized_bbox(
        polygon_xy,
        image_width=image_width,
        image_height=image_height,
        clip=clip,
    )
    bbox = bbox * float(scale)
    return [int(round(float(value))) for value in bbox.tolist()]
