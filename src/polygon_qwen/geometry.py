from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_points_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Expected polygon points with shape (N, 2), received {array.shape}."
        )
    if array.shape[0] < 3:
        raise ValueError("A polygon needs at least 3 points.")
    return array


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
