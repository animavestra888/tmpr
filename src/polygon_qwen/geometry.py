from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - import error depends on environment
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for minimum-area-rectangle preprocessing."
        )


def _as_points_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Expected polygon points with shape (N, 2), received {array.shape}."
        )
    if array.shape[0] < 3:
        raise ValueError("A polygon needs at least 3 points.")
    return array


def order_box_corners(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return 4 box corners in a stable order.

    The points are sorted by angle around the centroid and then rotated so the
    first point is the top-left-like corner according to the smallest (y + x).
    """

    box = np.asarray(points, dtype=np.float32)
    if box.shape != (4, 2):
        raise ValueError(f"Expected box corners with shape (4, 2), received {box.shape}.")

    center = box.mean(axis=0)
    angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = box[order]

    start_index = int(np.argmin(ordered[:, 1] + ordered[:, 0]))
    ordered = np.roll(ordered, -start_index, axis=0)
    return ordered.astype(np.float32)


def polygon_to_minrect_8coords(
    polygon_xy: Sequence[Sequence[float]] | np.ndarray,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Convert an arbitrary polygon to 8 normalized min-area-rectangle coordinates."""

    _require_cv2()
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    polygon = _as_points_array(polygon_xy)
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_box_corners(box)

    coords = box.reshape(8)
    coords[0::2] /= float(image_width)
    coords[1::2] /= float(image_height)
    if clip:
        coords = np.clip(coords, 0.0, 1.0)
    return coords.astype(np.float32)


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

    polygon = _as_points_array(polygon_xy)
    min_xy = polygon.min(axis=0)
    max_xy = polygon.max(axis=0)
    bbox = np.asarray(
        [
            min_xy[0] / float(image_width) * scale,
            min_xy[1] / float(image_height) * scale,
            max_xy[0] / float(image_width) * scale,
            max_xy[1] / float(image_height) * scale,
        ],
        dtype=np.float32,
    )
    if clip:
        bbox = np.clip(bbox, 0.0, float(scale))
    return [int(round(float(value))) for value in bbox.tolist()]
