from __future__ import annotations

from typing import Any


def _size_get(size: Any, key: str) -> Any:
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def _size_set(size: Any, key: str, value: Any) -> None:
    if isinstance(size, dict):
        size[key] = value
    else:
        setattr(size, key, value)


def configure_processor(processor: Any, *, max_pixels: int | None) -> None:
    """Apply tokenizer padding and vision-size settings shared by train/eval scripts."""

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or max_pixels is None:
        return

    size = getattr(image_processor, "size", None)
    if size is not None and _size_get(size, "longest_edge") is not None:
        _size_set(size, "longest_edge", max_pixels)
        shortest_edge = _size_get(size, "shortest_edge")
        if shortest_edge is not None and shortest_edge > max_pixels:
            _size_set(size, "shortest_edge", max_pixels)
        return

    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
