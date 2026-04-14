from __future__ import annotations

import os

import torch


SUPPORTED_DEVICE_NAMES = ("auto", "cuda", "npu", "cpu")


def _import_torch_npu(*, required: bool) -> bool:
    try:
        import torch_npu
    except ImportError as exc:
        if required:
            raise ImportError(
                "NPU device was requested, but torch_npu is not installed"
            ) from exc
        return False
    return True


def is_npu_available() -> bool:
    _import_torch_npu(required=False)
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and npu.is_available())


def resolve_device(requested_device: str = "auto") -> torch.device:
    if requested_device not in SUPPORTED_DEVICE_NAMES:
        raise ValueError(
            f"Unsupported device '{requested_device}'. Expected one of {SUPPORTED_DEVICE_NAMES}."
        )

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_npu_available():
            return torch.device("npu")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")

    if requested_device == "npu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        _import_torch_npu(required=True)
        if not is_npu_available():
            raise RuntimeError("NPU device was requested, but torch.npu.is_available() is false.")
        return torch.device("npu")

    return torch.device("cpu")


def resolve_auto_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "npu":
        return torch.bfloat16
    return torch.float32
