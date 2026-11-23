"""Triton kernels of the NeuronSpark operators (Triton 3.5.1).

This copy of ``neuronspark/ops/triton/__init__.py`` holds the definitions the exported operator uses,
and nothing else of the package file.
"""

from __future__ import annotations

import functools
from typing import Union
import torch

__all__ = ["num_sms", "require_cuda"]


_Device = Union[torch.device, str, int, torch.Tensor]


def _as_device(dev: _Device) -> torch.device:
    if isinstance(dev, torch.Tensor):
        dev = dev.device
    d = torch.device(dev) if not isinstance(dev, int) else torch.device("cuda", dev)
    if d.type != "cuda":
        return d
    return torch.device("cuda", torch.cuda.current_device() if d.index is None else d.index)


@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def num_sms(dev: _Device) -> int:
    """Streaming-multiprocessor count of a CUDA device (cached per device)."""
    d = _as_device(dev)
    if d.type != "cuda":
        raise RuntimeError(f"neuronspark: a CUDA device is required, got {d}")
    return _sm_count(d.index)


def require_cuda(t: torch.Tensor, what: str) -> None:
    """Raise unless ``t`` is a CUDA tensor."""
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise RuntimeError(f"neuronspark.{what}: CUDA tensors are required")
