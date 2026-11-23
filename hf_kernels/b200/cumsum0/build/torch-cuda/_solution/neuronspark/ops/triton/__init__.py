"""Triton kernels of the NeuronSpark operators (Triton 3.5.1).

This copy of ``neuronspark/ops/triton/__init__.py`` holds the definitions the exported operator uses,
and nothing else of the package file.
"""

from __future__ import annotations

import torch

__all__ = ["require_cuda"]


def require_cuda(t: torch.Tensor, what: str) -> None:
    """Raise unless ``t`` is a CUDA tensor."""
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise RuntimeError(f"neuronspark.{what}: CUDA tensors are required")
