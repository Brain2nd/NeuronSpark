"""Triton kernels of the NeuronSpark operators (Triton 3.5.1).

This copy of ``neuronspark/ops/triton/__init__.py`` holds the definitions the exported operator uses,
and nothing else of the package file.
"""

from __future__ import annotations

import torch

__all__ = ["require_cuda", "require_no_grad"]


def require_cuda(t: torch.Tensor, what: str) -> None:
    """Raise unless ``t`` is a CUDA tensor."""
    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise RuntimeError(f"neuronspark.{what}: CUDA tensors are required")


def require_no_grad(what: str, *tensors) -> None:
    """Raise if grad mode is on and an operand (``None`` skipped) requires grad: the inference operator ``what`` has no
    backward, so its result would silently carry no gradient."""
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise RuntimeError(f"neuronspark.{what}: an inference operator (no autograd): call it under torch.no_grad()")
