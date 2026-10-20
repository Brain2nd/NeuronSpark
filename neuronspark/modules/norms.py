"""RMS normalisation building blocks (pure torch: the module-level form and the reference of the norm ops).

``RMSNorm(dim, eps, zero_centered)``: ``y = (w_eff * (xf * rsqrt(mean(xf^2, -1) + eps))).to(x.dtype)`` with
``xf = x.float()`` and ``w_eff = weight`` or, zero-centred, ``1.0 + weight`` (the stored parameter is the deviation of
the gain from 1, initialised at 0).  ``w_eff`` is formed in the weight's dtype (fp32) and the product is fp32.  The
output dtype equals the input dtype.  State-dict key: ``weight``.

``frozen_rms_norm(x, eps)``: the same chain with the gain fixed at 1.0 and no parameter (the attention-input norm of
the memory layer): ``(xf * rsqrt(mean(xf^2, -1) + eps)).to(x.dtype)``.

The op order is fixed (``x.pow(2).mean(-1, keepdim=True)``, ``torch.rsqrt(v + eps)``, the gain product last): it is
the eager arithmetic that the fused norm kernels reproduce.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "frozen_rms_norm", "rms_norm_ref"]


def rms_norm_ref(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float,
                 zero_centered: bool = False) -> torch.Tensor:
    """The RMSNorm chain as a function (``weight`` None: gain 1, no product)."""
    input_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    if weight is None:
        return xf.to(input_dtype)
    w = (1.0 + weight) if zero_centered else weight
    return (w * xf).to(input_dtype)


def frozen_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalisation with the gain frozen at 1.0 (no parameter); output dtype = input dtype."""
    return rms_norm_ref(x, None, eps)


class RMSNorm(nn.Module):
    """``RMSNorm(dim, eps=1e-6, zero_centered=False)``; the weight is fp32 by default.

    ``init``: ``"default"`` fills ones (plain) or zeros (zero-centred, i.e. gain 1); ``"empty"`` leaves the weight
    uninitialised (for a build followed by a checkpoint load).
    """

    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = False,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                 init: str = "default") -> None:
        super().__init__()
        if init not in ("default", "empty"):
            raise ValueError(f"RMSNorm: init must be 'default' or 'empty', got {init!r}")
        self.dim = int(dim)
        self.eps = float(eps)
        self.zero_centered = bool(zero_centered)
        self.weight = nn.Parameter(torch.empty((self.dim,), device=device,
                                               dtype=torch.float32 if dtype is None else dtype))
        if init == "default":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.zero_centered:
            self.weight.zero_()
        else:
            self.weight.fill_(1.0)

    def effective_weight(self) -> torch.Tensor:
        """The gain the norm multiplies by: ``weight`` or ``1.0 + weight`` (zero-centred), in the weight's dtype."""
        return (1.0 + self.weight) if self.zero_centered else self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_ref(x, self.weight, self.eps, self.zero_centered)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, zero_centered={self.zero_centered}"
