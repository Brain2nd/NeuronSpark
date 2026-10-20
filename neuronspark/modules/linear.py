"""Linear building blocks: a bias-free ``Linear`` whose forward is :func:`neuronspark.ops.linear.linear`, and the
weight holder of a causal depthwise short convolution.

Raw matrix parameters (``W_out``, ``eta_V`` / ``eta_U``, ``pool_xq_*``, ``pool_glu_*``, ``pool_vr_gate``) are plain
``nn.Parameter`` attributes of their modules, which call :func:`neuronspark.ops.linear.linear` directly; this file only
holds the module form.  State-dict keys: ``<name>.weight``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neuronspark.ops.linear import linear as _linear

__all__ = ["Linear", "ShortConvWeight", "LINEAR_INITS", "SHORT_CONV_INITS"]

LINEAR_INITS = ("default", "empty")
SHORT_CONV_INITS = ("identity", "empty")


class Linear(nn.Linear):
    """``nn.Linear`` without bias whose forward is the fp32-weight-gradient linear op (bf16 output).

    ``init``: ``"default"`` keeps PyTorch's ``nn.Linear`` initialisation (kaiming-uniform, a = sqrt(5)), drawn from the
    global RNG stream at construction; ``"empty"`` leaves the weight uninitialised (``torch.empty``, no random draw),
    for modules that draw their own launch values or for a build that is followed by a checkpoint load.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                 init: str = "default") -> None:
        if bias:
            raise ValueError("neuronspark Linear is bias-free (every projection of the model has no bias)")
        if init not in LINEAR_INITS:
            raise ValueError(f"Linear: init must be one of {LINEAR_INITS}, got {init!r}")
        # read by reset_parameters, which nn.Linear.__init__ calls
        object.__setattr__(self, "_ns_init", init)
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        if getattr(self, "_ns_init", "default") == "empty":
            return
        super().reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(x, self.weight)


class ShortConvWeight(nn.Module):
    """Holder of the depthwise causal short-convolution weight ``weight`` of shape ``(channels, 1, width)`` (fp32 by
    default); tap ``width - 1`` multiplies the current position (lag 0).  The convolution itself is an op
    (``short_conv_train`` / ``short_conv_silu_prefill`` / ``short_conv_step``).

    ``init``: ``"identity"`` (zeros, last tap 1: the conv starts as a pass-through), or ``"empty"`` (no init).
    """

    def __init__(self, channels: int, width: int, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None, init: str = "identity") -> None:
        super().__init__()
        if init not in SHORT_CONV_INITS:
            raise ValueError(f"ShortConvWeight: init must be one of {SHORT_CONV_INITS}, got {init!r}")
        if channels <= 0 or width <= 0:
            raise ValueError(f"ShortConvWeight: channels and width must be positive, got {channels}, {width}")
        self.channels = int(channels)
        self.width = int(width)
        self.weight = nn.Parameter(torch.empty((self.channels, 1, self.width), device=device,
                                               dtype=torch.float32 if dtype is None else dtype))
        if init == "identity":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.weight.zero_()
        self.weight[:, :, -1] = 1.0

    def extra_repr(self) -> str:
        return f"channels={self.channels}, width={self.width}"
