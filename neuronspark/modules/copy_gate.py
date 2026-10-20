"""Copy arbitration of the hippocampal layer: the novelty predictor, the confidence-gated read gain and the PKM copy
gate.

The copy detector's statistics are model-level (:mod:`neuronspark.modules.copy_detector`: the maximum and the
logsumexp of the 32-scaled cosines of the order-1 lexical rows over the strict past s <= t - 2, one computation per
call shared by the six hippocampal layers).  Each layer turns them into its confidence with its own calibrated sink
``pkm_copy_tau`` (a PKM buffer):

    pi  = exp(32 max - logaddexp(32 tau, lse))                                 (T, B) fp32, no gradient

and uses it at two sites:

* read gain (``hippocampal_read_copy_gain`` kappa = 4.0) with the novelty predictor c of this layer:
      c   = sigmoid( z . w + b + rms(r) . v )                                  (T, B) fp32, with gradient
      f   = (kappa pi) c                                                       fp32
      h'  = h + bf16(f32(r) (1 + f))                                           r = the centred attention residual
  ``z`` is the layer's frozen-norm input (fp32), ``rms(r) = f32(r) rsqrt(mean(f32(r)^2) + 1e-6)``; the predictor's
  matvecs and the gain run with autocast disabled (fp32; TF32 under the training regime, the matvecs' backward too:
  :func:`neuronspark.utils.precision.regime_product`);
* PKM copy gate with the fixed strength ``pkm_copy_strength`` (4.0, a PKM buffer); the novelty factor c does not
  enter it (c scales the read gain only):
      rho = 2 sigmoid(-strength pi)                                            (T, B) fp32, no gradient

Parameters (state-dict keys, registered flat on the layer): ``copy_novelty_w`` (D,) = 0, ``copy_novelty_b`` (1,) =
``hippocampal_copy_novelty_gate_init`` (4.0), ``copy_novelty_v`` (D,) = 0.

The supervised copy switch (the model-level auxiliary loss) scores the predictor's logit on the detached inputs; the
whole-row forward hands it ``(z.detach(), r.detach(), pi)`` per layer (``aux_out``), and
:meth:`CopyGateMixin.copy_novelty_logit_detached` is that logit's expression.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from neuronspark.modules.copy_detector import CopyStats, copy_pi
from neuronspark.utils.precision import regime_product

__all__ = ["CopyGateMixin", "NOVELTY_RMS_EPS"]

# eps of the predictor's rms(r)
NOVELTY_RMS_EPS = 1e-6


class CopyGateMixin:
    """Mixin of :class:`neuronspark.modules.hippocampal_layer.HippocampalLayer`; reads ``pkm_copy_tau`` /
    ``pkm_copy_strength`` registered by :class:`neuronspark.modules.pkm.PKMMixin`."""

    def _build_copy_gate(self, config, device, dtype, init: str) -> None:
        D = int(config.D)
        f32 = torch.float32 if dtype is None else dtype
        self.read_copy_gain = float(config.hippocampal_read_copy_gain)
        if init == "launch":
            w = torch.zeros(D, device=device, dtype=f32)
            b = torch.full((1,), float(config.hippocampal_copy_novelty_gate_init), device=device, dtype=f32)
            v = torch.zeros(D, device=device, dtype=f32)
        else:
            w = torch.empty(D, device=device, dtype=f32)
            b = torch.empty((1,), device=device, dtype=f32)
            v = torch.empty(D, device=device, dtype=f32)
        self.copy_novelty_w = nn.Parameter(w)
        self.copy_novelty_b = nn.Parameter(b)
        self.copy_novelty_v = nn.Parameter(v)

    # ------------------------------------------------------------------------------------------------ the detector
    def copy_confidence(self, copy_stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """(T, B) fp32 pi of this layer from the model-level statistics ``(maximum, lse)`` (each (T, B) fp32) and the
        layer's sink ``pkm_copy_tau``; no gradient."""
        if not (isinstance(copy_stats, (tuple, list)) and len(copy_stats) == 2):
            raise RuntimeError("hippocampal layer: the copy-detector statistics must be the pair (maximum, lse)")
        stats = copy_stats if isinstance(copy_stats, CopyStats) else CopyStats(*copy_stats)
        with torch.no_grad(), torch.autocast(stats.maximum.device.type, enabled=False):
            return copy_pi(stats, self.pkm_copy_tau)

    # ------------------------------------------------------------------------------------------------ the predictor
    def copy_novelty(self, z: torch.Tensor, r: torch.Tensor,
                     rms: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """(T, B) fp32 novelty factor c = sigmoid(z . w + b + rms(r) . v) with gradient to w, b, v, z and r.  ``rms`` is
        the regime's fp32 rms normalisation of the bf16 residual (training: ``ops.nov_norm``; inference:
        ``ops.rmsnorm_rows``).  With autograd the two matvecs' backward runs under the TF32 setting of this call."""
        if tuple(r.shape) != tuple(z.shape):
            raise RuntimeError(f"novelty predictor: residual {tuple(r.shape)} does not match z {tuple(z.shape)}")
        with torch.autocast(z.device.type, enabled=False):
            logit = regime_product(torch.matmul, z.float(), self.copy_novelty_w.float()) + self.copy_novelty_b.float()
            rf = rms(r)
            logit = logit + regime_product(torch.matmul, rf, self.copy_novelty_v.float())
            return torch.sigmoid(logit)

    def copy_novelty_logit_detached(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """(T, B) fp32 predictor logit from the detached inputs (the supervised copy switch's operand): z . w + b +
        rms(r) . v with rms in torch; gradient to w, b, v only."""
        with torch.autocast(z.device.type, enabled=False):
            logit = z.detach().float() @ self.copy_novelty_w.float() + self.copy_novelty_b.float()
            rf = r.detach().float()
            rf = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + NOVELTY_RMS_EPS)
            return logit + rf @ self.copy_novelty_v.float()

    # ------------------------------------------------------------------------------------------------ the two sites
    def read_gain_factor(self, pi: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        """(T, B) fp32 f = (kappa pi) c of the read gain h' = h + bf16(f32(r) (1 + f)) (autocast disabled)."""
        with torch.autocast(pi.device.type, enabled=False):
            f = self.read_copy_gain * pi
            if c is not None:
                f = f * c
            return f

    def pkm_copy_rho(self, pi: torch.Tensor) -> torch.Tensor:
        """(T, B) fp32 rho = 2 sigmoid(-strength pi) of the PKM copy gate (no gradient: pi and the strength are
        constants of the call)."""
        with torch.autocast(pi.device.type, enabled=False):
            strength = self.pkm_copy_strength.float()
            return 2.0 * torch.sigmoid(-strength * pi)
