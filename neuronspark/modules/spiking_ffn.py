"""Dendritic FFN of the hippocampal layer (state-dict prefix ``snn_ffn.``).

    g   = gate_neuron(bf16(z @ gate_proj^T))        (T, B, D_ff) bf16 spikes (PLIF, scalar launch tau)
    u   = up_neuron(bf16(z @ up_proj^T))            (T, B, D_ff) bf16 spikes
    out = bf16( bf16(bf16(g * u) @ down_proj^T) + bf16(z @ skip_proj^T) )     (T, B, D) bf16

``z`` is the spike output of the layer's ``input_neuron2`` (fp32).  Every projection is the fp32-weight-gradient linear
op (bf16 operands, fp32 accumulation, bf16 output).  The two neurons carry their own membranes in an inference call
(``gate_state`` / ``up_state``, the ``ffn_gate`` / ``ffn_up`` entries of the layer's ``HippocampalState``).

Launch (``init="launch"``, global RNG stream, in this order): the four projections' PyTorch default draws, the gate
neuron's and the up neuron's draws, then gate_proj / up_proj Xavier-uniform, down_proj N(0, out_std), skip_proj 0.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from neuronspark.modules.linear import Linear
from neuronspark.modules.plif import PLIFNeuron, check_init, linear_init
from neuronspark.ops import namespace
from neuronspark.utils.cache import PLIFState

__all__ = ["SpikingFFN"]


class SpikingFFN(nn.Module):
    """``SpikingFFN(D, D_ff, *, init_tau, alpha, eta_rank, out_std, device, dtype, impl, init)``; keys
    ``{gate,up,down,skip}_proj.weight`` and ``{gate,up}_neuron.*``."""

    def __init__(self, D: int, D_ff: int, *, init_tau: float = 2.0, alpha: float = 8.0, eta_rank: int = 16,
                 out_std: float = 0.02, device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None, impl: str = "kernel", init: str = "launch") -> None:
        super().__init__()
        check_init(init)
        self.D, self.D_ff = int(D), int(D_ff)
        self.impl = impl
        self._ops = namespace(impl)
        li = linear_init(init)
        self.gate_proj = Linear(self.D, self.D_ff, device=device, dtype=dtype, init=li)
        self.up_proj = Linear(self.D, self.D_ff, device=device, dtype=dtype, init=li)
        self.down_proj = Linear(self.D_ff, self.D, device=device, dtype=dtype, init=li)
        self.skip_proj = Linear(self.D, self.D, device=device, dtype=dtype, init=li)
        kw = dict(centred=False, alpha=alpha, eta_rank=eta_rank, device=device, dtype=dtype, impl=impl, init=init)
        self.gate_neuron = PLIFNeuron(self.D_ff, init_tau, **kw)
        self.up_neuron = PLIFNeuron(self.D_ff, init_tau, **kw)
        if init == "launch":
            with torch.no_grad():
                nn.init.xavier_uniform_(self.gate_proj.weight)
                nn.init.xavier_uniform_(self.up_proj.weight)
                self.down_proj.weight.normal_(std=float(out_std))
                self.skip_proj.weight.zero_()

    def forward(self, z: torch.Tensor, gate_state: Optional[PLIFState] = None, up_state: Optional[PLIFState] = None,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z (T, B, D) -> (T, B, D) bf16.  Both states None: whole-row form; both given: inference form."""
        if (gate_state is None) != (up_state is None):
            raise RuntimeError("SpikingFFN: pass both neuron states (inference) or neither (whole row)")
        lin = self._ops.linear
        g = self.gate_neuron(lin(z, self.gate_proj.weight), gate_state, pos)
        u = self.up_neuron(lin(z, self.up_proj.weight), up_state, pos)
        return lin(g * u, self.down_proj.weight) + lin(z, self.skip_proj.weight)
