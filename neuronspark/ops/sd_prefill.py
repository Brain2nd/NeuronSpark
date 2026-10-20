# Adapted from mamba_ssm (Apache-2.0): the composition of the chunked scan forward below follows
# mamba_ssm/ops/triton/ssd_combined.py, _mamba_chunk_scan_combined_fwd (mamba_ssm 2.3.2.post1, source lines 343-392,
# sha256 39c19c1e4c8e36982847079bc12b4f07ed3af03ac68f6a6653200ef80df5515d).
# Copyright (c) 2024, Tri Dao, Albert Gu.  Licensed under the Apache License, Version 2.0; the license text is
# neuronspark/ops/triton/mamba/LICENSE (see NOTICE).
# Modified by NeuronSpark: specialised to the SD layer's call (ngroups 1, D of shape (nheads,), no z, no seq_idx, no
# varlen states, dt_softplus on, dt_limit (0, inf)); the kernels run at fixed launch configurations; the chunk states
# and the state passing are one kernel (neuronspark.ops.triton.sd_states.csp_prefill); the chunk-scan forward is the
# Gluon kernel of neuronspark.ops.gluon.sd_chunk_scan_fwd for calls of whole chunks and the tensor-descriptor kernel of
# neuronspark.ops.triton.sd_scan otherwise; the einops rearranges are reshapes.
"""SD prefill scan: the selective state-space scan of an SD mixer over a prompt chunk, started from the carried state.

Op (registry name ``sd_chunk_scan_prefill``; inference prefill, no autograd)::

    y, final_state = sd_chunk_scan_prefill(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size=256)

* ``x``   (B, T, H, P) bf16 -- the conv output's x channels, a view whose (H, P) part is contiguous (row stride free);
* ``dt``  (B, T, H)    bf16 -- the raw step sizes, any strides (the strided slice of the in-projection);
* ``A``   (H,) fp32 = ``-exp(A_log)``; ``D`` (H,) fp32; ``dt_bias`` (H,) fp32;
* ``Bm`` / ``Cm`` (B, T, 1, N) bf16 -- the shared input / output projections, unit last stride;
* ``initial_state`` (B, H, P, N) fp32 or bf16 -- the carried compartments in the cache's storage dtype (the state-passing
  kernel widens a bf16 state to fp32 on load: the same values as an upcast copy);
* returns ``y`` (B, T, H, P) bf16 contiguous and ``final_state`` (B, H, P, N) fp32 (the caller stores it back in
  the cache's state dtype: one rounding per call for a bf16 state).

1MIX: H 144, P 64, N 128, chunk 256.  The math, per head h, soma p, compartment n and step t (``dt_t =
softplus(dt_raw + dt_bias)`` clamped to [0, inf), ``a_t = exp(dt_t A)``)::

    S_t = a_t S_{t-1} + dt_t x_t B_t^T          (S_{-1} = initial_state)
    y_t = S_t C_t + D x_t                       final_state = S_{T-1}

evaluated chunk-wise as the mamba_ssm chunked scan (in this order, each piece one kernel launch):

1. ``_chunk_cumsum_fwd``: dt (softplus, clamp) and the per-chunk inclusive cumsum of ``dt A`` (fp32);
2. ``csp_prefill``: each chunk's own end state ``sum_s exp(dA_end - dA_s) dt_s x_s B_s^T`` (fp32, kept in registers)
   and the states entering each chunk, ``s_c = exp(dA_chunk_end) s_{c-1} + chunk_state_c`` from ``initial_state`` in
   fp32, stored bf16 (C's dtype) for the scan, and the final state in fp32 -- mamba_ssm's ``_chunk_state_fwd`` and
   ``_state_passing_fwd`` in one kernel;
3. ``_bmm_chunk_fwd``: CB = C B^T per chunk (fp32);
4. the chunk-scan forward: ``y[m] = exp(dA_m) C_m s_c^T + sum_{k <= m} bf16(CB[m, k] exp(min(dA_m - dA_k, 0)) dt_k)
   x_k + D x_m``, rounded once to bf16 -- for T a multiple of the chunk size the persistent warp-specialised Gluon
   kernel (:func:`neuronspark.ops.gluon.sd_chunk_scan_fwd.chunk_scan_fwd_gluon`, whole chunks only), otherwise the
   tensor-descriptor port of mamba_ssm's chunk-scan forward (:func:`neuronspark.ops.triton.sd_scan.chunk_scan_fwd_tma`).

Kernels 1 and 3 are the vendored mamba_ssm kernels (:mod:`neuronspark.ops.triton.mamba`, bodies unchanged).  Launch
configurations: chunk cumsum ``{BLOCK_SIZE_H 1}`` 4 warps 3 stages, chunk states + passing (BN 64, BK 32, 4 warps, 4 stages), CB ``{M 64, N 32, K 32}`` 2 warps 5 stages, chunk scan
Gluon (the launch constants of :mod:`neuronspark.ops.gluon.sd_chunk_scan_fwd`: BM = BK 64, 2 heads per item, x ring
2, accumulator buffers 2, 4 warps per compute partition, work-split cost 3.0, registers 48 / 128, dA / dt ring 2, no
K-step prefetch, P ring 3, states ring 1, 4 columns per thread group; grid = #SM) or TMA (BM 64, BK 32, 4 warps,
3 stages); none is tuned at run time.

Numerics.  The rounding points are the stock chunked scan's: the chunk states summed in fp32 over K steps of 32
tokens (the stock chunk-state kernel's K block) and passed in fp32, the states entering each chunk rounded to bf16
before the inter-chunk product, ``CB * decay * dt`` rounded to bf16 before the intra-chunk product, fp32
accumulation, y rounded once.  The Gluon kernel accumulates the inter-chunk term separately and adds it in its
epilogue, and takes the decay as ``ex2.approx.ftz`` (results below 2^-126 flush to zero): fp32 order / underflow
differences of the per-operator rule class.  The association of a prompt split over several calls equals the single
call's only when every split point is a multiple of the chunk size (256); other split points give the same math with
another association (the rule class).  A partial last chunk is supported (by the TMA chunk-scan kernel).  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing`, each fused kernel against the stock mamba_ssm kernels
it replaces.

Contract (anything else raises): CUDA tensors on one device of a Blackwell GPU (compute capability >= 10.0: TMA
descriptors, tcgen05 / TMEM); the dtypes and shapes above with ngroups 1, N in {64, 128}, P in {16, 32, 64, 128}
(P <= 64 and H even when T is a multiple of 256: the Gluon kernel's two heads per item in tensor memory),
``chunk_size`` 256, T <= 64 chunks (16384 tokens) per call; ``x`` with a contiguous (H, P) part (or a non-unit last
stride, which is copied as mamba_ssm does); ``initial_state`` fp32 or bf16.  For whole-chunk calls an ``x`` / ``Cm`` whose
batch stride is not T times its row stride (a time slice of a longer buffer) is copied to the standard layout first
(a layout change only).  The Gluon kernel builds its work split on the first call per (B, nchunks, device): that
call must not be inside a CUDA-graph capture (it raises).

References (pure torch), one per stage group, each with the kernels' bf16 rounding points:
:func:`sd_prefill_chunk_quantities_ref` (stages 1 and 3), :func:`sd_prefill_states_ref` (stage 2),
:func:`sd_prefill_chunk_scan_ref` (stage 4); :func:`sd_chunk_scan_prefill_ref` (the registered reference) chains them.
:func:`sd_prefill_stage_tensors` returns the op's own stage results (the same launches), so that each stage reference
can be evaluated on the values the op's next stage reads (why: :func:`sd_prefill_chunk_scan_ref`).  The references
transcribe the element-wise expressions of the stock kernels the op reproduces: mamba_ssm 2.3.2.post1
``ssd_chunk_state.py`` ``_chunk_cumsum_fwd_kernel`` 39-86 and ``_chunk_state_fwd_kernel`` 191-268,
``ssd_state_passing.py`` ``_state_passing_fwd_kernel`` 29-87, ``ssd_bmm.py`` ``_bmm_chunk_fwd_kernel`` 36-93 (the
vendored copies in :mod:`neuronspark.ops.triton.mamba`) and ``ssd_chunk_scan.py`` ``_chunk_scan_fwd_kernel`` (the
expressions of :func:`neuronspark.ops.triton.sd_scan.chunk_scan_fwd_tma`).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from neuronspark.ops.gluon.sd_chunk_scan_fwd import chunk_scan_fwd_gluon
from neuronspark.ops.triton import require_cuda, require_sm100
from neuronspark.ops.triton.mamba import (
    BMM_CHUNK_FWD,
    CHUNK_CUMSUM_FWD_PREFILL,
    _bmm_chunk_fwd,
    _chunk_cumsum_fwd,
)
from neuronspark.ops.triton.sd_scan import chunk_scan_fwd_tma
from neuronspark.ops.triton.sd_states import CSP_MAX_CHUNKS, csp_prefill

__all__ = [
    "CHUNK_SIZE",
    "DT_LIMIT",
    "sd_chunk_scan_prefill",
    "sd_chunk_scan_prefill_ref",
    "sd_prefill_stage_tensors",
    "sd_prefill_chunk_quantities_ref",
    "sd_prefill_states_ref",
    "sd_prefill_chunk_scan_ref",
]

# ------------------------------------------------------------------------------------------------ fixed constants
CHUNK_SIZE = 256                       # the SD layer's chunk (sd_chunk_size); the launch configurations assume it
DT_SOFTPLUS = True                     # dt = softplus(dt_raw + dt_bias) inside the chunk cumsum
DT_LIMIT = (0.0, float("inf"))         # mamba_ssm's default clamp of dt (applied by the chunk cumsum kernel)
_HEADDIMS = (16, 32, 64, 128)          # chunk-scan forward: power-of-two head width
_DSTATES = (64, 128)                   # chunk states: N tiles of 64; chunk-scan forward: the state row in one box
_GLUON_MAX_HEADDIM = 64                # Gluon chunk scan: 2 heads x 2 buffers x 2 accumulators x P <= 512 TMEM columns
_GLUON_HEADS_PER_ITEM = 2              # Gluon chunk scan: heads per work item (H must be a multiple)


# ------------------------------------------------------------------------------------------------ private launcher
def _rows_packed(t: torch.Tensor) -> torch.Tensor:
    """``t`` (B, T, ...) if its batch stride is T times its row stride, else a copy in the standard layout (the Gluon
    chunk scan reads x and C as (B * T)-row tables); the same values."""
    if int(t.stride(0)) == int(t.shape[1]) * int(t.stride(1)):
        return t
    return torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device).copy_(t)


def _prefill_stages(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor, Cm: torch.Tensor,
                    D: torch.Tensor, dt_bias: torch.Tensor, initial_state: torch.Tensor,
                    chunk_size: int) -> Dict[str, torch.Tensor]:
    """mamba_ssm ``_mamba_chunk_scan_combined_fwd(x, dt, A, Bm, Cm, chunk_size, D=D, z=None, dt_bias=dt_bias,
    initial_states=initial_state, seq_idx=None, dt_softplus=True)`` with the chunk states and the state passing in one
    kernel and the chunk-scan forward routed by the call's length (operands checked by the caller).  Returns every
    stage's result: ``dt_out`` / ``dA_cumsum`` (B, H, nchunks, chunk) fp32 (stage 1), ``states`` (B, nchunks, H, P, N)
    in Cm's dtype (the states entering each chunk) and ``final_state`` (B, H, P, N) fp32 (stage 2), ``CB`` (B,
    nchunks, 1, chunk, chunk) fp32 (stage 3) and ``y`` (B, T, H, P) in x's dtype (stage 4)."""
    batch, seqlen, nheads, headdim = x.shape
    # layout rules: copies only for operands without a unit inner stride (a layout change only, the same values; the
    # chunk-state kernel reads x by descriptor rows, which need the unit last stride)
    if Bm.stride(-1) != 1:
        Bm = Bm.contiguous()
    if Cm.stride(-1) != 1:
        Cm = Cm.contiguous()
    if x.stride(-1) != 1:
        x = x.contiguous()
    if D.stride(-1) != 1:
        D = D.contiguous()
    dA_cumsum, dt_out = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=DT_SOFTPLUS,
                                          dt_limit=DT_LIMIT, config=CHUNK_CUMSUM_FWD_PREFILL)
    if not dt_out.is_contiguous():
        dt_out = dt_out.contiguous()
    if not dA_cumsum.is_contiguous():
        dA_cumsum = dA_cumsum.contiguous()
    states, final_state = csp_prefill(Bm, x, dt_out, dA_cumsum, initial_state, Cm.dtype)
    CB = _bmm_chunk_fwd(Cm, Bm, chunk_size, seq_idx=None, output_dtype=torch.float32, config=BMM_CHUNK_FWD)
    if seqlen % chunk_size == 0:
        out = chunk_scan_fwd_gluon(CB, _rows_packed(x), dt_out, dA_cumsum, _rows_packed(Cm), states, D)
    else:
        out = chunk_scan_fwd_tma(CB, x, dt_out, dA_cumsum, Cm, states, D)
    return {"dt_out": dt_out, "dA_cumsum": dA_cumsum, "states": states, "final_state": final_state, "CB": CB,
            "y": out}


def _sd_chunk_scan_prefill_fwd(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor,
                               Cm: torch.Tensor, D: torch.Tensor, dt_bias: torch.Tensor,
                               initial_state: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(out, final_states) of :func:`_prefill_stages`."""
    r = _prefill_stages(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size)
    return r["y"], r["final_state"]


@torch.library.custom_op(
    "neuronspark::sd_chunk_scan_prefill", mutates_args=(),
    schema="(Tensor x, Tensor dt, Tensor A, Tensor Bm, Tensor Cm, Tensor D, Tensor dt_bias, Tensor initial_state, "
           "int chunk_size) -> (Tensor, Tensor)")
def _sd_chunk_scan_prefill_op(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size):
    """(y (B, T, H, P) in x's dtype, contiguous; final_state (B, H, P, N) fp32 contiguous)"""
    return _sd_chunk_scan_prefill_fwd(x, dt, A, Bm, Cm, D, dt_bias, initial_state, int(chunk_size))


@_sd_chunk_scan_prefill_op.register_fake
def _sd_chunk_scan_prefill_fake(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size):
    batch, seqlen, nheads, headdim = x.shape
    dstate = Bm.shape[-1]
    return (torch.empty((batch, seqlen, nheads, headdim), dtype=x.dtype, device=x.device),
            torch.empty((batch, nheads, headdim, dstate), dtype=torch.float32, device=x.device))


# ------------------------------------------------------------------------------------------------ checks
def _fail(msg: str) -> None:
    raise RuntimeError(f"neuronspark.sd_chunk_scan_prefill: {msg}")


def _check_operands(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size) -> None:
    tensors = (("x", x), ("dt", dt), ("A", A), ("Bm", Bm), ("Cm", Cm), ("D", D), ("dt_bias", dt_bias),
               ("initial_state", initial_state))
    for name, t in tensors:
        if not isinstance(t, torch.Tensor):
            _fail(f"{name} must be a tensor, got {type(t).__name__}")
    require_cuda(x, "sd_chunk_scan_prefill")
    for name, t in tensors:
        if t.device != x.device:
            _fail(f"{name} is on {t.device}, x on {x.device} (one device expected)")
    require_sm100(x.device, "sd_chunk_scan_prefill")
    if isinstance(chunk_size, bool) or int(chunk_size) != CHUNK_SIZE:
        _fail(f"chunk_size must be {CHUNK_SIZE} (the launch configurations are fixed for it), got {chunk_size!r}")
    if x.dim() != 4 or dt.dim() != 3 or Bm.dim() != 4 or Cm.dim() != 4:
        _fail(f"expected x (B, T, H, P), dt (B, T, H), Bm / Cm (B, T, 1, N); got {tuple(x.shape)} {tuple(dt.shape)} "
              f"{tuple(Bm.shape)} {tuple(Cm.shape)}")
    batch, seqlen, nheads, headdim = (int(s) for s in x.shape)
    ngroups, dstate = int(Bm.shape[2]), int(Bm.shape[3])
    if batch < 1 or seqlen < 1:
        _fail(f"empty input {tuple(x.shape)}")
    if seqlen > CSP_MAX_CHUNKS * CHUNK_SIZE:
        _fail(f"T = {seqlen} exceeds {CSP_MAX_CHUNKS} chunks of {CHUNK_SIZE} per call (split the prompt)")
    if ngroups != 1 or tuple(Bm.shape) != (batch, seqlen, 1, dstate) or tuple(Cm.shape) != tuple(Bm.shape):
        _fail(f"Bm / Cm must be (B, T, 1, N) of x's (B, T) = ({batch}, {seqlen}); got {tuple(Bm.shape)} "
              f"{tuple(Cm.shape)}")
    if tuple(dt.shape) != (batch, seqlen, nheads):
        _fail(f"dt must be ({batch}, {seqlen}, {nheads}), got {tuple(dt.shape)}")
    if headdim not in _HEADDIMS or dstate not in _DSTATES:
        _fail(f"head width P {headdim} (one of {_HEADDIMS}) and state width N {dstate} (one of {_DSTATES}) expected")
    if seqlen % CHUNK_SIZE == 0 and (headdim > _GLUON_MAX_HEADDIM or nheads % _GLUON_HEADS_PER_ITEM):
        _fail(f"a whole-chunk call (T = {seqlen}) needs P <= {_GLUON_MAX_HEADDIM} and H a multiple of "
              f"{_GLUON_HEADS_PER_ITEM} (the Gluon chunk scan), got P {headdim} H {nheads}")
    for name, t in (("x", x), ("dt", dt), ("Bm", Bm), ("Cm", Cm)):
        if t.dtype != torch.bfloat16:
            _fail(f"{name} must be bf16, got {t.dtype}")
    for name, t in (("A", A), ("D", D), ("dt_bias", dt_bias)):
        if t.dtype != torch.float32 or tuple(t.shape) != (nheads,):
            _fail(f"{name} must be fp32 ({nheads},), got {t.dtype} {tuple(t.shape)}")
    if (initial_state.dtype not in (torch.float32, torch.bfloat16)
            or tuple(initial_state.shape) != (batch, nheads, headdim, dstate)):
        _fail(f"initial_state must be fp32 or bf16 ({batch}, {nheads}, {headdim}, {dstate}), got {initial_state.dtype} "
              f"{tuple(initial_state.shape)}")
    if x.stride(3) == 1 and x.stride(2) != headdim:
        # (a non-unit last stride is copied to the contiguous layout by the launcher's layout rules)
        _fail(f"x needs a contiguous (H, P) part (the scan's descriptors read (T, H * P) rows), strides "
              f"{tuple(x.stride())}")
    if torch.is_grad_enabled() and any(t.requires_grad for _, t in tensors):
        _fail("an inference operator (no autograd): call it under torch.no_grad()")


# ------------------------------------------------------------------------------------------------ public function
def sd_chunk_scan_prefill(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor, Cm: torch.Tensor,
                          D: torch.Tensor, dt_bias: torch.Tensor, initial_state: torch.Tensor, *,
                          chunk_size: int = CHUNK_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(y (B, T, H, P) bf16 contiguous, final_state (B, H, P, N) fp32)``: the SD prefill scan from
    ``initial_state`` (see the module docstring for shapes, layouts and the numerics contract).  No autograd; the
    inputs are not modified."""
    _check_operands(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size)
    y, final_state = torch.ops.neuronspark.sd_chunk_scan_prefill(x, dt, A, Bm, Cm, D, dt_bias, initial_state,
                                                                  int(chunk_size))
    return y, final_state


def sd_prefill_stage_tensors(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor, Cm: torch.Tensor,
                             D: torch.Tensor, dt_bias: torch.Tensor, initial_state: torch.Tensor, *,
                             chunk_size: int = CHUNK_SIZE) -> Dict[str, torch.Tensor]:
    """Every stage's result of the op on these operands -- the same checks and the same launch sequence as
    :func:`sd_chunk_scan_prefill` (:func:`_prefill_stages`): ``dt_out`` / ``dA_cumsum`` (B, H, nchunks, chunk) fp32,
    ``states`` (B, nchunks, H, P, N) in Cm's dtype (entering each chunk), ``final_state`` (B, H, P, N) fp32, ``CB``
    (B, nchunks, 1, chunk, chunk) fp32, ``y`` (B, T, H, P).  For the stage-wise reference comparison; not an operator
    of the model.  No autograd; the inputs are not modified."""
    _check_operands(x, dt, A, Bm, Cm, D, dt_bias, initial_state, chunk_size)
    return _prefill_stages(x, dt, A, Bm, Cm, D, dt_bias, initial_state, int(chunk_size))


# ------------------------------------------------------------------------------------------------ references
_LOG2E = 1.4426950408889634            # rounds to the fp32 constant tl.exp multiplies by (0x3FB8AA3B)


def _exp_k(v: torch.Tensor) -> torch.Tensor:
    """``tl.exp`` of an fp32 tensor in the form the kernels issue it: ``exp2(v * fp32(log2 e))`` (one fp32 multiply,
    then the base-2 exponential; see ``neuronspark.ops.triton._ptx._ex2ftz``)."""
    return torch.exp2(v * torch.tensor(_LOG2E, dtype=torch.float32, device=v.device))


def _to_chunks(t: torch.Tensor, nchunks: int, chunk: int) -> torch.Tensor:
    """(b, l, ...) -> (b, nchunks, chunk, ...) in t's dtype, zero padded past l (the kernels' masked loads)."""
    pad = nchunks * chunk - int(t.shape[1])
    if pad:
        t = F.pad(t, [0, 0] * (t.dim() - 2) + [0, pad])
    return t.reshape(int(t.shape[0]), nchunks, chunk, *t.shape[2:])


def sd_prefill_chunk_quantities_ref(dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor, Cm: torch.Tensor,
                                    dt_bias: torch.Tensor, *, chunk_size: int = CHUNK_SIZE
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure torch: stages 1 and 3 in the kernels' layouts, ``(dt_out, dA_cumsum, CB)`` fp32.

    * ``dt_out = clamp(softplus_k(dt + dt_bias), 0, inf)`` with ``softplus_k(v) = where(v <= 20, log(exp(v) + 1), v)``,
      0 past the sequence (vendored ``ssd_chunk_state.py`` ``_chunk_cumsum_fwd_kernel`` 75-85);
    * ``dA_cumsum`` = the inclusive cumsum over each chunk of the fp32 products ``dt_out * A`` (87-88), accumulated
      in float64 and rounded once (the association of the kernel's ``tl.cumsum`` is the rule class);
    * ``CB[c, l, s] = C_l . B_s`` (vendored ``ssd_bmm.py`` ``_bmm_chunk_fwd_kernel`` 75-90: bf16 products, fp32 result),
      accumulated in float64 and rounded once, 0 past the sequence.

    ``dt`` (B, T, H) any strides, ``A`` / ``dt_bias`` (H,), ``Bm`` / ``Cm`` (B, T, 1, N).  No kernel; any device."""
    batch, seqlen, nheads = (int(s) for s in dt.shape)
    chunk = int(chunk_size)
    nchunks = -(-seqlen // chunk)
    v = dt.float() + dt_bias.float()                                                      # (b, l, h)
    sp = torch.where(v <= 20.0, torch.log(_exp_k(torch.clamp_max(v, 20.0)) + 1.0), v)
    dtf = torch.clamp(sp, min=DT_LIMIT[0], max=DT_LIMIT[1])
    dt_out = _to_chunks(dtf, nchunks, chunk).permute(0, 3, 1, 2).contiguous()           # (b, h, c, l)
    dA = dt_out * A.float()[None, :, None, None]
    dA_cumsum = torch.cumsum(dA.double(), dim=-1).float()
    Bc = _to_chunks(Bm[:, :, 0], nchunks, chunk).double()                                 # (b, c, l, n)
    Cc = _to_chunks(Cm[:, :, 0], nchunks, chunk).double()
    CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc).float()[:, :, None].contiguous()        # (b, c, 1, l, s)
    return dt_out, dA_cumsum, CB


def _chunk_grid(fn: str, x: torch.Tensor, dt_out: torch.Tensor, dA_cumsum: torch.Tensor) -> Tuple[int, int]:
    """(nchunks, chunk) of stage-1 results that match x (B, T, H, P); raises otherwise."""
    batch, seqlen, nheads, _ = (int(s) for s in x.shape)
    nchunks, chunk = int(dt_out.shape[2]), int(dt_out.shape[3])
    if tuple(dt_out.shape) != (batch, nheads, nchunks, chunk) or tuple(dA_cumsum.shape) != tuple(dt_out.shape) \
            or not (nchunks - 1) * chunk < seqlen <= nchunks * chunk:
        raise ValueError(f"{fn}: dt_out {tuple(dt_out.shape)} / dA_cumsum {tuple(dA_cumsum.shape)} do not match x "
                         f"{tuple(x.shape)}")
    return nchunks, chunk


def sd_prefill_states_ref(x: torch.Tensor, Bm: torch.Tensor, initial_state: torch.Tensor, dt_out: torch.Tensor,
                          dA_cumsum: torch.Tensor, *, out_dtype: torch.dtype = torch.bfloat16
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure torch: stage 2 on given stage-1 results (``dt_out`` / ``dA_cumsum`` (B, H, nchunks, chunk) fp32), with
    the expressions of the stock pair it replaces; returns ``(states (B, nchunks, H, P, N) in out_dtype -- the state
    entering each chunk, final_state (B, H, P, N) fp32)``::

        chunk state (ssd_chunk_state.py _chunk_state_fwd_kernel 225-238):
            b_k = bf16(B_k * (exp(min(dA_last - dA_k, 0)) * dt_k)),  chunk_state = sum_k x_k b_k^T   (fp32)
        state passing (ssd_state_passing.py _state_passing_fwd_kernel 71-91):
            s_0 = initial_state,  s_{c+1} = exp(dA_last_c) s_c + chunk_state_c (fp32),
            states[:, c] = out_dtype(s_c),  final_state = s_nchunks

    ``dA_last`` is the chunk's last cumsum entry (index chunk - 1; constant past the sequence).  The element-wise fp32
    expressions before the bf16 rounding are the kernel's (same operands and order; ``exp`` as :func:`_exp_k`); the
    sums are accumulated in float64 and rounded once (the kernels' fp32 dot / fma order is the rule class).  The op
    stores the states in Cm's dtype (``out_dtype=Cm.dtype``).  The same math as
    :func:`neuronspark.ops.triton.sd_states.csp_prefill_ref` in the op's argument order.  No kernel; any device."""
    nchunks, chunk = _chunk_grid("sd_prefill_states_ref", x, dt_out, dA_cumsum)
    xc = _to_chunks(x, nchunks, chunk)                                                    # (b, c, l, h, p)
    Bc = _to_chunks(Bm[:, :, 0], nchunks, chunk).float()                                  # (b, c, l, n)
    dtq, daq = dt_out.float(), dA_cumsum.float()
    s = initial_state.float()                                                              # fp32 running state
    entering = []
    for c in range(nchunks):
        dt_c, da_c = dtq[:, :, c], daq[:, :, c]                                            # (b, h, l)
        scale = _exp_k(torch.clamp_max(da_c[..., -1:] - da_c, 0.0)) * dt_c                 # (b, h, l)
        b_k = (Bc[:, c][:, None] * scale[..., None]).to(x.dtype)                           # (b, h, l, n)
        chunk_state = torch.einsum("blhp,bhln->bhpn", xc[:, c].double(), b_k.double()).float()   # (b, h, p, n)
        entering.append(s.to(out_dtype))
        s = (_exp_k(da_c[..., -1]).double()[..., None, None] * s.double() + chunk_state.double()).float()
    return torch.stack(entering, dim=1), s


def sd_prefill_chunk_scan_ref(x: torch.Tensor, Cm: torch.Tensor, D: torch.Tensor, dt_out: torch.Tensor,
                              dA_cumsum: torch.Tensor, CB: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Pure torch: stage 4 on given stage 1 / 2 / 3 results (``dt_out`` / ``dA_cumsum`` (B, H, nchunks, chunk) fp32,
    ``states`` (B, nchunks, H, P, N) as stored for the scan, ``CB`` (B, nchunks, 1, chunk, chunk) fp32); returns y
    (B, T, H, P) in x's dtype, contiguous (the stock ``_chunk_scan_fwd_kernel`` expressions, as
    :func:`neuronspark.ops.triton.sd_scan.chunk_scan_fwd_tma` issues them; the reference of both chunk-scan
    kernels)::

        y_m = exp(dA_m) C_m states_c^T + sum_{k <= m} bf16((CB[m, k] exp(min(dA_m - dA_k, 0))) dt_k) x_k + D x_m,
        rounded once to x's dtype

    The element-wise fp32 expression before the bf16 rounding is the kernel's (same operands and order; ``exp`` as
    :func:`_exp_k`, the form ``tl.exp`` issues; the Gluon kernel's flush of decays below 2^-126 is not reproduced); the
    sums are accumulated in float64 and rounded once (the kernels' fp32 dot order, and the Gluon kernel's separate
    inter-chunk accumulator, are the rule class).

    Why the operator test evaluates each stage reference on the op's own inputs to that stage
    (:func:`sd_prefill_stage_tensors`): the values entering the bf16 rounding points of stages 2 and 4 are fp32
    reductions (the chunk cumsum's scan, the CB dot, the chunk-state dot and the passing) whose association a torch
    reduction does not reproduce.  A one-ulp fp32 difference there can move one bf16 rounding of ``CB[m, k] decay dt_k``
    by one bf16 ulp; that term multiplies a whole x row, so wherever the sum cancels (``|y| < ~|term| / 2``) the moved
    rounding exceeds both bounds of the per-operator rule.  With each stage's inputs taken from the op, the reference
    rounds the same fp32 values as the kernels, and every stage output is compared with its reference under the rule.
    No kernel; any device."""
    batch, seqlen, nheads, headdim = (int(s) for s in x.shape)
    nchunks, chunk = _chunk_grid("sd_prefill_chunk_scan_ref", x, dt_out, dA_cumsum)
    if tuple(CB.shape) != (batch, nchunks, 1, chunk, chunk) or tuple(states.shape[:4]) != (batch, nchunks, nheads,
                                                                                            headdim):
        raise ValueError(f"sd_prefill_chunk_scan_ref: CB {tuple(CB.shape)} / states {tuple(states.shape)} do not match "
                         f"x {tuple(x.shape)}")
    xc = _to_chunks(x, nchunks, chunk)                                                    # (b, c, l, h, p)
    Cc = _to_chunks(Cm[:, :, 0], nchunks, chunk)                                          # (b, c, l, n), Cm's dtype
    dtq, daq, cbq = dt_out.float(), dA_cumsum.float(), CB.float()
    Dd = D.double()[None, None, :, None]
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()           # [m, k]: k <= m
    ys = []
    for c in range(nchunks):
        dt_c, da_c = dtq[:, :, c], daq[:, :, c]                                            # (b, h, l)
        x64 = xc[:, c].double()                                                            # (b, l, h, p)
        decay = _exp_k(torch.clamp_max(da_c[..., :, None] - da_c[..., None, :], 0.0))      # (b, h, m, k)
        w = (cbq[:, c, 0][:, None] * decay) * dt_c[:, :, None, :]
        w = torch.where(causal, w, torch.zeros_like(w)).to(x.dtype)
        y_off = torch.einsum("bmn,bhpn->bmhp", Cc[:, c].double(), states[:, c].double()) \
            * _exp_k(da_c).double().permute(0, 2, 1)[..., None]
        y_in = torch.einsum("bhmk,bkhp->bmhp", w.double(), x64)
        ys.append((y_off + y_in + Dd * x64).float())
    return torch.cat(ys, dim=1)[:, :seqlen].to(x.dtype).contiguous()


def sd_chunk_scan_prefill_ref(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, Bm: torch.Tensor,
                              Cm: torch.Tensor, D: torch.Tensor, dt_bias: torch.Tensor, initial_state: torch.Tensor,
                              *, chunk_size: int = CHUNK_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure torch, the whole prefill scan (the registered reference, same parameters as the op): mamba_ssm's chunked
    composition (``_mamba_chunk_scan_combined_fwd``, ssd_combined.py 343-392) with its bf16 rounding points:
    :func:`sd_prefill_chunk_quantities_ref`, :func:`sd_prefill_states_ref` (states in Cm's dtype),
    :func:`sd_prefill_chunk_scan_ref`.  Chained on its own intermediates it differs from the op in the fp32
    association of the reductions before the rounding points (see :func:`sd_prefill_chunk_scan_ref`); the operator
    test compares the op stage by stage.  Returns ``(y in x's dtype, final_state fp32)``.  No kernel; any device."""
    dt_out, dA_cumsum, CB = sd_prefill_chunk_quantities_ref(dt, A, Bm, Cm, dt_bias, chunk_size=chunk_size)
    states, final_state = sd_prefill_states_ref(x, Bm, initial_state, dt_out, dA_cumsum, out_dtype=Cm.dtype)
    return sd_prefill_chunk_scan_ref(x, Cm, D, dt_out, dA_cumsum, CB, states), final_state
