"""The gated RMSNorm of a dendritic neuron layer split by heads across the two GPUs of a tensor-parallel step.

Each GPU holds the channels of its heads (``N`` of the layer's ``N_all`` = d_inner channels); the norm runs over all
of them, so the per-row sum of squares of the gated values is exchanged between the GPUs inside the kernel (the
mailbox of :mod:`neuronspark.ops.triton.tp_exchange`) and the divisor is ``N_all``.  Everything else is the one-GPU
kernels' arithmetic on the local channels.

Ops (inference, no autograd; ``send`` / ``recv`` the peer's and this GPU's mailbox region):

* ``tp_step_core(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps, n_all, send, recv)
  -> xs``: the decode step of the local heads: the one-GPU K1 routes of
  :mod:`neuronspark.ops.triton.dendritic_step` (conv update, membrane update, the gate) on the local projection
  ``[z | x | B | C | dt]`` (B and C are the layer's, computed on both GPUs), then the tensor-parallel K2: the row's
  local sum of squares, the exchange (one word per (row, program): B x programs words), ``rstd = 1 / sqrt((s + s_peer)
  / n_all + eps)``, n = bf16(g rstd), the soma read-out of the program's slice; the row's B / C conv tails rolled by
  slice 0, as in the one-GPU K2.
* ``tp_gated_norm_soma(y, z, theta, gamma, eps, n_all, send, recv) -> x``: the prefill's gated norm and soma on (M, N)
  bf16 rows (one program per row, the whole local row in registers, one word per row).

References: :func:`tp_step_core_ref`, :func:`tp_gated_norm_soma_ref` take the peer's per-row sums of squares.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _sr_bf16_to_f32, _sr_f32_to_bf16, _sr_mul_rn, _sr_release
from neuronspark.ops.triton.dendritic_gated_norm import NUM_WARPS as GN_WARPS, THREADS as GN_THREADS, VEC as GN_VEC
from neuronspark.ops.triton.dendritic_gated_norm import _rep_gate, _rep_load, _soma_store
from neuronspark.ops.triton.dendritic_soma import dendritic_soma_readout_ref
from neuronspark.ops.triton.dendritic_step import (_launch_k1_gated, _launch_k1_plain, _launch_k1_split, _widths,
                                                  dendritic_step_k1_ref, k1_route, k1_tile, k2_programs)
from neuronspark.ops.triton.tp_exchange import mailbox_swap1

__all__ = ["tp_step_core", "tp_step_core_ref", "tp_gated_norm_soma", "tp_gated_norm_soma_ref",
           "k2_words"]

def k2_words(b: int, di: int) -> int:
    """Mailbox words of one tensor-parallel K2 call: one per (row, program)."""
    slice_ = triton.next_power_of_2(triton.cdiv(di, k2_programs(b)))
    return int(b) * triton.cdiv(di, slice_)


# ================================================================================================ kernels
@triton.jit
def _tp_k2_kernel(Y, PROJ, CS, TH, GA, XO, SCR, SEND, RECV,
                  s_y_b, s_proj_b, s_cs_b, s_cs_c, s_cs_l, s_xo_b, s_scr_b,
                  N, n_all, eps,
                  DI: tl.constexpr, NS: tl.constexpr, BLOCK_N: tl.constexpr, SLICE: tl.constexpr,
                  NPART: tl.constexpr, GATED: tl.constexpr, LANES: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if GATED:                                # the fp32 gate of the gated / split K1
        x = tl.load(Y + row * s_y_b + cols, mask=mask, other=0.).to(tl.float32)
    else:                                    # the bf16 y of the plain K1, gated here
        x = tl.load(Y + row * s_y_b + cols, mask=mask, other=0.).to(tl.float32)
        z = tl.load(PROJ + row * s_proj_b + cols, mask=mask, other=0.).to(tl.float32)
        x *= z * tl.sigmoid(z)
    xbar = tl.where(mask, x, 0.)
    ss = tl.sum(xbar * xbar, axis=0)                          # this GPU's channels of the row
    w = row * NPART + part
    peer = mailbox_swap1(ss, SEND + w, RECV + w, LANES)
    var = (ss + peer) / n_all
    rstd = 1 / tl.sqrt(var + eps)
    y = x * rstd
    smask = mask & (cols >= part * SLICE) & (cols < (part + 1) * SLICE)
    tl.store(SCR + row * s_scr_b + cols, y.to(tl.bfloat16), mask=smask)
    tl.debug_barrier()
    sc = part * SLICE + tl.arange(0, SLICE)
    sm = sc < N
    th = tl.load(TH + sc, mask=sm, other=0.0)
    ga = tl.load(GA + sc, mask=sm, other=0.0)
    v = _sr_bf16_to_f32(tl.load(SCR + row * s_scr_b + sc, mask=sm, other=0.0))
    s, _a, _b = _sr_release(v, th)
    tl.store(XO + row * s_xo_b + sc, _sr_f32_to_bf16(_sr_mul_rn(s, ga)), mask=sm)
    if part == 0:                            # the row's B / C conv tails (read by every K1 program of the row)
        cbc = DI + tl.arange(0, 2 * NS)
        p = CS + row * s_cs_b + cbc * s_cs_c
        t1 = tl.load(p + s_cs_l)
        t2 = tl.load(p + 2 * s_cs_l)
        xn = tl.load(PROJ + row * s_proj_b + DI + cbc).to(tl.float32)
        tl.debug_barrier()
        tl.store(p, t1)
        tl.store(p + s_cs_l, t2)
        tl.store(p + 2 * s_cs_l, xn)


@triton.jit
def _tp_gated_norm_soma_kernel(X, Z, Y, NOUT, TH, GA, RSTD, SEND, RECV, sx, sz, sy, N, n_all, eps,
                               VEC: tl.constexpr, T: tl.constexpr, NREP_EFF: tl.constexpr, KEEP: tl.constexpr,
                               STORE_N: tl.constexpr):
    """The one-GPU prefill kernel's two passes (its register-resident gated row and its summation order, the helpers of
    :mod:`neuronspark.ops.triton.dendritic_gated_norm`) with the exchange of the row's sum of squares between them.
    KEEP stores the row's rstd, STORE_N the bf16 n (in the layout of the output)."""
    row = tl.program_id(0).to(tl.int64)
    X += row * sx
    Z += row * sz
    Y += row * sy
    NOUT += row * sy
    tl.static_assert(VEC == 8, "the chain peels 8 slots")
    tl.static_assert(NREP_EFF <= 8, "the chain unrolls up to 8 reps")
    acc = tl.zeros([T], dtype=tl.float32)
    if NREP_EFF > 0:
        x0, z0 = _rep_load(X, Z, 0, N, T, VEC)
        g0, acc = _rep_gate(x0, z0, acc, 0, N, T, VEC)
    if NREP_EFF > 1:
        x1, z1 = _rep_load(X, Z, 1, N, T, VEC)
        g1, acc = _rep_gate(x1, z1, acc, 1, N, T, VEC)
    if NREP_EFF > 2:
        x2, z2 = _rep_load(X, Z, 2, N, T, VEC)
        g2, acc = _rep_gate(x2, z2, acc, 2, N, T, VEC)
    if NREP_EFF > 3:
        x3, z3 = _rep_load(X, Z, 3, N, T, VEC)
        g3, acc = _rep_gate(x3, z3, acc, 3, N, T, VEC)
    if NREP_EFF > 4:
        x4, z4 = _rep_load(X, Z, 4, N, T, VEC)
        g4, acc = _rep_gate(x4, z4, acc, 4, N, T, VEC)
    if NREP_EFF > 5:
        x5, z5 = _rep_load(X, Z, 5, N, T, VEC)
        g5, acc = _rep_gate(x5, z5, acc, 5, N, T, VEC)
    if NREP_EFF > 6:
        x6, z6 = _rep_load(X, Z, 6, N, T, VEC)
        g6, acc = _rep_gate(x6, z6, acc, 6, N, T, VEC)
    if NREP_EFF > 7:
        x7, z7 = _rep_load(X, Z, 7, N, T, VEC)
        g7, acc = _rep_gate(x7, z7, acc, 7, N, T, VEC)
    ss = tl.sum(acc, axis=0)                                  # the one-GPU kernel's sum over this GPU's channels
    peer = mailbox_swap1(ss, SEND + row, RECV + row, T)       # T: the program's threads
    var = (ss + peer) / n_all
    rstd = 1 / tl.sqrt(var + eps)
    if KEEP:
        tl.store(RSTD + row, rstd)
    if NREP_EFF > 0:
        _soma_store(Y, NOUT, TH, GA, g0 * rstd, 0, N, T, VEC, STORE_N)
    if NREP_EFF > 1:
        _soma_store(Y, NOUT, TH, GA, g1 * rstd, 1, N, T, VEC, STORE_N)
    if NREP_EFF > 2:
        _soma_store(Y, NOUT, TH, GA, g2 * rstd, 2, N, T, VEC, STORE_N)
    if NREP_EFF > 3:
        _soma_store(Y, NOUT, TH, GA, g3 * rstd, 3, N, T, VEC, STORE_N)
    if NREP_EFF > 4:
        _soma_store(Y, NOUT, TH, GA, g4 * rstd, 4, N, T, VEC, STORE_N)
    if NREP_EFF > 5:
        _soma_store(Y, NOUT, TH, GA, g5 * rstd, 5, N, T, VEC, STORE_N)
    if NREP_EFF > 6:
        _soma_store(Y, NOUT, TH, GA, g6 * rstd, 6, N, T, VEC, STORE_N)
    if NREP_EFF > 7:
        _soma_store(Y, NOUT, TH, GA, g7 * rstd, 7, N, T, VEC, STORE_N)


# ================================================================================================ launchers
def _region(send, recv, words, device, what):
    for name, t in (("send", send), ("recv", recv)):
        if (t.dtype != torch.int64 or t.dim() != 1 or int(t.numel()) < words or t.device != device
                or not t.is_contiguous()):
            raise RuntimeError(f"neuronspark.{what}: {name} must be a contiguous int64 region of at least {words} "
                               f"words on {device}")


def tp_step_core(proj: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor, conv_w: torch.Tensor,
                 conv_b: torch.Tensor, A: torch.Tensor, dt_bias: torch.Tensor, D: torch.Tensor, theta: torch.Tensor,
                 gamma: torch.Tensor, eps: float, n_all: int, send: torch.Tensor, recv: torch.Tensor) -> torch.Tensor:
    """xs (b, di_local) bf16; the local conv / SSM states advanced in place (see the module docstring)."""
    require_cuda(proj, "tp_step_core")
    b = int(proj.shape[0])
    di, conv_dim, H, N, P = _widths(proj, conv_w, A, theta)
    if int(proj.shape[1]) != 2 * di + 2 * N + H or proj.dtype != torch.bfloat16 or proj.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_step_core: proj must be bf16 (b, {2 * di + 2 * N + H}), got "
                           f"{proj.dtype} {tuple(proj.shape)}")
    _region(send, recv, k2_words(b, di), proj.device, "tp_step_core")
    route = k1_route(b)
    bm, nw = k1_tile(b)
    with torch.cuda.device(proj.device.index):
        if route == "gated":
            y = _launch_k1_gated(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw)
        elif route == "split":
            y = _launch_k1_split(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P)
        else:
            y = _launch_k1_plain(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, di, H, N, P, bm, nw)
        block_n = triton.next_power_of_2(di)
        slice_ = triton.next_power_of_2(triton.cdiv(di, k2_programs(b)))
        split = triton.cdiv(di, slice_)
        xo = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
        scr = torch.empty((b, di), dtype=torch.bfloat16, device=proj.device)
        k2_warps = min(max(block_n // 256, 1), 8)
        _tp_k2_kernel[(b, split)](
            y, proj, conv_state, theta, gamma, xo, scr, send, recv,
            y.stride(0), proj.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
            xo.stride(0), scr.stride(0), di, float(n_all), float(eps),
            DI=di, NS=N, BLOCK_N=block_n, SLICE=slice_, NPART=split, GATED=route != "plain",
            LANES=32 * k2_warps, num_warps=k2_warps)
    return xo


def tp_gated_norm_soma(y: torch.Tensor, z: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, eps: float,
                       n_all: int, send: torch.Tensor, recv: torch.Tensor,
                       rstd_out: Optional[torch.Tensor] = None, n_out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x (M, N) bf16 contiguous from y, z (M, N) bf16 views with unit last stride (see the module docstring);
    ``rstd_out`` (M,) fp32: the rows' rstd stored there as well; ``n_out`` (M, N) bf16 contiguous: the norm n the
    read-out was formed from (both for the stage-wise tests)."""
    require_cuda(y, "tp_gated_norm_soma")
    if y.dim() != 2 or tuple(z.shape) != tuple(y.shape) or y.dtype != torch.bfloat16 or z.dtype != torch.bfloat16 \
            or y.stride(-1) != 1 or z.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_gated_norm_soma: y, z must be bf16 (M, N) with unit last strides, got "
                           f"{y.dtype} {tuple(y.shape)}, {z.dtype} {tuple(z.shape)}")
    M, N = (int(s) for s in y.shape)
    if N > 16384:
        raise RuntimeError(f"neuronspark.tp_gated_norm_soma: at most 16384 channels per GPU, got {N}")
    _region(send, recv, M, y.device, "tp_gated_norm_soma")
    if n_out is not None and (tuple(n_out.shape) != (M, N) or n_out.dtype != torch.bfloat16
                              or not n_out.is_contiguous()):
        raise RuntimeError(f"neuronspark.tp_gated_norm_soma: n_out must be contiguous bf16 ({M}, {N})")
    out = torch.empty((M, N), dtype=torch.bfloat16, device=y.device)
    if M:
        with torch.cuda.device(y.device.index):
            _tp_gated_norm_soma_kernel[(M,)](y, z, out, n_out if n_out is not None else out, theta, gamma,
                                             rstd_out if rstd_out is not None else theta,
                                             send, recv, y.stride(0), z.stride(0), out.stride(0), N, float(n_all),
                                             float(eps), VEC=GN_VEC, T=GN_THREADS,
                                             NREP_EFF=triton.cdiv(N, GN_THREADS * GN_VEC),
                                             KEEP=rstd_out is not None, STORE_N=n_out is not None,
                                             num_warps=GN_WARPS)
    return out


# ================================================================================================ references
def _norm_soma_ref(g: torch.Tensor, ss_peer: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, eps: float,
                   n_all: int, rstd: Optional[torch.Tensor] = None) -> torch.Tensor:
    if rstd is None:
        rstd = torch.rsqrt((g.pow(2).sum(dim=-1) + ss_peer) / n_all + eps)
    n = (g * rstd.unsqueeze(-1)).to(torch.bfloat16)
    return dendritic_soma_readout_ref(n, theta, gamma)


def tp_gated_norm_soma_ref(y: torch.Tensor, z: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, eps: float,
                           n_all: int, ss_peer: torch.Tensor, rstd: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pure torch: g = y z sigmoid(z) (fp32), rstd from this GPU's and the peer's (M,) sums of squares over n_all
    channels (or the given (M,) ``rstd``: the read-out stage alone), n = bf16(g rstd), the soma read-out."""
    zf = z.float()
    g = y.float() * (zf * torch.sigmoid(zf))
    return _norm_soma_ref(g, ss_peer, theta, gamma, eps, n_all, rstd)


def tp_step_core_ref(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps, n_all,
                     ss_peer: torch.Tensor) -> torch.Tensor:
    """Pure torch: the one-GPU step's K1 reference on the local heads (states advanced in place, every conv tail
    rolled), the gate ``bf16(y) * silu(z)``, then the gated norm with the peer's (b,) sums of squares and the soma
    read-out."""
    di = int(theta.shape[0])
    y = dendritic_step_k1_ref(proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta)
    z = proj[:, :di].float()
    g = y.to(torch.bfloat16).float() * (z * torch.sigmoid(z))
    return _norm_soma_ref(g, ss_peer, theta, gamma, eps, n_all)
