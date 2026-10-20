"""Per-head preparation of the memory layer's pool read (training).

Operators (registry names):

* ``pool_heads(q, k, gate, gain, wbias, H, dk)``: for every head ``h < H`` (channel slice ``[h dk, (h + 1) dk)``)
  ``q_h = F.normalize(q[.., h]) * gain[h]``, ``k_h = F.normalize(k[.., h])`` and ``g_h = gate[.., h] + wbias[h]``, from
  bf16 ``q`` / ``k`` (T, B, H dk) after RoPE, the fp32 write gate ``gate`` (T, B, H dk), the bf16 (H,) match gain
  ``gain`` and the fp32 (H,) write bias ``wbias``; returns ``(qn, kn, g)``, three H-tuples of fp32 (T, B, dk) tensors.
* ``value_split(v, H, dv)``: the H value slices ``v[..., h dv:(h + 1) dv]`` (views of v), with a backward that forms
  v's gradient in one pass per head.

Numerics contract
-----------------
* Forward: one launch for all heads.  The per-head norms are ``sqrt.rn`` of an in-kernel fp32 tree sum of squares
  (fp32 reduction-order class against torch's ``linalg_vector_norm``); ``clamp_min(norm, 1e-12)``, the IEEE divisions
  and the gain product are single-rounded inline PTX; ``gate + wbias`` is one ``add.rn``.  The match gain enters as
  its bf16 value (training regime: the gain is rounded to the query dtype before it is applied; the inference ring
  write keeps it in fp32, see ``pool_ring``).
* Backward: two launches (the k set; the q and gate sets) form the ``DivBackward0`` / ``ClampMinBackward0`` /
  ``LinalgVectorNormBackward0`` chain of ``F.normalize`` with the bf16 roundings torch makes (each head's slice gradient
  rounded to bf16, then the "+ 0" of the other heads' zero-filled slice gradients), the row sums in registers, and per
  program the partial sums of the gain / bias gradients (two small torch sums).  The gain gradient is rounded to bf16
  (the gain is a bf16 tensor), then "+ 0" canonicalised.  Every head's q, k and gate output must receive a gradient
  (the softmax pool reads all of them); a missing one raises.
* ``value_split`` backward: per head its bf16 gradient plus the other heads' zero fills (one bf16 ``+ 0`` per element
  when two or more heads contribute), written into one contiguous tensor -- what the H ``SliceBackward0`` nodes and
  v's input buffer give.

Inline PTX without FTZ, ``enable_fp_fusion=False``.  Launchers are ``torch.library`` custom ops
(``neuronspark::mem_heads_*``, ``neuronspark::mem_vsplit_bwd``) with fake implementations of the same layouts.
References: :func:`pool_heads_ref` (the per-head ``F.normalize``, gain and bias lines) and :func:`value_split_ref`.
"""
import collections
from typing import List, Sequence

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.mem_norms import (  # noqa: F401  (jit helpers resolved through this module's globals)
    _f32_bits,
    _g_add,
    _g_bits,
    _g_clamp_min,
    _g_div,
    _g_mul,
    _g_n,
    _g_neg,
    _g_r16,
    _g_sqrt,
    _g_w,
    _meta,
    _pow2,
)

__all__ = ["PoolHeadsFn", "pool_heads", "pool_heads_ref", "ValueSplitFn", "value_split", "value_split_ref"]

# ------------------------------------------------------------------------------------------------ fixed constants
# Launch configuration of the training regime (fixed):
HEAD_BT, HEAD_WARPS = 4, 4            # forward: positions per program, warps (in-kernel norms)
HEAD_BWD_BT, HEAD_BWD_WARPS = 4, 4    # backward (two launches: the k set; the q + gate sets)
MAX_H = 8                             # the kernels take up to eight head pointers
NORM_EPS = 1e-12                      # F.normalize's eps
FLAT_BLOCK, FLAT_WARPS = 1024, 4      # value_split backward (flat elementwise kernel)


# ================================================================================================ kernels
@triton.jit
def _sel8(h, P0, P1, P2, P3, P4, P5, P6, P7):
    """the address of the h-th of eight pointers (int64)."""
    a = tl.where(h == 7, P7.to(tl.int64), P0.to(tl.int64))
    a = tl.where(h == 6, P6.to(tl.int64), a)
    a = tl.where(h == 5, P5.to(tl.int64), a)
    a = tl.where(h == 4, P4.to(tl.int64), a)
    a = tl.where(h == 3, P3.to(tl.int64), a)
    a = tl.where(h == 2, P2.to(tl.int64), a)
    a = tl.where(h == 1, P1.to(tl.int64), a)
    return a


@triton.jit
def _heads_fwd_kernel(K, Q, GATE, GAIN, BIAS,
                   KH0, KH1, KH2, KH3, KH4, KH5, KH6, KH7, QG0, QG1, QG2, QG3, QG4, QG5, QG6, QG7,
                   OG0, OG1, OG2, OG3, OG4, OG5, OG6, OG7, KN0, KN1, KN2, KN3, KN4, KN5, KN6, KN7,
                   QN0, QN1, QN2, QN3, QN4, QN5, QN6, QN7,
                   T, B, DK, EPSB, GNS, BNS,
                   ks0, ks1, ks2, qs0, qs1, qs2, gs0, gs1, gs2, hs0, hs1, hs2, os0, os1, os2, ns0, ns1,
                   BT: tl.constexpr, BD: tl.constexpr):
    """head h, positions t of one b: kn = |k_h|, qn = |q_h| (fp32), kh = k / clamp(kn), qg = (q / clamp(qn)) * gain[h],
    og = gate_h + bias[h]; every head's outputs are separate tensors of shared strides."""
    tt = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)
    t = (tt * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD)
    cm = c < DK
    m = tm[:, None] & cm[None, :]
    ch = (h * DK + c).to(tl.int64)
    eps = _g_bits(EPSB)
    k = _g_w(tl.load(K + (t[:, None] * ks0 + b * ks1 + ch[None, :] * ks2), mask=m, other=0.0))
    q = _g_w(tl.load(Q + (t[:, None] * qs0 + b * qs1 + ch[None, :] * qs2), mask=m, other=0.0))
    kn = _g_sqrt(tl.sum(_g_mul(k, k), axis=1))
    qn = _g_sqrt(tl.sum(_g_mul(q, q), axis=1))
    KN = _sel8(h, KN0, KN1, KN2, KN3, KN4, KN5, KN6, KN7).to(tl.pointer_type(tl.float32))
    QN = _sel8(h, QN0, QN1, QN2, QN3, QN4, QN5, QN6, QN7).to(tl.pointer_type(tl.float32))
    no = t * ns0 + b * ns1
    tl.store(KN + no, kn, mask=tm)
    tl.store(QN + no, qn, mask=tm)
    kd = tl.broadcast_to(_g_clamp_min(kn, eps)[:, None], (BT, BD))
    qd = tl.broadcast_to(_g_clamp_min(qn, eps)[:, None], (BT, BD))
    gain = _g_w(tl.load(GAIN + h * GNS))
    KH = _sel8(h, KH0, KH1, KH2, KH3, KH4, KH5, KH6, KH7).to(tl.pointer_type(tl.float32))
    QG = _sel8(h, QG0, QG1, QG2, QG3, QG4, QG5, QG6, QG7).to(tl.pointer_type(tl.float32))
    ho = t[:, None] * hs0 + b * hs1 + c[None, :].to(tl.int64) * hs2
    tl.store(KH + ho, _g_div(k, kd), mask=m)
    tl.store(QG + ho, _g_mul(_g_div(q, qd), gain), mask=m)
    g = tl.load(GATE + (t[:, None] * gs0 + b * gs1 + ch[None, :] * gs2), mask=m, other=0.0)
    bias = tl.load(BIAS + h * BNS)
    OG = _sel8(h, OG0, OG1, OG2, OG3, OG4, OG5, OG6, OG7).to(tl.pointer_type(tl.float32))
    tl.store(OG + (t[:, None] * os0 + b * os1 + c[None, :].to(tl.int64) * os2), _g_add(g, bias), mask=m)


@triton.jit
def _norm_input_grad(g, x, n, s, eps, CANON: tl.constexpr):
    """bf16 gradient of a normalize input slice: bf16(g / d) + bf16(where(n >= eps, s, 0) * masked(x / n)) [, + 0
    when other heads' zero-filled slice gradients are added to it]."""
    d = _g_clamp_min(n, eps)
    a = _g_r16(_g_div(g, d))                                              # DivBackward0 self, cast to bf16
    gs = tl.where(n >= eps, s, 0.0)                                       # ClampMinBackward0
    t = tl.where(n == 0.0, 0.0, _g_div(x, n))                             # (x / norm).masked_fill_(norm == 0, 0)
    b = _g_r16(_g_mul(gs, t))                                             # LinalgVectorNormBackward0, to bf16
    v = _g_r16(_g_add(a, b))                                              # the slice node's input buffer
    if CANON:
        v = _g_add(v, 0.0)                                                # + the other heads' zero fill
    return _g_n(v)


@triton.jit
def _heads_bwd_k_kernel(K, GK0, GK1, GK2, GK3, GK4, GK5, GK6, GK7, KN0, KN1, KN2, KN3, KN4, KN5, KN6, KN7,
                     OK, T, B, DK, EPSB, ks0, ks1, ks2, as0, as1, as2, ns0, ns1, os0, os1, os2,
                     CANON: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """the k set of head h: P_k = (-gk) * ((k / kd) / kd), its row sum s_k, the bf16 k gradient into OK's slice."""
    tt = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)
    t = (tt * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD)
    cm = c < DK
    m = tm[:, None] & cm[None, :]
    ch = (h * DK + c).to(tl.int64)
    c64 = c.to(tl.int64)
    eps = _g_bits(EPSB)
    GK = _sel8(h, GK0, GK1, GK2, GK3, GK4, GK5, GK6, GK7).to(tl.pointer_type(tl.float32))
    KN = _sel8(h, KN0, KN1, KN2, KN3, KN4, KN5, KN6, KN7).to(tl.pointer_type(tl.float32))
    gk = tl.load(GK + (t[:, None] * as0 + b * as1 + c64[None, :] * as2), mask=m, other=0.0)
    k = _g_w(tl.load(K + (t[:, None] * ks0 + b * ks1 + ch[None, :] * ks2), mask=m, other=0.0))
    no = t * ns0 + b * ns1
    kn1 = tl.load(KN + no, mask=tm, other=1.0)
    kd = tl.broadcast_to(_g_clamp_min(kn1, eps)[:, None], (BT, BD))
    pk = _g_mul(_g_neg(gk), _g_div(_g_div(k, kd), kd))
    sk = tl.broadcast_to(tl.sum(pk, axis=1)[:, None], (BT, BD))
    kn = tl.broadcast_to(kn1[:, None], (BT, BD))
    oo = t[:, None] * os0 + b * os1 + ch[None, :] * os2
    tl.store(OK + oo, _norm_input_grad(gk, k, kn, sk, eps, CANON), mask=m)


@triton.jit
def _heads_bwd_qg_kernel(Q, GAIN, GQ0, GQ1, GQ2, GQ3, GQ4, GQ5, GQ6, GQ7, QN0, QN1, QN2, QN3, QN4, QN5, QN6, QN7,
                      GG0, GG1, GG2, GG3, GG4, GG5, GG6, GG7,
                      OQ, OG, PG, BG, T, B, DK, NP, EPSB, GNS,
                      qs0, qs1, qs2, bs0, bs1, bs2, ns0, ns1, ggs0, ggs1, ggs2, os0, os1, os2, gos0, gos1, gos2,
                      HAS_G: tl.constexpr, CANON: tl.constexpr, CANON_G: tl.constexpr,
                      BT: tl.constexpr, BD: tl.constexpr):
    """the q and gate sets of head h: P_q = (-(gq gain)) * ((q / qd) / qd) and its row sum s_q, the bf16 q gradient into
    OQ's slice, the gain summands' partial sum PG[h, prog] = sum gq (q / qd), the gate gradient copied (+ 0) into OG and
    its partial sum BG[h, prog]."""
    tt = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)
    prog = tt * B + tl.program_id(1)
    t = (tt * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD)
    cm = c < DK
    m = tm[:, None] & cm[None, :]
    ch = (h * DK + c).to(tl.int64)
    c64 = c.to(tl.int64)
    eps = _g_bits(EPSB)
    GQ = _sel8(h, GQ0, GQ1, GQ2, GQ3, GQ4, GQ5, GQ6, GQ7).to(tl.pointer_type(tl.float32))
    QN = _sel8(h, QN0, QN1, QN2, QN3, QN4, QN5, QN6, QN7).to(tl.pointer_type(tl.float32))
    gq = tl.load(GQ + (t[:, None] * bs0 + b * bs1 + c64[None, :] * bs2), mask=m, other=0.0)
    q = _g_w(tl.load(Q + (t[:, None] * qs0 + b * qs1 + ch[None, :] * qs2), mask=m, other=0.0))
    no = t * ns0 + b * ns1
    qn1 = tl.load(QN + no, mask=tm, other=1.0)
    qd = tl.broadcast_to(_g_clamp_min(qn1, eps)[:, None], (BT, BD))
    gain = _g_w(tl.load(GAIN + h * GNS))
    gqq = _g_mul(gq, gain)
    pq = _g_mul(_g_neg(gqq), _g_div(_g_div(q, qd), qd))
    pg = _g_mul(gq, _g_div(q, qd))
    sq = tl.broadcast_to(tl.sum(pq, axis=1)[:, None], (BT, BD))
    tl.store(PG + (h * NP + prog), tl.sum(tl.sum(pg, axis=1), axis=0))
    qn = tl.broadcast_to(qn1[:, None], (BT, BD))
    oo = t[:, None] * os0 + b * os1 + ch[None, :] * os2
    tl.store(OQ + oo, _norm_input_grad(gqq, q, qn, sq, eps, CANON), mask=m)
    if HAS_G:
        GG = _sel8(h, GG0, GG1, GG2, GG3, GG4, GG5, GG6, GG7).to(tl.pointer_type(tl.float32))
        gg = tl.load(GG + (t[:, None] * ggs0 + b * ggs1 + c64[None, :] * ggs2), mask=m, other=0.0)
        tl.store(BG + (h * NP + prog), tl.sum(tl.sum(gg, axis=1), axis=0))
        if CANON_G:
            gg = _g_add(gg, 0.0)
        tl.store(OG + (t[:, None] * gos0 + b * gos1 + ch[None, :] * gos2), gg, mask=m)


@triton.jit
def _canon_bf16_kernel(G, O, N, S0, S1, gs0, gs1, gs2, os0, os1, os2, CANON: tl.constexpr, BLOCK: tl.constexpr):
    """O = bf16(float(G) + 0.0) (CANON: the other heads' zero-filled bf16 slice gradients added to this head's;
    -0 -> +0) or O = G (bit copy, a single contribution)."""
    lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = lin < N
    i0 = lin % S0
    rr = lin // S0
    i1 = rr % S1
    i2 = rr // S1
    g = tl.load(G + (i0 * gs0 + i1 * gs1 + i2 * gs2), mask=m, other=0.0)
    if CANON:
        g = _g_n(_g_add(_g_w(g), 0.0))
    tl.store(O + (i0 * os0 + i1 * os1 + i2 * os2), g, mask=m)


# ================================================================================================ host helpers
def _s3(t: torch.Tensor):
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def _s2(t: torch.Tensor):
    return int(t.stride(0)), int(t.stride(1))


def _pad8(ts: Sequence[torch.Tensor], dummy: torch.Tensor) -> List[torch.Tensor]:
    return list(ts) + [dummy] * (MAX_H - len(ts))


def _axes3(ref_strides, shape, tensors):
    """Iteration axes of a 3-D elementwise launch ordered by ref_strides (innermost = smallest; ties: last axis first);
    returns (S0, S1, N) and, per tensor (a list of 3 strides), its strides in that order."""
    order = sorted(range(3), key=lambda a: (ref_strides[a], -a))
    S = [int(shape[a]) for a in order]
    out = []
    for st in tensors:
        out.extend(int(st[a]) for a in order)
    return S[0], S[1], S[0] * S[1] * S[2], out


def _one_layout(ts: Sequence[torch.Tensor]):
    """the heads' gradients of one set in ONE layout (the most common one; a head in another layout is copied into
    it: same values) -> (tensors, strides)."""
    st = collections.Counter(tuple(t.stride()) for t in ts).most_common(1)[0][0]
    out = []
    for t in ts:
        if tuple(t.stride()) != st:
            t = torch.empty_strided(tuple(t.shape), st, dtype=t.dtype, device=t.device).copy_(t)
        out.append(t)
    return out, st


# ================================================================================================ pool_heads
def _heads_fwd_alloc(q, k, gate, gain, bias, H, dk):
    T, B, D = (int(v) for v in q.shape)
    dev = q.device
    k_h, q_h = k[..., :dk], q[..., :dk]
    knm = _meta(k_h).norm(2.0, -1, keepdim=True).to(torch.float32)     # (T, B, 1), the layout of torch's norm
    denom = knm.expand(T, B, dk)
    kh_st = (_meta(k_h) / denom).stride()
    qg_st = ((_meta(q_h) / denom) * _meta(gain[0])).stride()
    og_st = (_meta(gate[..., :dk]) + _meta(bias[0])).stride()
    if qg_st != kh_st:
        raise RuntimeError("neuronspark.pool_heads: q / k output layouts differ")
    kn_l = [torch.empty_strided((T, B, 1), knm.stride(), dtype=torch.float32, device=dev) for _ in range(H)]
    qn_l = [torch.empty_strided((T, B, 1), knm.stride(), dtype=torch.float32, device=dev) for _ in range(H)]
    outs_k = [torch.empty_strided((T, B, dk), kh_st, dtype=torch.float32, device=dev) for _ in range(H)]
    outs_q = [torch.empty_strided((T, B, dk), qg_st, dtype=torch.float32, device=dev) for _ in range(H)]
    outs_g = [torch.empty_strided((T, B, dk), og_st, dtype=torch.float32, device=dev) for _ in range(H)]
    return outs_q, outs_k, outs_g, qn_l, kn_l, kh_st, og_st


@torch.library.custom_op("neuronspark::mem_heads_fwd", mutates_args=(),
                         schema="(Tensor q, Tensor k, Tensor gate, Tensor gain, Tensor bias, int H, int dk) -> Tensor[]")
def _heads_fwd(q, k, gate, gain, bias, H, dk):
    """[q_0..q_{H-1}, k_0.., g_0.., qn_0.., kn_0..]: the per-head tensors, one launch."""
    T, B, D = (int(v) for v in q.shape)
    outs_q, outs_k, outs_g, qn_l, kn_l, kh_st, og_st = _heads_fwd_alloc(q, k, gate, gain, bias, H, dk)
    grid = (triton.cdiv(T, HEAD_BT), B, H)
    _heads_fwd_kernel[grid](k, q, gate, gain, bias, *_pad8(outs_k, outs_k[0]), *_pad8(outs_q, outs_q[0]),
                         *_pad8(outs_g, outs_g[0]), *_pad8(kn_l, kn_l[0]), *_pad8(qn_l, qn_l[0]),
                         T, B, dk, _f32_bits(NORM_EPS), int(gain.stride(0)), int(bias.stride(0)),
                         *_s3(k), *_s3(q), *_s3(gate), *kh_st, *og_st, *_s2(kn_l[0]),
                         BT=HEAD_BT, BD=_pow2(dk), num_warps=HEAD_WARPS, enable_fp_fusion=False)
    return outs_q + outs_k + outs_g + qn_l + kn_l


@_heads_fwd.register_fake
def _heads_fwd_fake(q, k, gate, gain, bias, H, dk):
    outs_q, outs_k, outs_g, qn_l, kn_l, _, _ = _heads_fwd_alloc(q, k, gate, gain, bias, H, dk)
    return outs_q + outs_k + outs_g + qn_l + kn_l


def _heads_bwd_alloc(q, gate_shape, has_g, H):
    T, B, D = (int(v) for v in q.shape)
    dev = q.device
    gq_full = torch.empty((T, B, D), dtype=torch.bfloat16, device=dev)
    gk_full = torch.empty((T, B, D), dtype=torch.bfloat16, device=dev)
    g_full = (torch.empty(tuple(gate_shape), dtype=torch.float32, device=dev) if has_g
              else torch.empty((0,), dtype=torch.float32, device=dev))
    d_gain = torch.empty((H,), dtype=torch.bfloat16, device=dev)
    d_bias = torch.empty((H if has_g else 0,), dtype=torch.float32, device=dev)
    return gq_full, gk_full, g_full, d_gain, d_bias


@torch.library.custom_op("neuronspark::mem_heads_bwd", mutates_args=(),
                         schema="(Tensor[] gq, Tensor[] gk, Tensor[] gg, bool has_g, Tensor q, Tensor k, Tensor gain, "
                                "Tensor[] qn, Tensor[] kn, int[] gate_shape, int H, int dk) "
                                "-> (Tensor, Tensor, Tensor, Tensor, Tensor)")
def _heads_bwd(gq, gk, gg, has_g, q, k, gain, qn, kn, gate_shape, H, dk):
    """(gq_full, gk_full, g_full, d_gain, d_bias): two launches + two small sums (g_full / d_bias empty without gate
    gradients)."""
    T, B, D = (int(v) for v in q.shape)
    dev = q.device
    gq_full, gk_full, g_full, _, _ = _heads_bwd_alloc(q, gate_shape, has_g, H)
    bt, nw = HEAD_BWD_BT, HEAD_BWD_WARPS
    npt = triton.cdiv(T, bt)
    NP = npt * B
    pg = torch.empty((H, NP), dtype=torch.float32, device=dev)
    bg = torch.empty((H, NP), dtype=torch.float32, device=dev) if has_g else None
    canon = H >= 2
    gk_l, gk_st = _one_layout(gk)
    gq_l, gq_st = _one_layout(gq)
    if has_g:
        gg_l, _ = _one_layout(gg)
    dummy_g = gg_l if has_g else [gq_l[0]] * H
    ggs = _s3(gg_l[0]) if has_g else (0, 0, 0)
    gos = _s3(g_full) if has_g else (0, 0, 0)
    qn_l, kn_l = list(qn), list(kn)
    _heads_bwd_k_kernel[(npt, B, H)](k, *_pad8(gk_l, gk_l[0]), *_pad8(kn_l, kn_l[0]), gk_full,
                                  T, B, dk, _f32_bits(NORM_EPS), *_s3(k), *gk_st, *_s2(kn_l[0]), *_s3(gk_full),
                                  CANON=canon, BT=bt, BD=_pow2(dk), num_warps=nw, enable_fp_fusion=False)
    _heads_bwd_qg_kernel[(npt, B, H)](q, gain, *_pad8(gq_l, gq_l[0]), *_pad8(qn_l, qn_l[0]), *_pad8(dummy_g, dummy_g[0]),
                                   gq_full, g_full if has_g else gq_full, pg, bg if has_g else pg,
                                   T, B, dk, NP, _f32_bits(NORM_EPS), int(gain.stride(0)),
                                   *_s3(q), *gq_st, *_s2(qn_l[0]), *ggs, *_s3(gq_full), *gos,
                                   HAS_G=has_g, CANON=canon, CANON_G=canon and has_g,
                                   BT=bt, BD=_pow2(dk), num_warps=nw, enable_fp_fusion=False)
    d_gain = pg.sum(1).to(torch.bfloat16)          # per head the reduced gain gradient, cast to bf16
    if canon:
        d_gain = d_gain + 0.0                       # + the other heads' select_backward zeros
    if has_g:
        d_bias = bg.sum(1)
        if canon:
            d_bias = d_bias + 0.0
    else:
        d_bias = torch.empty((0,), dtype=torch.float32, device=dev)
    return gq_full, gk_full, g_full, d_gain, d_bias


@_heads_bwd.register_fake
def _heads_bwd_fake(gq, gk, gg, has_g, q, k, gain, qn, kn, gate_shape, H, dk):
    return _heads_bwd_alloc(q, gate_shape, has_g, H)


def _heads_bwd_ok(gq_l, gk_l, gg_l, qn_l, kn_l) -> bool:
    """True when the fused backward applies: every head's q / k gradient present (fp32), the gate gradients all present
    (fp32) or all absent, the norms of one layout."""
    if any(g is None for g in gq_l) or any(g is None for g in gk_l):
        return False
    if any(g.dtype != torch.float32 for g in gq_l + gk_l):
        return False
    if any(g is None for g in gg_l) != all(g is None for g in gg_l):
        return False
    if gg_l[0] is not None and any(g.dtype != torch.float32 for g in gg_l):
        return False
    sn = tuple(kn_l[0].stride())
    return all(tuple(n.stride()) == sn for n in kn_l + qn_l)


class PoolHeadsFn(torch.autograd.Function):
    """All H heads' (normalised q times the gain, normalised k, biased gate) in one Function: 3H fp32 outputs."""

    @staticmethod
    def forward(ctx, q, k, gate, gain, bias, H, dk):
        outs = _heads_fwd(q, k, gate, gain, bias, int(H), int(dk))
        outs_q, outs_k, outs_g = outs[:H], outs[H:2 * H], outs[2 * H:3 * H]
        qn_l, kn_l = outs[3 * H:4 * H], outs[4 * H:5 * H]
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, k, gain, *qn_l, *kn_l)
        ctx.H, ctx.dk = int(H), int(dk)
        ctx.gate_shape = tuple(int(v) for v in gate.shape)
        return tuple(outs_q + outs_k + outs_g)

    @staticmethod
    def backward(ctx, *grads):
        H, dk = ctx.H, ctx.dk
        saved = ctx.saved_tensors
        q, k, gain = saved[0], saved[1], saved[2]
        qn_l, kn_l = list(saved[3:3 + H]), list(saved[3 + H:3 + 2 * H])
        gq_l, gk_l, gg_l = list(grads[:H]), list(grads[H:2 * H]), list(grads[2 * H:3 * H])
        if all(g is None for g in gq_l + gk_l + gg_l):
            return None, None, None, None, None, None, None
        if not _heads_bwd_ok(gq_l, gk_l, gg_l, qn_l, kn_l):
            raise RuntimeError("neuronspark.pool_heads backward: every head's q and k output needs an fp32 gradient and "
                               "the gate outputs need gradients for all heads or none (the pool read consumes all of "
                               "them); got a partial set")
        has_g = gg_l[0] is not None
        gq_full, gk_full, g_full, d_gain, d_bias = _heads_bwd(gq_l, gk_l, gg_l if has_g else gq_l, has_g, q, k, gain,
                                                              qn_l, kn_l, list(ctx.gate_shape), H, dk)
        return (gq_full, gk_full, g_full if has_g else None, d_gain, d_bias if has_g else None, None, None)


def pool_heads(q: torch.Tensor, k: torch.Tensor, gate: torch.Tensor, gain: torch.Tensor, wbias: torch.Tensor,
               H: int, dk: int):
    """(qn, kn, g): per head ``F.normalize(q[.., h]) * gain[h]``, ``F.normalize(k[.., h])`` and ``gate[.., h] +
    wbias[h]``, each an H-tuple of fp32 (T, B, dk) tensors.  ``q`` / ``k`` bf16 (T, B, H dk) CUDA, ``gate`` dense fp32
    of the same shape, ``gain`` bf16 (H,), ``wbias`` fp32 (H,).  Training regime (see the module docstring)."""
    H, dk = int(H), int(dk)
    if H < 1 or H > MAX_H:
        raise RuntimeError(f"neuronspark.pool_heads: 1..{MAX_H} heads supported, got {H}")
    for n, t, dt in (("q", q, torch.bfloat16), ("k", k, torch.bfloat16), ("gate", gate, torch.float32)):
        require_cuda(t, "pool_heads")
        if not (t.dtype == dt and t.dim() == 3 and int(t.shape[-1]) == H * dk):
            raise RuntimeError(f"neuronspark.pool_heads: {n} must be a 3-D CUDA {dt} (T, B, {H * dk}) tensor, got "
                               f"{t.dtype} {tuple(t.shape)}")
    if q.shape != k.shape or gate.shape != q.shape or k.device != q.device or gate.device != q.device:
        raise RuntimeError("neuronspark.pool_heads: q, k, gate shapes / devices differ")
    if not (isinstance(gain, torch.Tensor) and gain.dtype == torch.bfloat16 and tuple(gain.shape) == (H,)
            and gain.device == q.device):
        raise RuntimeError(f"neuronspark.pool_heads: gain must be a bf16 ({H},) tensor, got "
                           f"{getattr(gain, 'dtype', None)} {tuple(getattr(gain, 'shape', ()))}")
    if not (isinstance(wbias, torch.Tensor) and wbias.dtype == torch.float32 and tuple(wbias.shape) == (H,)
            and wbias.device == q.device):
        raise RuntimeError(f"neuronspark.pool_heads: wbias must be an fp32 ({H},) tensor, got "
                           f"{getattr(wbias, 'dtype', None)} {tuple(getattr(wbias, 'shape', ()))}")
    if gate.stride() != torch.empty_like(gate).stride() or q.numel() >= 2 ** 31:
        raise RuntimeError("neuronspark.pool_heads: gate must be dense, tensors below 2^31 elements")
    out = PoolHeadsFn.apply(q, k, gate, gain, wbias, H, dk)
    return out[:H], out[H:2 * H], out[2 * H:]


def pool_heads_ref(q: torch.Tensor, k: torch.Tensor, gate: torch.Tensor, gain: torch.Tensor, wbias: torch.Tensor,
                   H: int, dk: int):
    """The head loop's torch lines: ``F.normalize`` of the bf16 slices with fp32 norms (the bf16 autocast's norm
    policy), times the bf16 gain's value; the gate plus the fp32 bias."""
    qn, kn, g = [], [], []
    for h in range(int(H)):
        sl = slice(h * int(dk), (h + 1) * int(dk))
        kf = k[..., sl].float()
        qf = q[..., sl].float()
        kn.append(kf / kf.norm(2.0, -1, keepdim=True).clamp_min(NORM_EPS))
        qn.append((qf / qf.norm(2.0, -1, keepdim=True).clamp_min(NORM_EPS)) * gain[h].float())
        g.append(gate[..., sl].float() + wbias[h])
    return tuple(qn), tuple(kn), tuple(g)


# ================================================================================================ value_split
@torch.library.custom_op("neuronspark::mem_vsplit_bwd", mutates_args=(),
                         schema="(Tensor?[] gs, int[] shape, int dv) -> Tensor")
def _vsplit_bwd(gs, shape, dv):
    """v's gradient from the H head gradients (None = a head without gradient): per head its gradient plus the other
    heads' zero fills, into a contiguous tensor."""
    present = [g for g in gs if g is not None]
    out = torch.empty(tuple(shape), dtype=present[0].dtype, device=present[0].device)
    canon = len(present) >= 2
    for h, g in enumerate(gs):
        sl = out[..., h * dv:(h + 1) * dv]
        if g is None:
            sl.zero_()
            continue
        if g.dtype != out.dtype or tuple(g.shape) != tuple(sl.shape):
            raise RuntimeError(f"neuronspark.value_split backward: head {h} gradient {g.dtype} {tuple(g.shape)}")
        S0, S1, N, st = _axes3(g.stride(), tuple(sl.shape), [g.stride(), sl.stride()])
        _canon_bf16_kernel[(triton.cdiv(N, FLAT_BLOCK),)](g, sl, N, S0, S1, *st, CANON=canon, BLOCK=FLAT_BLOCK,
                                                          num_warps=FLAT_WARPS, enable_fp_fusion=False)
    return out


@_vsplit_bwd.register_fake
def _vsplit_bwd_fake(gs, shape, dv):
    present = [g for g in gs if g is not None]
    return torch.empty(tuple(shape), dtype=present[0].dtype, device=present[0].device)


class ValueSplitFn(torch.autograd.Function):
    """(v[..., h dv:(h + 1) dv] for h < H): views of v; backward v's gradient in one pass per head."""

    @staticmethod
    def forward(ctx, v, H, dv):
        ctx.set_materialize_grads(False)
        ctx.shape, ctx.H, ctx.dv = tuple(int(s) for s in v.shape), int(H), int(dv)
        return tuple(v[..., h * dv:(h + 1) * dv] for h in range(H))

    @staticmethod
    def backward(ctx, *gs):
        if all(g is None for g in gs):
            return None, None, None
        return _vsplit_bwd(list(gs), list(ctx.shape), ctx.dv), None, None


def value_split(v: torch.Tensor, H: int, dv: int):
    """The H value slices ``v[..., h dv:(h + 1) dv]`` of a bf16 (T, B, H dv) CUDA ``v`` (views); the backward forms v's
    gradient as the H ``SliceBackward0`` nodes and v's input buffer do (see the module docstring)."""
    require_cuda(v, "value_split")
    H, dv = int(H), int(dv)
    if not (v.dtype == torch.bfloat16 and v.dim() == 3 and int(v.shape[-1]) == H * dv and H >= 1
            and v.numel() < 2 ** 31):
        raise RuntimeError(f"neuronspark.value_split: v must be a 3-D bf16 CUDA (T, B, {H * dv}) tensor, got "
                           f"{v.dtype} {tuple(v.shape)}")
    return ValueSplitFn.apply(v, H, dv)


def value_split_ref(v: torch.Tensor, H: int, dv: int):
    return tuple(v[..., h * int(dv):(h + 1) * int(dv)] for h in range(int(H)))
