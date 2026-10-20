"""ops.triton.pool_ring: the inference ring kernels of the hippocampal layers' pool read.

The decode read (``ring_read_t1``, with and without the extra queries and the induction term, and under CUDA-graph
replay), the extra-query formation (``xq_form_t1``), the prefill ring write (``ring_write_prefill``, also on the
channel-strided T = 2 layout) and the prefill recall (``ring_recall_prefill`` with the induction matches), each against
its pure-torch reference under ``regime("infer")``, at the model's shapes (6 heads of 256, 4 extra queries of rank 64,
three induction orders, ring capacity W = 2048): decode b in {1, 2, 4, 8, 16, 32}, prefill b in {1, 8} x T in
{2, 3, 63, 64, 256, 1024, 2048}.  The rings hold unit key tags, tagged values and decreasing absolute clocks of the
row's events (+inf in empty slots), with wrapped ring pointers and rows longer than the ring among the cases.  Every
comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from neuronspark.modules.rope import rope_head_band_tables
from neuronspark.ops.triton.pool_ring import (

    ring_read_t1,
    ring_read_t1_ref,
    ring_read_t1_takes,
    ring_recall_prefill,
    ring_recall_prefill_ref,
    ring_write_prefill,
    ring_write_prefill_ref,
    xq_form_t1,
    xq_form_t1_ref,
    xq_head_gain,
    xq_weight_stack,
)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = pytest.mark.cuda

H, DK, DV = 6, 256, 256
M_XQ, R_XQ = 4, 64
N_ORD = 3
W = 2048
GAIN = 16.0                         # sqrt(dk): the pool's match scale
SPARSITY = 2.0
MATCH_DENSITY = 0.03
DECODE_BATCHES = (1, 2, 4, 8, 16, 32)
PREFILL_T = (2, 3, 63, 64, 256, 1024, 2048)
PREFILL_B = (1, 8)
# (tokens consumed before the decode step, ring pointer minus that count): first step, mid ring, wrapped pointer,
# a row longer than the ring
DECODE_CASES = ((0, 0), (700, 0), (1500, 3 * W + 517), (W + 300, 0))
# (tokens before the prefill call, ring pointer minus that count)
PREFILL_STARTS = ((0, 0), (700, 2 * W + 1100))


def _gen(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def _clocks(B, n, device, gen):
    """(B, H, n) fp32 absolute clocks of n consecutive events: the running sum of negative per-token log decays."""
    la = -(rand_tensor((B, H, n), torch.float32, device, gen).abs() * 0.1)
    return torch.cumsum(la, dim=-1)


def _rings(B, device, gen, w=W):
    pool_k = F.normalize(rand_tensor((B, H, w, DK), torch.float32, device, gen), dim=-1).to(torch.bfloat16)
    pool_gv = (rand_tensor((B, H, w, DV), torch.float32, device, gen) * 0.5).to(torch.bfloat16)
    return pool_k.contiguous(), pool_gv.contiguous()


def _head_gain(device, gen):
    return torch.exp(rand_tensor((H,), torch.float32, device, gen) * 0.3)


# ------------------------------------------------------------------------------------------------ decode read
def _decode_inputs(B, pos, shift, device, seed, with_xq=True, with_matches=True):
    g = _gen(device, seed)
    L = min(pos + 1, W)
    ptr = pos + shift                                                        # this step's slot is ptr % W
    clocks = _clocks(B, pos + 1, device, g)
    pool_cum = torch.full((B, H, W), float("inf"), dtype=torch.float32, device=device)
    d = torch.arange(L, device=device)
    slots = (ptr - d) % W
    pool_cum[:, :, slots] = clocks[:, :, pos - d]
    pool_k, pool_gv = _rings(B, device, g)
    q = F.normalize(rand_tensor((B, H, DK), torch.float32, device, g), dim=-1) * _head_gain(device, g).view(1, H, 1)
    xq = None
    if with_xq:
        psi = rand_tensor((B, M_XQ, 1), torch.float32, device, g).clamp(-3.0, 3.0)
        xq = (F.normalize(rand_tensor((B, M_XQ, DK), torch.float32, device, g), dim=-1) * torch.exp(psi)).to(
            torch.bfloat16)
    matches = None
    beta = None
    if with_matches:
        matches = [torch.rand((B, 1, W), device=device, generator=g) < MATCH_DENSITY for _ in range(N_ORD)]
        beta = rand_tensor((N_ORD, H), torch.float32, device, g)
    return {
        "q": q, "xq": xq, "cum_t": clocks[:, :, pos].contiguous(), "pool_k": pool_k, "pool_gv": pool_gv,
        "pool_cum": pool_cum, "pos": torch.full((B,), pos, dtype=torch.int64, device=device),
        "ring_ptr": torch.tensor([ptr], dtype=torch.int64, device=device), "matches": matches, "beta": beta,
    }


def _read(fn, a):
    with torch.no_grad(), regime("infer"):
        return fn(a["q"], a["xq"], a["cum_t"], a["pool_k"], a["pool_gv"], a["pool_cum"], a["pos"], a["ring_ptr"],
                  a["matches"], a["beta"], GAIN)


@pytest.mark.parametrize("B", DECODE_BATCHES)
@pytest.mark.parametrize("pos,shift", DECODE_CASES)
def test_ring_read_t1_vs_reference(device, B, pos, shift):
    a = _decode_inputs(B, pos, shift, device, seed=2000 + 7 * B + pos)
    assert_match(_read(ring_read_t1, a), _read(ring_read_t1_ref, a), f"ring_read_t1[b{B},pos{pos}]")


@pytest.mark.parametrize("with_xq,with_matches", [(False, False), (False, True), (True, False)])
def test_ring_read_t1_optional_operands(device, with_xq, with_matches):
    a = _decode_inputs(8, 900, 5, device, seed=2500, with_xq=with_xq, with_matches=with_matches)
    new = _read(ring_read_t1, a)
    M = M_XQ if with_xq else 0
    assert tuple(new.shape) == (8, 1, (H + M) * DV)
    assert_match(new, _read(ring_read_t1_ref, a), f"ring_read_t1[xq={with_xq},matches={with_matches}]")


def test_ring_read_t1_graph_replay(device):
    """A captured step replays with the current ring contents and counters (every operand read from device memory)."""
    a = _decode_inputs(32, 1200, 0, device, seed=2600)
    b = _decode_inputs(32, 1201, 0, device, seed=2601)
    with torch.no_grad(), regime("infer"):
        _read(ring_read_t1, a)                                               # eager warm-up: compiles the kernels
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ring_read_t1(a["q"], a["xq"], a["cum_t"], a["pool_k"], a["pool_gv"], a["pool_cum"], a["pos"],
                               a["ring_ptr"], a["matches"], a["beta"], GAIN)
        for k in ("q", "xq", "cum_t", "pool_k", "pool_gv", "pool_cum", "pos", "ring_ptr", "beta"):
            a[k].copy_(b[k])
        for m_a, m_b in zip(a["matches"], b["matches"]):
            m_a.copy_(m_b)
        graph.replay()
        torch.cuda.synchronize()
        assert_match(out, _read(ring_read_t1, b), "ring_read_t1[graph replay]")


def test_ring_read_t1_takes():
    assert ring_read_t1_takes(1) and ring_read_t1_takes(32)
    assert not any(ring_read_t1_takes(b) for b in (2, 4, 8, 16))


def test_ring_read_t1_refusals(device):
    a = _decode_inputs(2, 10, 0, device, seed=2700)
    with pytest.raises(RuntimeError):
        ring_read_t1(a["q"], a["xq"], a["cum_t"], a["pool_k"].float(), a["pool_gv"], a["pool_cum"], a["pos"],
                     a["ring_ptr"], a["matches"], a["beta"], GAIN)
    with pytest.raises(RuntimeError):
        ring_read_t1(a["q"], a["xq"], a["cum_t"], a["pool_k"], a["pool_gv"], a["pool_cum"], a["pos"],
                     a["ring_ptr"], a["matches"], None, GAIN)
    with pytest.raises(RuntimeError):
        ring_read_t1(a["q"], torch.zeros((2, 16, DK), device=device), a["cum_t"], a["pool_k"], a["pool_gv"],
                     a["pool_cum"], a["pos"], a["ring_ptr"], a["matches"], a["beta"], GAIN)


# ------------------------------------------------------------------------------------------------ extra-query formation
_ROPE = {}


def _rope_band(device):
    """the last head's band of the hippocampal layers' RoPE tables, fp32 (8192, 128)."""
    if device not in _ROPE:
        cos, sin = rope_head_band_tables(H * DK, H, 8192, 10000.0, device=device)
        lo = (H - 1) * (DK // 2)
        _ROPE[device] = (cos[:, lo:lo + DK // 2].contiguous(), sin[:, lo:lo + DK // 2].contiguous())
    return _ROPE[device]


def _xq_inputs(B, device, seed, shared_position):
    g = _gen(device, seed)
    cos_t, sin_t = _rope_band(device)
    if shared_position:
        p = torch.randint(0, 8192, (1,), device=device, generator=g)
        cos = cos_t[p].to(torch.bfloat16).expand(B, -1)
        sin = sin_t[p].to(torch.bfloat16).expand(B, -1)
    else:
        p = torch.randint(0, 8192, (B,), device=device, generator=g)
        cos, sin = cos_t[p].to(torch.bfloat16), sin_t[p].to(torch.bfloat16)
    weights = [rand_tensor((DK, R_XQ), torch.float32, device, g) * math.sqrt(2.0 / (DK + R_XQ)) for _ in range(M_XQ)]
    ratio = _head_gain(device, g)
    return {
        "u": rand_tensor((B, M_XQ * R_XQ), torch.float32, device, g).to(torch.bfloat16),
        "wb": xq_weight_stack(weights), "cos": cos, "sin": sin,
        "psi": (rand_tensor((B, M_XQ), torch.float32, device, g) * 1.5).to(torch.bfloat16),
        "head_gain": xq_head_gain(ratio, torch.float32),
    }


def _xq(fn, a):
    with torch.no_grad(), regime("infer"):
        return fn(a["u"], a["wb"], a["cos"], a["sin"], a["psi"], a["head_gain"])


@pytest.mark.parametrize("B", DECODE_BATCHES)
@pytest.mark.parametrize("shared", (False, True))
def test_xq_form_t1_vs_reference(device, B, shared):
    a = _xq_inputs(B, device, seed=3100 + B, shared_position=shared)
    assert_match(_xq(xq_form_t1, a), _xq(xq_form_t1_ref, a), f"xq_form_t1[b{B},shared={shared}]")


# ------------------------------------------------------------------------------------------------ prefill ring write
def _write_inputs(T, B, device, seed, variant="full", scale=1.0, strided=False):
    g = _gen(device, seed)

    def qkv():
        if strided:                                                          # the T = 2 conv form: channel stride T
            base = (rand_tensor((B, H * DK, T), torch.float32, device, g) * scale).to(torch.bfloat16)
            return base.permute(2, 0, 1).view(T, B, H, DK)
        return (rand_tensor((T, B, H, DK), torch.float32, device, g) * scale).to(torch.bfloat16)

    k, q, v = qkv(), qkv(), qkv()
    if variant == "zero_rows":
        k = k.clone()
        q = q.clone()
        k[::7] = 0
        q[3::11] = 0
    gate = rand_tensor((T, B, H, DK), torch.float32, device, g) * 2.0 * scale
    full = variant in ("full", "zero_rows")
    pool_k, pool_gv = _rings(B, device, g)
    return {
        "k": k, "q": q, "g": gate, "v": v,
        "qgain": _head_gain(device, g) if full else None,
        "wbias": rand_tensor((H,), torch.float32, device, g) * 0.5 if full else None,
        "sparsity": SPARSITY if full else 0.0,
        "pool_k": pool_k, "pool_gv": pool_gv,
        "ring_ptr": torch.randint(0, 4 * W, (1,), device=device, generator=g, dtype=torch.int64),
        "T": T, "B": B,
    }


def _write(fn, a):
    pk, pgv = a["pool_k"].clone(), a["pool_gv"].clone()
    with torch.no_grad(), regime("infer"):
        qo = fn(a["k"], a["q"], a["g"], a["v"], a["qgain"], a["wbias"], a["sparsity"], pk, pgv, a["ring_ptr"],
                a["T"], a["B"], H, W)
    return {"pool_k": pk, "pool_gv": pgv, "q": qo}


@pytest.mark.parametrize("B", PREFILL_B)
@pytest.mark.parametrize("T", PREFILL_T)
def test_ring_write_prefill_vs_reference(device, B, T):
    a = _write_inputs(T, B, device, seed=4100 + T + B)
    assert_all_match(_write(ring_write_prefill, a), _write(ring_write_prefill_ref, a),
                     prefix=f"ring_write_prefill[T{T},b{B}]")


@pytest.mark.parametrize("variant,scale,strided", [
    ("full", 0.01, False), ("full", 30.0, False), ("plain", 1.0, False), ("zero_rows", 1.0, False),
    ("full", 1.0, True)])
def test_ring_write_prefill_variants(device, variant, scale, strided):
    T = 2 if strided else 37
    a = _write_inputs(T, 8, device, seed=4200, variant=variant, scale=scale, strided=strided)
    assert_all_match(_write(ring_write_prefill, a), _write(ring_write_prefill_ref, a),
                     prefix=f"ring_write_prefill[{variant},x{scale},strided={strided}]")


# ------------------------------------------------------------------------------------------------ prefill recall
def _recall_inputs(T, B, pos0, shift, device, seed, xq, w=W):
    g = _gen(device, seed)
    N = pos0 + T
    p0 = pos0 + shift                                                       # the ring pointer at the call's start
    clocks = _clocks(B, N, device, g)
    pool_cum = torch.full((B, H, w), float("inf"), dtype=torch.float32, device=device)
    j = torch.arange(max(0, N - w), N, device=device)
    pool_cum[:, :, (p0 + (j - pos0)) % w] = clocks[:, :, j]
    pool_k, pool_gv = _rings(B, device, g, w)
    cum_tbh = clocks[:, :, pos0:N].permute(2, 0, 1).contiguous()             # (T, B, H) query clocks
    gain = _head_gain(device, g)
    if xq:
        psi = rand_tensor((M_XQ, T, B, 1), torch.float32, device, g).clamp(-3.0, 3.0)
        q_base = (F.normalize(rand_tensor((M_XQ, T, B, DK), torch.float32, device, g), dim=-1) * torch.exp(psi)
                  * gain[-1]).to(torch.bfloat16)                            # (M, T, B, dk): the stacked queries
        q4, ct3, G = q_base.permute(2, 0, 1, 3), None, M_XQ
    else:
        q_base = F.normalize(rand_tensor((T, B, H, DK), torch.float32, device, g), dim=-1) * gain.view(1, 1, H, 1)
        q4, ct3, G = q_base.permute(1, 2, 0, 3), cum_tbh.permute(1, 2, 0), H
    matches = [torch.rand((B, T, w), device=device, generator=g) < MATCH_DENSITY for _ in range(N_ORD)]
    beta = rand_tensor((N_ORD, H), torch.float32, device, g)
    return {"q_base": q_base, "q4": q4, "ct3": ct3, "cum_tbh": cum_tbh, "pool_k": pool_k, "pool_gv": pool_gv,
            "pool_cum": pool_cum, "p0": torch.tensor([p0], dtype=torch.int64, device=device), "matches": matches,
            "beta": beta, "T": T, "B": B, "G": G, "xq": xq, "W": w}


def _recall(fn, a, out_dtype=torch.bfloat16):
    with torch.no_grad(), regime("infer"):
        return fn(a["q4"], a["pool_k"], a["pool_gv"], a["pool_cum"], a["ct3"], a["p0"], (a["matches"], a["beta"]),
                  GAIN, a["T"], a["B"], a["G"], a["W"], DK, DV, out_dtype, a["xq"], H - 1 if a["xq"] else 0)


@pytest.mark.parametrize("xq", (False, True), ids=("heads", "extra_queries"))
@pytest.mark.parametrize("B", PREFILL_B)
@pytest.mark.parametrize("T", PREFILL_T)
@pytest.mark.parametrize("pos0,shift", PREFILL_STARTS)
def test_ring_recall_prefill_vs_reference(device, pos0, shift, T, B, xq):
    a = _recall_inputs(T, B, pos0, shift, device, seed=5100 + T + B + pos0, xq=xq)
    assert_match(_recall(ring_recall_prefill, a), _recall(ring_recall_prefill_ref, a),
                 f"ring_recall_prefill[T{T},b{B},pos{pos0},xq={xq}]")


@pytest.mark.parametrize("w", (384, 768), ids=("split1", "split2"))
def test_ring_recall_prefill_ring_sizes(device, w):
    """rings whose W / 128 has fewer factors of two than RECALL_STATS_SPLIT: the statistics pass over fewer slot ranges"""
    a = _recall_inputs(256, 2, 300, 9, device, seed=5300 + w, xq=False, w=w)
    assert_match(_recall(ring_recall_prefill, a), _recall(ring_recall_prefill_ref, a), f"ring_recall_prefill[W{w}]")


@pytest.mark.parametrize("T", (17, 256))
def test_ring_recall_prefill_fp32_output_and_no_term(device, T):
    a = _recall_inputs(T, 2, 300, 9, device, seed=5200 + T, xq=False)
    out = _recall(ring_recall_prefill, a, out_dtype=torch.float32)
    assert out.dtype == torch.float32
    assert_match(out, _recall(ring_recall_prefill_ref, a, out_dtype=torch.float32), f"ring_recall_prefill[fp32,T{T}]")
    with torch.no_grad(), regime("infer"):
        args = (a["q4"], a["pool_k"], a["pool_gv"], a["pool_cum"], a["ct3"], a["p0"], None, GAIN, T, 2, H, W, DK, DV,
                torch.bfloat16, False, 0)
        assert_match(ring_recall_prefill(*args), ring_recall_prefill_ref(*args), f"ring_recall_prefill[no term,T{T}]")
