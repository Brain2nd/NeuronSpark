"""The operator-signature contract of the package.

Every operator the model calls is listed here once, with the module that implements it, its kernel entry point
``<name>`` and its pure-torch reference ``<name>_ref``.  Both have the same parameter list (``params`` below); shapes are
given for 1MIX.  Modules do not import kernels directly: they pick a namespace once, ``ops = neuronspark.ops.namespace(
impl)`` with ``impl = "kernel"`` (``KERNELS``) or ``"ref"`` (``REFS``), and call ``ops.<name>(...)``.  The namespace
imports the implementing module lazily on first use.

Regimes: ``train`` = whole-row forward (+ backward), ``prefill`` = inference call with T >= 2, ``decode`` = inference
call with T == 1.  ``mutates`` names the arguments an op writes in place (declared to ``torch.library`` by the
implementing module).  Pure-torch computations of the modules (lexical hash and lookup, induction matches, copy
statistics, decay, pool constants, novelty, PKM softmax, the decode-time torch forms) have no entry: the module code
is their own reference.

The registered callables are plain Python functions (the public function of each family, which applies the family's
autograd Function or calls its ``torch.library`` custom ops), so :func:`check_signature` can read their parameter lists.
The numerics contract of each op (rounding points, reduction orders, launch constants) is in the implementing module's
docstring.  Each op is checked against its reference with the per-operator tolerance rule of
:mod:`neuronspark.utils.testing`.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

__all__ = ["OpSpec", "OPS", "op_names", "resolve", "check_signature", "check_all_signatures"]


@dataclass(frozen=True)
class OpSpec:
    """One registered operator."""

    name: str
    module: str                      # implementing module (``neuronspark.ops....``)
    params: Tuple[str, ...]          # parameter names in order; "*" marks the start of keyword-only parameters
    returns: str                     # outputs (shapes / dtypes, 1MIX)
    regimes: Tuple[str, ...]         # subset of ("train", "prefill", "decode")
    mutates: Tuple[str, ...] = ()    # arguments written in place
    doc: str = ""                    # inputs (shapes / dtypes, 1MIX)

    @property
    def kernel(self) -> str:
        return f"{self.module}:{self.name}"

    @property
    def ref(self) -> str:
        return f"{self.module}:{self.name}_ref"


_T = "neuronspark.ops.triton."


def _op(name, module, params, returns, regimes, mutates=(), doc=""):
    return OpSpec(name=name, module=module, params=tuple(params.split()), returns=returns,
                  regimes=tuple(regimes.split()), mutates=tuple(mutates.split()) if mutates else (), doc=doc)


_SPECS: List[OpSpec] = [
    # ---------------------------------------------------------------------------------------------- shared
    _op("linear", "neuronspark.ops.linear", "x weight",
        "(..., N) bf16; backward: dx in x.dtype, dW fp32 as weight.grad", "train prefill decode",
        doc="x (..., K) fp32|bf16, weight (N, K) fp32; inference: the bf16 weight cache when present"),
    # ---------------------------------------------------------------------------------------------- SD layer
    _op("sd_norm", _T + "sd_prenorm", "h gain eps",
        "hn bf16, same shape and layout as h", "train",
        doc="h (T, B, D) fp32 with unit last stride, gain (D,) fp32"),
    _op("sd_resid_norm", _T + "sd_prenorm", "h upd gain_next eps",
        "(h_new (T, B, D) fp32, hn_next (T, B, D) bf16)", "train",
        doc="h (T, B, D) fp32, upd (T, B, D) bf16, gain_next (D,) fp32: h_new = h + upd.float(), hn_next = norm(h_new)"),
    _op("sd_chunk_scan_soma", "neuronspark.ops.sd_combined",
        "zxbcdt conv_w conv_b dt_bias A D theta gamma * alpha chunk_size",
        "xs (B*T, 9216) bf16; grads for zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma", "train",
        doc="zxbcdt (B, T, 18832) bf16, conv_w (9472, 4) fp32, conv_b (9472,), dt_bias (144,), A (144,) fp32 = "
            "-exp(A_log), D (144,), theta (9216,), gamma (9216,); alpha = 8.0, chunk_size = 256; T % 256 == 0"),
    _op("sd_prenorm_rows", _T + "sd_step", "x gain eps",
        "(M, D) bf16", "prefill decode",
        doc="x (M, D) fp32 rows, gain (D,) fp32"),
    _op("sd_conv_silu_prefill", _T + "sd_conv", "xBC conv_w conv_b tail",
        "(xbc (B, T, 9472) bf16 contiguous, tail_new (B, 9472, 3) fp32)", "prefill",
        doc="xBC (B, T, 9472) bf16 strided view of the projection, conv_w (9472, 4), conv_b (9472,), "
            "tail (B, 9472, 3) fp32 = the previous three inputs"),
    _op("sd_chunk_scan_prefill", "neuronspark.ops.sd_prefill",
        "x dt A Bm Cm D dt_bias initial_state * chunk_size",
        "(y (B, T, 144, 64) bf16, final_state (B, 144, 64, 128) fp32)", "prefill",
        doc="x (B, T, 144, 64) bf16, dt (B, T, 144) bf16, A (144,) fp32, Bm / Cm (B, T, 1, 128) bf16, D (144,), "
            "dt_bias (144,), initial_state (B, 144, 64, 128) fp32; chunk_size = 256"),
    _op("sd_gated_norm_soma", _T + "sd_gated_norm", "y z theta gamma eps",
        "x (M, 9216) bf16 = bf16(S(n, theta) * gamma), n the bf16 gated RMSNorm (unit weight)", "prefill",
        doc="y (M, 9216) bf16, z (M, 9216) bf16 (strided view of the projection), theta / gamma (9216,) fp32"),
    _op("sd_step_core", _T + "sd_step",
        "proj conv_state ssm_state conv_w conv_b A dt_bias D theta gamma eps",
        "xs (B, 9216) bf16", "decode", mutates="conv_state ssm_state",
        doc="proj (B, 18832) bf16, conv_state (B, 9472, 3) fp32, ssm_state (B, 144, 64, 128) bf16|fp32 (stored in its "
            "own dtype), conv_w (9472, 4), conv_b, A (144,), dt_bias, D, theta (9216,), gamma (9216,)"),
    _op("sd_res_add", _T + "sd_residual", "h upd",
        "(T, B, 1536) fp32 contiguous = h + upd.float()", "prefill",
        doc="h (T, B, 1536) fp32 residual stream, upd (T, B, 1536) bf16 with unit last stride (the transposed view of "
            "the (B, T, 1536) W_out output)"),
    # ---------------------------------------------------------------------------------------------- memory norms
    _op("row_rmsnorm", _T + "mem_norms", "x weight eps",
        "(T, B, C) fp32", "train",
        doc="x (T, B, C) fp32|bf16, weight (C,) fp32 or None (gain 1)"),
    _op("rmsnorm_rows", _T + "mem_norms", "x eps weight zero_centered out_dtype",
        "(M, N) in out_dtype (default x.dtype)", "prefill decode",
        doc="x (M, N) any float, weight (N,) fp32 or None, zero_centered: gain 1 + weight; N % 128 == 0, N <= 4096"),
    _op("rmsnorm_gate_rows", _T + "mem_norms", "x eps weight gate",
        "(..., N) bf16 = bf16(sigmoid(gate) * zero-centred RMSNorm(x))", "prefill decode",
        doc="x (..., 1536) fp32, weight (1536,) fp32 (zero-centred), gate (..., 1536) bf16 output-gate logits"),
    _op("out_norm_gate", _T + "mem_norms", "attn w gate_logits eps",
        "(T, B, 1536) fp32 = zero-centred RMSNorm(attn) * sigmoid(gate_logits)", "train",
        doc="attn (T, B, 1536) fp32, w (1536,) fp32 (zero-centred), gate_logits (T, B, 1536) bf16"),
    _op("nov_norm", _T + "mem_norms", "r eps",
        "(T, B, 1536) fp32", "train",
        doc="r (T, B, 1536) bf16"),
    # ---------------------------------------------------------------------------------------------- memory heads / rope
    _op("pool_heads", _T + "mem_heads", "q k gate gain wbias H dk",
        "(qn [H x (T, B, dk) fp32], kn [H x (T, B, dk) fp32], g [H x (T, B) fp32])", "train",
        doc="q, k (T, B, 1536) bf16 after RoPE, gate (T, B, 1536) fp32, gain (6,) bf16 match gain, wbias (6,) fp32"),
    _op("value_split", _T + "mem_heads", "v H dv",
        "H tensors (T, B, dv) bf16", "train",
        doc="v (T, B, 1536) bf16"),
    _op("silu_rope", _T + "mem_rope", "x cos sin",
        "(T, B, C) bf16 = rope(silu(x))", "train",
        doc="x (T, B, C) bf16, cos / sin (T, 768) bf16"),
    _op("rope_paired", _T + "mem_rope", "x cos sin",
        "(T, B, C) bf16", "train",
        doc="x (T, B, C) bf16, cos / sin (T, 768) bf16"),
    _op("rope_interleaved", _T + "mem_rope", "x cos sin",
        "(T, B, C) bf16", "prefill",
        doc="x (T, B, C) bf16, cos / sin broadcastable bf16"),
    # ---------------------------------------------------------------------------------------------- memory residual glue
    _op("value_residual", _T + "mem_residual", "v lam carry glu_add",
        "v' (T, B, 1536) bf16", "train",
        doc="v (T, B, 1536) bf16, lam fp32 mix, carry (T, B, 1536) bf16 or None (memory layer 0), "
            "glu_add (T, B, 256) fp32"),
    _op("zc_center", _T + "mem_residual", "z zcum",
        "(T, B, C) fp32 contiguous", "train",
        doc="z (T, B, C) fp32, zcum (T, B, C) fp32 running sum"),
    _op("read_gain_residual", _T + "mem_residual", "h r f",
        "h' (T, B, D) fp32", "train prefill decode",
        doc="h (T, B, D) fp32, r (T, B, D) bf16, f (T, B) fp32 read gain"),
    _op("pkm_tail", _T + "mem_residual", "h ffn y rho",
        "h_out (T, B, D) fp32", "train prefill decode",
        doc="h (T, B, D) fp32, ffn (T, B, D) bf16, y (T, B, D) bf16, rho (T, B) fp32 (no gradient)"),
    # ---------------------------------------------------------------------------------------------- cumulative sums
    _op("cumsum0", _T + "cumsum", "x",
        "inclusive cumsum over dim 0, fp32", "train",
        doc="x (T, ...) fp32; the narrow / wide dispatch is internal and fixed"),
    _op("exact_cumsum0", _T + "cumsum", "x",
        "(T, N) fp32", "prefill",
        doc="x (T, N) fp32: sequential RN additions"),
    _op("centred_running", _T + "cumsum", "x run_sum count out_dtype",
        "(T, B, C) in out_dtype", "prefill", mutates="run_sum",
        doc="x (T, B, C), run_sum (B, C) fp32 carried sum, count (B,) int64 tokens before this call"),
    # ---------------------------------------------------------------------------------------------- qkv short conv
    _op("short_conv_train", _T + "short_conv", "qkv weight",
        "(T, B, 4608) bf16 T-major; backward: dW (4608, 1, 4) fp32 as weight.grad", "train",
        doc="qkv (T, B, 4608) bf16, weight (4608, 1, 4) fp32"),
    _op("short_conv_silu_prefill", _T + "short_conv", "x hist weight",
        "(T, B, 4608) bf16 silu(conv(x))", "prefill", mutates="hist",
        doc="x (T, B, 4608) bf16 with T >= 3, hist (B, 4608, 3) fp32, weight (4608, 1, 4) fp32"),
    _op("short_conv_step", _T + "short_conv", "x hist weight",
        "(B, 4608) bf16", "decode", mutates="hist",
        doc="x (B, 4608) bf16, hist (B, 4608, 3) fp32, weight (4608, 1, 4) fp32"),
    # ---------------------------------------------------------------------------------------------- pool attention (train)
    _op("pool_induction_term", _T + "pool_bias", "beta ind",
        "(B, T, T) fp32", "train",
        doc="beta (3,) fp32 = the head's slice of pool_ind_beta, ind 3 x (B, T, T) bool match matrices"),
    _op("pool_bias_mask", _T + "pool_bias", "cum X T B",
        "(B, 1, T, T) bf16 additive mask", "train",
        doc="cum (T, B) fp32 clocks, X (B, T, T) fp32 induction term"),
    _op("pool_attention", _T + "pool_attn", "q k v mask scale kshared",
        "(T, B, [H,] 256) bf16", "train",
        doc="q, k (T, B, [H,] 256) fp32 unit rows, v bf16, mask (B, 1, T, T) bf16, scale 16.0, kshared: k shared "
            "across the query group"),
    # ---------------------------------------------------------------------------------------------- inference rings
    _op("ring_write_prefill", _T + "pool_ring",
        "k_h q_h g_h v_h qgain wbias sparsity pool_k pool_gv ring_ptr T B H W",
        "q fp32: the gained unit queries of the call", "prefill", mutates="pool_k pool_gv",
        doc="per-head k / q / gate / v of this call, qgain (H,) fp32 match gain, wbias (H,) fp32, sparsity 2.0, "
            "rings (B, H, W, 256) bf16"),
    _op("ring_recall_prefill", _T + "pool_ring",
        "q4 k_ring gv_ring cum_ring ct3 p0 ind4 scale T B G W dk dv out_dtype xq hr",
        "(T, B, G * dv) out_dtype", "prefill",
        doc="queries, rings and clocks of the call, p0 first position, ind4 = (match matrices, beta), scale 16.0; "
            "xq = False / hr = 0 for the six heads, xq = True for the extra queries"),
    _op("ring_read_t1", _T + "pool_ring",
        "q xq cum_t pool_k pool_gv pool_cum pos ring_ptr matches beta gain",
        "(B, 1, 1536 + extra-query width) bf16", "decode",
        doc="q (B, 6, 256), xq (B, 4, 256), cum_t this step's clocks, rings, counters, induction matches, beta, gain; "
            "decode batch b in {1, 32}"),
    _op("xq_form_t1", _T + "pool_ring", "u wb cos sin psi head_gain",
        "(B, 4, 256) fp32", "decode",
        doc="u (B, 256), wb (4, 256, 64) bf16, cos / sin, psi (B, 4), head_gain; decode batch b not in {1, 32}"),
    # ---------------------------------------------------------------------------------------------- PLIF
    _op("plif_centred_input", _T + "plif", "x",
        "x.dtype (T, B, C): x minus its running mean", "train",
        doc="x (T, B, C) fp32"),
    _op("plif_scan_train", _T + "plif", "w_eta b_eta zp x v_th * alpha out_dtype",
        "spikes S (T, B, C) out_dtype; grads for w_eta, b_eta, zp, x, v_th", "train",
        doc="w_eta, b_eta (C,) fp32, zp (T, B, C) bf16 eta projection, x (T, B, C) fp32|bf16, v_th (C,) fp32; "
            "alpha = 8.0, T >= 2"),
    _op("plif_gate_prefill", _T + "plif", "x w_eta b_eta zp",
        "(a, drive) (T, B, C) fp32", "prefill",
        doc="x (T, B, C), w_eta, b_eta (C,) fp32, zp (T, B, C) bf16"),
    _op("plif_scan_prefill", _T + "plif", "a drive v_th v0 out_dtype",
        "spikes S (T, B, C) out_dtype", "prefill", mutates="v0",
        doc="a, drive (T, B, C) fp32, v_th (C,) fp32, v0 (B, C) fp32 carried membrane"),
    # ---------------------------------------------------------------------------------------------- PKM
    _op("pkm_topk_train", _T + "pkm_topk", "s k",
        "(sc fp32, u, ii, jj int64) each (T, B, 4, 16); backward -> ds", "train",
        doc="s (T, B, 4, 2, 128) fp32 half-query scores, k = 16"),
    _op("pkm_value_read", _T + "pkm_read", "w u V",
        "(T, B, 768) fp32; backward: dw, dV", "train",
        doc="w (T, B, 64) fp32 slot weights, u (T, B, 64) int64 slot ids, V (16384, 768) fp32"),
    _op("pkm_topk_prefill", _T + "pkm_topk", "s1 s2 k",
        "(sc, u, ii, jj)", "prefill",
        doc="s1, s2 (T, B, 4, 128) fp32 half-query scores, k = 16"),
    _op("pkm_weighted_sum", _T + "pkm_read", "w u V",
        "(T, B, 768) fp32", "prefill",
        doc="w fp32 slot weights, u int64 slot ids, V (16384, 768) fp32"),
    _op("pkm_topk_softmax_t1", _T + "pkm_topk", "s_raw beta k",
        "(sc, w, u, ii, jj)", "decode",
        doc="s_raw (B, 4, 2, 128) fp32, beta (4, 2) fp32 score gains, k = 16"),
    # ---------------------------------------------------------------------------------------------- losses
    _op("copy_switch_aux", _T + "copy_switch", "ids targets feats weights lam",
        "scalar fp32", "train",
        doc="ids, targets (B, T) int64, feats 6 x (z, r, pi) detached, weights 6 x (w, v, b), lam = 0.5"),
    _op("balance_ce_per_token", _T + "lm_tail", "logits labels contribs widths",
        "(R,) fp32 per-token cross entropy of the balanced logits", "train",
        doc="logits (R, V) bf16 dense, labels (R,) int64 (-100 ignored), contribs ((R, 16), (R, 32)) bf16, "
            "widths (16, 32)"),
    _op("balance_add", _T + "balance_add", "logits contribs widths",
        "(R, V) bf16 dense balanced logits", "train prefill decode",
        doc="logits (R, 151669) bf16 readout (unit column stride), contribs ((R, 16), (R, 32)) bf16 contiguous, "
            "widths (16, 32)"),
]

OPS: Dict[str, OpSpec] = {s.name: s for s in _SPECS}
if len(OPS) != len(_SPECS):  # pragma: no cover - guards the table above
    raise RuntimeError("neuronspark.ops.registry: duplicate operator names")


def op_names() -> Tuple[str, ...]:
    return tuple(OPS)


def _load(path: str) -> Callable:
    module, _, attr = path.partition(":")
    mod = importlib.import_module(module)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise AttributeError(f"neuronspark.ops: {module} does not define {attr} (required by the op registry)") from exc


def resolve(name: str, impl: str = "kernel") -> Callable:
    """The callable of op ``name``: ``impl="kernel"`` -> ``<module>:<name>``, ``impl="ref"`` -> ``<module>:<name>_ref``."""
    if name not in OPS:
        raise KeyError(f"neuronspark.ops: unknown op {name!r}")
    if impl not in ("kernel", "ref"):
        raise ValueError(f"impl must be 'kernel' or 'ref', got {impl!r}")
    spec = OPS[name]
    return _load(spec.kernel if impl == "kernel" else spec.ref)


def _param_layout(fn: Callable) -> Tuple[str, ...]:
    out: List[str] = []
    star = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"{fn.__qualname__}: registered ops take explicit parameters only")
        if p.kind == p.KEYWORD_ONLY and not star:
            out.append("*")
            star = True
        out.append(p.name)
    return tuple(out)


def check_signature(name: str) -> None:
    """Raise unless the kernel and the reference of op ``name`` both have exactly the registered parameter list."""
    spec = OPS[name]
    for impl in ("kernel", "ref"):
        got = _param_layout(resolve(name, impl))
        if got != spec.params:
            raise TypeError(f"neuronspark.ops.{name} ({impl}): parameters {got} differ from the registry {spec.params}")


def check_all_signatures(names: Iterable[str] = ()) -> None:
    """:func:`check_signature` for ``names`` (default: every registered op)."""
    for n in (tuple(names) or op_names()):
        check_signature(n)


# ================================================================================================== memory prefill block
# The memory layers' prefill operators (copy-detector statistics, extra-query formation, lexical read, value-residual
# mix), registered into OPS after the main table.
_MEMORY_PREFILL_SPECS: List[OpSpec] = [
    _op("copy_stats_prefill", _T + "copy_stats", "ring ring_ptr pos T",
        "(maximum, lse) each (T, B) fp32 contiguous; -inf where a query has no valid slot", "prefill",
        doc="ring (B, W, 96) fp32 unit operands (the call's own written at (ring_ptr + t) % W), ring_ptr (1,) int64 "
            "pointer at the call's start, pos (B,) int64 tokens before the call, T the call's tokens"),
    _op("xq_queries_prefill", _T + "xq_query", "qs psi head_gain cos sin",
        "(B, 4, T, 256) bf16 gained unit extra queries (view of a contiguous (4, T, B, 256) buffer)", "prefill",
        doc="qs 4 x (T, B, 256) bf16 contiguous projections, psi (T, B, 4) fp32 clamped gain projection, head_gain "
            "one-element bf16 match gain of the last head, cos / sin (T, B, 128) bf16 last-head RoPE band"),
    _op("lex_read_prefill", _T + "lex_read", "h q_raw k_raw o q_gain k_gain beta eps",
        "(T, B, 1536) fp32 contiguous h + alpha * o", "prefill",
        doc="h (T, B, 1536) fp32, q_raw / k_raw (T, B, 64) bf16, o (T, B, 1536) bf16, q_gain / k_gain (64,) fp32, "
            "beta (1,) fp32, eps = 1e-6"),
    _op("vr_mix_prefill", _T + "vr_mix", "v carry gl lam0 glu_add",
        "(T, B, 1536) bf16 contiguous value stream after the value-residual mix and the GLU tail", "prefill",
        doc="v (T, B, 1536) bf16, carry (T, B, 1536) bf16 or None (memory layer 0), gl (T, B, 6) bf16 gate projection, "
            "lam0 (1536,) fp32 contiguous static mix, glu_add (T, B, 256) fp32 GLU addend of the last head"),
]
for _mem_spec in _MEMORY_PREFILL_SPECS:
    if _mem_spec.name in OPS:
        raise RuntimeError(f"neuronspark.ops.registry: duplicate operator name {_mem_spec.name!r}")
    OPS[_mem_spec.name] = _mem_spec
_SPECS.extend(_MEMORY_PREFILL_SPECS)
del _mem_spec
