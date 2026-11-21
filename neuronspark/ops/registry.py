"""The operator-signature contract of the package.

Every operator the model calls is listed here once, with the module that implements it, its kernel entry point
``<name>`` and its pure-torch reference ``<name>_ref``.  Both have the same parameter list (``params`` below); shapes are
given for the default configuration.  Modules do not import kernels directly: they pick a namespace once,
``ops = neuronspark.ops.namespace(impl)`` with ``impl = "kernel"`` (``KERNELS``) or ``"ref"`` (``REFS``), and call
``ops.<name>(...)``.  The namespace imports the implementing module lazily on first use.

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
    returns: str                     # outputs (shapes / dtypes, default configuration)
    regimes: Tuple[str, ...]         # subset of ("train", "prefill", "decode")
    mutates: Tuple[str, ...] = ()    # arguments written in place
    doc: str = ""                    # inputs (shapes / dtypes, default configuration)

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
    _op("linear_add", "neuronspark.ops.linear", "x weight addend x_scale",
        "(..., N) in addend's dtype = addend + linear(x [* x_scale], weight), a new tensor", "decode",
        doc="x (..., K) fp32|bf16, weight (N, K) fp32, addend (..., N) fp32|bf16, x_scale None or bf16 of x's shape "
            "(x bf16 then)"),
    # ---------------------------------------------------------------------------------------------- dendritic neuron
    # layer
    _op("dendritic_norm", _T + "dendritic_prenorm", "h gain eps",
        "hn bf16, same shape and layout as h", "train",
        doc="h (T, B, D) fp32 with unit last stride, gain (D,) fp32"),
    _op("dendritic_resid_norm", _T + "dendritic_prenorm", "h upd gain_next eps",
        "(h_new (T, B, D) fp32, hn_next (T, B, D) bf16)", "train",
        doc="h (T, B, D) fp32, upd (T, B, D) bf16, gain_next (D,) fp32: h_new = h + upd.float(), hn_next = norm(h_new)"),
    _op("dendritic_chunk_scan_soma", "neuronspark.ops.dendritic_combined",
        "zxbcdt conv_w conv_b dt_bias A D theta gamma * alpha chunk_size",
        "xs (B*T, 9216) bf16; grads for zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma", "train",
        doc="zxbcdt (B, T, 18832) bf16, conv_w (9472, 4) fp32, conv_b (9472,), dt_bias (144,), A (144,) fp32 = "
            "-exp(A_log), D (144,), theta (9216,), gamma (9216,); alpha = 8.0, chunk_size = 256; T % 256 == 0"),
    _op("dendritic_prenorm_rows", _T + "dendritic_step", "x gain eps",
        "(M, D) bf16", "prefill decode",
        doc="x (M, D) fp32 rows, gain (D,) fp32"),
    _op("dendritic_conv_silu_prefill", _T + "dendritic_conv", "xBC conv_w conv_b tail",
        "(xbc (B, T, 9472) bf16 contiguous, tail_new (B, 9472, 3) fp32)", "prefill",
        doc="xBC (B, T, 9472) bf16 strided view of the projection, conv_w (9472, 4), conv_b (9472,), "
            "tail (B, 9472, 3) fp32 = the previous three inputs"),
    _op("dendritic_chunk_scan_prefill", "neuronspark.ops.dendritic_prefill",
        "x dt A Bm Cm D dt_bias initial_state * chunk_size",
        "(y (B, T, 144, 64) bf16, final_state (B, 144, 64, 128) fp32)", "prefill",
        doc="x (B, T, 144, 64) bf16, dt (B, T, 144) bf16, A (144,) fp32, Bm / Cm (B, T, 1, 128) bf16, D (144,), "
            "dt_bias (144,), initial_state (B, 144, 64, 128) fp32; chunk_size = 256"),
    _op("dendritic_gated_norm_soma", _T + "dendritic_gated_norm", "y z theta gamma eps",
        "x (M, 9216) bf16 = bf16(S(n, theta) * gamma), n the bf16 gated RMSNorm (unit weight)", "prefill",
        doc="y (M, 9216) bf16, z (M, 9216) bf16 (strided view of the projection), theta / gamma (9216,) fp32"),
    _op("dendritic_step_core", _T + "dendritic_step",
        "proj conv_state ssm_state conv_w conv_b A dt_bias D theta gamma eps",
        "xs (B, 9216) bf16", "decode", mutates="conv_state ssm_state",
        doc="proj (B, 18832) bf16, conv_state (B, 9472, 3) fp32, ssm_state (B, 144, 64, 128) bf16|fp32 (stored in its "
            "own dtype), conv_w (9472, 4), conv_b, A (144,), dt_bias, D, theta (9216,), gamma (9216,)"),
    _op("dendritic_res_add", _T + "dendritic_residual", "h upd",
        "(T, B, 1536) fp32 contiguous = h + upd.float()", "prefill decode",
        doc="h (T, B, 1536) fp32 residual stream, upd (T, B, 1536) bf16 with unit last stride (the transposed view of "
            "the (B, T, 1536) W_out output)"),
    # ---------------------------------------------------------------------------------------------- memory norms
    _op("row_rmsnorm", _T + "hippo_norms", "x weight eps",
        "(T, B, C) fp32, row-major", "train",
        doc="x (T, B, C) fp32, weight (C,) fp32 or None (gain 1)"),
    _op("row_rmsnorm_tee", _T + "hippo_norms", "x weight eps fp32 bf16",
        "(x, y (T, B, C) fp32 or None, bf16 x [(T, B, C) bf16]): the residual stream handed on, the norm, its bf16 "
        "roundings", "train",
        doc="x (T, B, C) fp32, weight (C,) fp32 or None (gain 1), fp32: return the fp32 norm, bf16 in (0, 1, 2): the "
            "number of bf16 outputs (one per bf16 consumer)"),
    _op("rmsnorm_rows", _T + "hippo_norms", "x eps weight zero_centered out_dtype",
        "(M, N) in out_dtype (default x.dtype)", "prefill decode",
        doc="x (M, N) any float, weight (N,) fp32 or None, zero_centered: gain 1 + weight; N % 128 == 0, N <= 4096"),
    _op("rmsnorm_gate_rows", _T + "hippo_norms", "x eps weight gate",
        "(..., N) bf16 = bf16(sigmoid(gate) * zero-centred RMSNorm(x))", "prefill decode",
        doc="x (..., 1536) fp32, weight (1536,) fp32 (zero-centred), gate (..., 1536) bf16 output-gate logits"),
    _op("out_norm_gate", _T + "hippo_norms", "attn w gate_logits eps",
        "(T, B, 1536) fp32 = zero-centred RMSNorm(attn) * sigmoid(gate_logits)", "train",
        doc="attn (T, B, 1536) fp32, w (1536,) fp32 (zero-centred), gate_logits (T, B, 1536) bf16"),
    # ---------------------------------------------------------------------------------------------- memory rope
    _op("silu_rope", _T + "hippo_rope", "x cos sin",
        "(T, B, C) bf16 = rope(silu(x))", "train",
        doc="x (T, B, C) bf16, cos / sin (T, 768) bf16"),
    _op("rope_interleaved", _T + "hippo_rope", "x cos sin",
        "(T, B, C) bf16", "prefill",
        doc="x (T, B, C) bf16, cos / sin broadcastable bf16"),
    # ---------------------------------------------------------------------------------------------- memory residual glue
    _op("value_residual", _T + "hippo_residual", "v lam0 gate carry glu_add",
        "(v' (T, B, 1536) bf16, silu(v) (T, B, 1536) bf16 for hippocampal layer 0, else None)", "train",
        doc="v (T, B, 1536) bf16 before its SiLU, lam0 (1536,) fp32 static mix, gate (T, B, 6) bf16 gate logits, "
            "carry (T, B, 1536) bf16 or None (hippocampal layer 0), glu_add (T, B, 256) bf16"),
    _op("centre_rows", _T + "hippo_residual", "r",
        "(T, B, D) bf16 = r - mean_c(r)", "train",
        doc="r (T, B, D) bf16 output projection of the attention sub-block"),
    _op("read_gain_residual", _T + "hippo_residual", "h r f",
        "h' (T, B, D) fp32", "train prefill decode",
        doc="h (T, B, D) fp32, r (T, B, D) bf16, f (T, B) fp32 read gain"),
    _op("pkm_tail", _T + "hippo_residual", "h ffn y rho",
        "h_out (T, B, D) fp32", "train prefill decode",
        doc="h (T, B, D) fp32, ffn (T, B, D) bf16, y (T, B, D) bf16, rho (T, B) fp32 (no gradient)"),
    _op("novelty_read_gain", _T + "novelty_gain", "h z r pi w b v kappa eps",
        "(h' (T, B, D) fp32 in h's layout, c (T, B) fp32 novelty factor); backward: h, z, r, w, b, v", "train",
        doc="h, z (T, B, D) fp32, r (T, B, D) bf16, pi (T, B) fp32 copy confidence (no gradient), w / v (D,) fp32, "
            "b (1,) fp32, kappa = 4.0 read gain, eps = 1e-6"),
    # ---------------------------------------------------------------------------------------------- lexical read (train)
    _op("lex_read_train", _T + "lex_read_train", "h q_raw k_raw o q_gain k_gain beta eps",
        "(T, B, 1536) fp32 h + alpha * o in h's layout; backward: every tensor argument", "train",
        doc="h (T, B, 1536) fp32, q_raw / k_raw (T, B, 64) bf16, o (T, B, 1536) bf16, q_gain / k_gain (64,) fp32, "
            "beta (1,) fp32, eps = 1e-6"),
    # ---------------------------------------------------------------------------------------------- cumulative sums
    _op("cumsum0", _T + "cumsum", "x",
        "inclusive cumsum over dim 0, fp32", "train",
        doc="x (T, ...) fp32; the narrow / wide dispatch is internal and fixed"),
    _op("centred_past", _T + "cumsum", "x",
        "(T, B, C) bf16 contiguous: x minus the mean of the earlier tokens", "train",
        doc="x (T, B, C) fp32"),
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
    # ---------------------------------------------------------------------------------------------- pool attention (train)
    _op("pool_attend", "neuronspark.ops.pool_attend",
        "q k gate v gain wbias sparsity xq psi cos sin cum beta qtok ktok orders scale",
        "(heads (T, B, 1536) fp32, xreads (T, B, 1024) bf16, q_all (T, B, 10, 256) bf16, k_all (T, B, 6, 256) bf16)",
        "train",
        doc="q, k (T, B, 1536) bf16 after SiLU and RoPE, gate (T, B, 1536) fp32 write-gate spikes, v (T, B, 1536) "
            "bf16, gain (6,) bf16 match gain, wbias (6,) fp32, sparsity 2.0, xq 4 x (T, B, 256) bf16 extra-query "
            "projections, psi (T, B, 4) bf16, cos / sin (T, 128) bf16 last-head RoPE band, then the operands of "
            "pool_read: the head preparation, the tagged values, the extra queries and the read"),
    _op("shared_operand", _T + "shared_operand", "x uses",
        "uses x (T, B, 1536) bf16 row-major: the bf16 operand of the projections that read x", "train",
        doc="x (T, B, 1536) fp32 (any strides), uses the number of projections (1 .. 8); x's gradient is the fp32 "
            "sum of the uses' bf16 gradients"),
    _op("clock_increments", _T + "pool_clock", "logit rate gains base floor",
        "(T, B, 6) fp32 = gains mean_c log(max(exp(-rate softplus(logit)) base, floor))", "train",
        doc="logit (T, B, 256) bf16 decay projection, rate (256,) bf16, gains (6,) fp32 eligibility slopes, base the "
            "base decay, floor the decay floor"),
    _op("pool_read", "neuronspark.ops.pool_read", "q k gv cum beta qtok ktok orders scale",
        "(T, B, 10, 256) bf16: the 6 heads' reads, then the last head's 4 extra-query reads", "train",
        doc="q (T, B, 10, 256) fp32 gained unit queries (6 heads, 4 extra queries), k (T, B, 6, 256) fp32 unit keys, "
            "gv (T, B, 6, 256) bf16 tagged values, cum (T, B, 6) fp32 clocks, beta (3, 6) fp32 = pool_ind_beta, "
            "qtok / ktok (B, T, 4) int32 id windows, orders (2, 3, 4), scale 16.0"),
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
        doc="q (B, 6, 256), xq (B, 4, 256), cum_t this step's clocks, rings, counters, induction matches, beta, gain"),
    _op("xq_form_t1", _T + "pool_ring", "u wb cos sin psi head_gain pos",
        "(B, 4, 256) bf16", "decode",
        doc="u (B, 256) bf16, wb (4, 256, 64) bf16, cos / sin (B, 128) bf16 (with pos (B,) int64: the (positions, "
            "128) table band, read at pos), psi (B, 4) bf16, head_gain (1,) fp32"),
    # ---------------------------------------------------------------------------------------------- PLIF
    _op("plif_centred_input", _T + "plif", "x out_dtype",
        "out_dtype (default x.dtype) (T, B, C): x minus its running mean", "train",
        doc="x (T, B, C) fp32; out_dtype bf16 for the eta projection's operand"),
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
    _op("plif_prefill_pair", _T + "plif", "x0 w0 b0 zp0 th0 v0 x1 w1 b1 zp1 th1 v1 out_dtype",
        "spikes (2, T, B, C) out_dtype", "prefill", mutates="v0 v1",
        doc="two populations of one shape: x (T, B, C), w_eta, b_eta, v_th (C,) fp32, zp (T, B, C) bf16, v (B, C) fp32"),
    # ---------------------------------------------------------------------------------------------- PKM
    _op("pkm_unit_queries", _T + "pkm_query", "q heads half",
        "(T, B, 4, 2, 64) fp32 unit half-queries; backward: dq bf16", "train",
        doc="q (T, B, 512) bf16 query projection, heads = 4, half = 64"),
    _op("pkm_gained_keys", _T + "pkm_query", "subkeys log_gain",
        "(4, 2, 128, 64) fp32 unit sub-keys times sqrt(64) exp(log_gain); backward: dsubkeys, dlog_gain", "train",
        doc="subkeys (4, 2, 128, 64) fp32, log_gain (8,) fp32"),
    _op("pkm_topk_train", _T + "pkm_topk", "s k",
        "(sc fp32, u, ii, jj int64) each (T, B, 4, 16); backward -> ds", "train prefill",
        doc="s (T, B, 4, 2, 128) fp32 half-query scores, k = 16"),
    _op("pkm_value_read", _T + "pkm_read", "w u V",
        "(T, B, 768) fp32; backward: dw, dV", "train prefill",
        doc="w (T, B, 64) fp32 slot weights, u (T, B, 64) int64 slot ids, V (16384, 768) fp32"),
    _op("pkm_norm_gate", _T + "pkm_gate", "val gain glog eps",
        "(T, B, 768) bf16 = bf16(zero-centred RMSNorm(val) * sigmoid(glog)); backward: dval, dgain, dglog", "train prefill",
        doc="val (T, B, 768) fp32 value read, gain (768,) fp32 (zero-centred), glog (T, B, 768) bf16 gate logits"),
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
            "widths (16, 32); a stack of three or four power-of-two widths up to 512: (R, w) bf16 per width"),
    _op("balance_add", _T + "balance_add", "logits contribs widths",
        "(R, V) bf16 dense balanced logits", "train prefill decode",
        doc="logits (R, 151669) bf16 readout (unit column stride), contribs ((R, 16), (R, 32)) bf16 contiguous, "
            "widths (16, 32); one to four widths, (R, w) bf16 per width"),
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
# The hippocampal layers' prefill operators (copy-detector statistics, extra-query formation, lexical read,
# value-residual
# mix), registered into OPS after the main table.
_HIPPOCAMPAL_PREFILL_SPECS: List[OpSpec] = [
    _op("copy_stats_prefill", _T + "copy_stats", "ring ring_ptr pos T",
        "(maximum, lse) each (T, B) fp32 contiguous; -inf where a query has no valid slot", "prefill",
        doc="ring (B, W, 96) fp32 unit operands (the call's own written at (ring_ptr + t) % W), ring_ptr (1,) int64 "
            "pointer at the call's start, pos (B,) int64 tokens before the call, T the call's tokens"),
    _op("xq_queries_prefill", _T + "xq_query", "qs psi head_gain cos sin",
        "(B, 4, T, 256) bf16 gained unit extra queries (view of a contiguous (4, T, B, 256) buffer)", "prefill",
        doc="qs 4 x (T, B, 256) bf16 contiguous projections, psi (T, B, 4) fp32 clamped gain projection, head_gain "
            "one-element bf16 match gain of the last head, cos / sin (T, B, 128) bf16 last-head RoPE band"),
    _op("lex_read_prefill", _T + "lex_read", "h q_raw k_raw o q_gain k_gain beta eps",
        "(T, B, 1536) fp32 contiguous h + alpha * o", "prefill decode",
        doc="h (T, B, 1536) fp32, q_raw / k_raw (T, B, 64) bf16, o (T, B, 1536) bf16, q_gain / k_gain (64,) fp32, "
            "beta (1,) fp32, eps = 1e-6"),
    _op("vr_mix_prefill", _T + "vr_mix", "v carry gl lam0 glu_add",
        "(T, B, 1536) bf16 contiguous value stream after the value-residual mix and the GLU tail", "prefill",
        doc="v (T, B, 1536) bf16, carry (T, B, 1536) bf16 or None (hippocampal layer 0), gl (T, B, 6) bf16 gate "
            "projection, "
            "lam0 (1536,) fp32 contiguous static mix, glu_add (T, B, 256) fp32 GLU addend of the last head"),
    _op("copy_gate_prefill", _T + "copy_gate", "maximum lse zw rv tau bias strength kappa",
        "(f, rho): the read gain's factor and the PKM copy gate, (T, B) fp32 each", "prefill",
        doc="maximum / lse (T, B) fp32 the copy detector's statistics, zw / rv (T, B) fp32 the novelty predictor's "
            "products, tau / bias / strength one fp32 value each, kappa the read gain (4.0)"),
    _op("pool_clock_prefill", _T + "pool_clock_prefill", "decay_logit rate base_decay floor gains run cum_ring ptr",
        "(B, 6, T) fp32 contiguous clocks of the call's tokens", "prefill", mutates="run cum_ring",
        doc="decay_logit (T, B, 256) bf16 decay projection, rate (256,) bf16 decay rate, base_decay / floor the "
            "decay's factor and lower clamp, gains (6,) fp32 eligibility slopes, run (B, 6) fp32 carried clocks, "
            "cum_ring (B, 6, W) fp32 clock rings (written at (ptr + t) % W), ptr (1,) int64 ring pointer"),
]
for _hippocampal_spec in _HIPPOCAMPAL_PREFILL_SPECS:
    if _hippocampal_spec.name in OPS:
        raise RuntimeError(f"neuronspark.ops.registry: duplicate operator name {_hippocampal_spec.name!r}")
    OPS[_hippocampal_spec.name] = _hippocampal_spec
_SPECS.extend(_HIPPOCAMPAL_PREFILL_SPECS)
del _hippocampal_spec


# ================================================================================================== decode-step block
# The operators of the single-token step (one kernel per torch expression chain of the decode forms), registered into OPS
# after the main table.
_DECODE_STEP_SPECS: List[OpSpec] = [
    _op("plif_step", _T + "plif_step", "x eta_V eta_U w_eta b_eta v_th v out_dtype run_sum count",
        "spikes S (B, C) out_dtype", "decode", mutates="v run_sum",
        doc="x (B, C) fp32|bf16, eta_V (rank, C) / eta_U (C, rank) fp32 eta projection, w_eta / b_eta / v_th (C,) "
            "fp32, v (B, C) fp32 carried membrane; the centred neuron: run_sum (B, C) fp32 carried input sum, count "
            "(B,) int64 tokens before this one"),
    _op("centred_step", _T + "plif_step", "x run_sum count out_dtype",
        "(B, C) in out_dtype", "decode", mutates="run_sum",
        doc="x (B, C), run_sum (B, C) fp32 carried sum, count (B,) int64 tokens before this one"),
    _op("ring_write_step", _T + "ring_step",
        "qkv conv_hist conv_weight cos sin pos ring_ptr gate gl lam0 carry ga gb glu_u dlog rate base_decay gains "
        "qgain wbias sparsity pool_k pool_gv pool_cum pool_run",
        "(q (B, 6, 256) bf16 gained unit queries, v (B, 1536) bf16 values after the SiLU)", "decode",
        mutates="conv_hist pool_k pool_gv pool_cum pool_run",
        doc="qkv (B, 4608) bf16 projection, conv_hist (B, 4608, 3) fp32, conv_weight (4608, 1, 4), cos / sin (8192, 768) "
            "bf16 tables, pos (B,) int64, ring_ptr (1,) int64, gate (B, 1536) fp32 spikes, gl (B, 6) bf16, lam0 (1536,), "
            "carry (B, 1536) bf16 or None, ga / gb (B, 128) bf16, glu_u (256, 128), dlog (B, 256) bf16, rate (256,) "
            "bf16, base_decay, gains / qgain / wbias (6,), sparsity, the rings and the carried clocks"),
    _op("read_gain_step", _T + "hippo_step",
        "h z r cmax clse sink strength nov_w nov_v nov_b norm_weight kappa scale eps_r eps_n",
        "(h' (B, 1536) fp32, x_hat (B, 1536) fp32, x_hat (B, 1536) bf16, rho (B,) fp32)", "decode",
        doc="h, z (B, 1536) fp32, r (B, 1536) bf16 output projection, cmax / clse (B,) fp32, sink / strength (1,), "
            "nov_w / nov_v (1536,), nov_b (1,), norm_weight (1536,), kappa 4.0, scale 32.0, eps 1e-6 / 1e-6"),
    _op("pkm_tail_step", _T + "hippo_step", "h ffn y rho",
        "(B, 1536) fp32", "decode",
        doc="h (B, 1536) fp32, ffn / y (B, 1536) bf16, rho (B,) fp32"),
    _op("induction_step", _T + "induction_step", "ids win keys ring_ptr pad",
        "(4, B, W) uint8: entry k - 1 the order-k match of every ring slot", "decode", mutates="win keys",
        doc="ids (B,) int64, win (B, 4) int64 the rows' last ids, keys (B, W, 4) int64 ring, ring_ptr (1,) int64, "
            "pad 151669"),
    _op("induction_match_prefill", _T + "induction_prefill", "keys qwin ring_ptr pad",
        "(4, B, T, W) uint8: entry k - 1 the order-k match of every query against every ring slot", "prefill",
        doc="keys (B, W, 4) int64 ring with the call's keys written, qwin (B, T, 4) int64 the queries' last ids, "
            "ring_ptr (1,) int64 the pointer at the start of the call, pad 151669"),
    _op("lexical_step", _T + "lex_step", "ids win tables orders hash_mult hash_add hash_wrap vocab_size scale",
        "(B, 160) fp32 lexical vector of one token", "decode", mutates="win",
        doc="ids (B,) int64, win (B, 2) int64 the rows' last ids, tables 5 x (rows, d) fp32, orders (1, 2, 2, 3, 3), "
            "hash_mult / hash_add / hash_wrap (4,) int64, vocab_size 151669, scale 50.0"),
    _op("pkm_scores_step", _T + "pkm_step", "q kh",
        "(B, 4, 2, 128) fp32", "decode",
        doc="q (B, 512) bf16 query projection, kh (4, 2, 128, 64) fp32 unit sub-keys"),
    _op("pkm_value_step", _T + "pkm_step", "w u values gain glog eps",
        "(B, 768) bf16", "decode",
        doc="w (B, 64) fp32, u (B, 64) int64, values (16384, 768) fp32, gain (768,) fp32, glog (B, 768) bf16, eps 1e-6"),
    _op("copy_stats_step", _T + "copy_step", "z ring ring_ptr pos",
        "(maximum, lse) each (1, B) fp32; -inf where the row has no valid slot", "decode", mutates="ring",
        doc="z (B, 96) fp32 order-1 slice of the token's lexical vector, ring (B, W, 96) fp32 unit operands (the "
            "token's written at ring_ptr % W), ring_ptr (1,) int64, pos (B,) int64 tokens before this one"),
    _op("attn_out_step", _T + "attn_out_step", "red xq_add gate weight eps",
        "(B, 1536) bf16 gated output norm (the output projection's operand)", "decode",
        doc="red (B, 1536 + extra-query width) bf16 ring read, xq_add (B, 256) bf16, gate (B, 1536) bf16 logits, "
            "weight (1536,) fp32 zero-centred norm weight, eps"),
    _op("lex_read_step", _T + "lex_read_step", "h q_raw k_raw o q_gain k_gain beta eps eps_z",
        "(h' (B, 1536) fp32, h' bf16, z (B, 1536) fp32 frozen norm of h', z bf16)", "decode",
        doc="h (B, 1536) fp32, q_raw / k_raw (B, 64) bf16, o (B, 1536) bf16, q_gain / k_gain (64,) fp32, beta (1,) "
            "fp32, eps = 1e-6, eps_z the frozen norm's eps"),
]
for _decode_spec in _DECODE_STEP_SPECS:
    if _decode_spec.name in OPS:
        raise RuntimeError(f"neuronspark.ops.registry: duplicate operator name {_decode_spec.name!r}")
    OPS[_decode_spec.name] = _decode_spec
_SPECS.extend(_DECODE_STEP_SPECS)
del _decode_spec


# ================================================================================================== tensor-parallel block
# The exchanges of the tensor-parallel inference on two GPUs (neuronspark.utils.tensor_parallel, which calls the kernels
# themselves: a reference cannot wait for the other GPU), registered into OPS after the main table.  ``send`` is the
# peer's mailbox region and ``recv`` this GPU's (int64 words: the fp32 bits and a flag); ``flag_peer`` / ``flag`` the
# peer's and this GPU's flag word for a site and ``epoch`` the call's epoch (int64 (1,) each).  The references make the
# exchange in torch on the words already in place (tp_exchange.publish_ref, take_ref) and refuse a word a kernel would
# wait for.  One GPU calls them in loopback (``send`` is ``recv``, ``flag_peer`` is ``flag``) or with the peer's words
# placed in ``recv`` before the call.  Shapes: one GPU's half of the 1.3B model's dendritic neuron layer (72 of 144
# heads, 4,608 of 9,216 channels) and the model's width 1,536.
_TENSOR_PARALLEL_SPECS: List[OpSpec] = [
    _op("tp_step_core", _T + "tp_dendritic",
        "proj conv_state ssm_state conv_w conv_b A dt_bias D theta gamma eps n_all send recv",
        "xs (B, 4608) bf16: the soma read-out of this GPU's heads, normalized over n_all channels", "decode",
        mutates="conv_state ssm_state send recv",
        doc="proj (B, 9544) bf16 local projection [z | x | B | C | dt], conv_state (B, 4864, 3) fp32, ssm_state (B, 72, "
            "64, 128) bf16|fp32, conv_w (4864, 4) / conv_b (4864,) / A / dt_bias / D (72,) / theta / gamma (4608,) "
            "fp32, eps 1e-5, n_all 9216, send / recv k2_words(B, 4608) words (one per row and K2 program)"),
    _op("tp_join", _T + "tp_exchange", "partial h send recv gain eps",
        "(h_new (B, 1536) fp32 = h + f32(bf16(partial + partial_peer)), hn (B, 1536) bf16 pre-norm of h_new or None)",
        "decode", mutates="send recv",
        doc="partial (B, 1536) fp32 out-projection partial, h (B, 1536) fp32 residual stream, send / recv B x 1536 "
            "words, gain (1536,) fp32 the next dendritic layer's pre-norm gain or None, eps 1e-5"),
    _op("tp_tail_gather", _T + "tp_exchange", "h ffn y rho lo batch send recv gain eps",
        "(h_out (batch, 1536) fp32 every row: this GPU's the tail of its rows, the others the peer's; hn (batch, "
        "1536) bf16 or None)", "decode", mutates="send recv",
        doc="h (n, 1536) fp32, ffn / y (n, 1536) bf16, rho (n,) fp32 (this GPU's rows [lo, lo + n) of the batch), "
            "send / recv batch x 1536 words, gain (1536,) fp32 or None, eps 1e-5"),
    _op("tp_greedy_pick", _T + "tp_exchange", "value index offset send recv",
        "(B,) int64 greedy tokens across the two vocabulary shards", "prefill decode", mutates="send recv",
        doc="value (B,) bf16 this shard's per-row maximum, index (B,) int64 its shard-local id, offset the shard's first "
            "vocabulary row (0 or 75808), send / recv 2 B words"),
    _op("tp_gated_norm_soma", _T + "tp_dendritic", "y z theta gamma eps n_all send recv rstd_out n_out",
        "x (M, 4608) bf16 contiguous soma read-out, normalized over n_all channels", "prefill",
        mutates="send recv rstd_out n_out",
        doc="y (M, 4608) bf16, z (M, 4608) bf16 view with unit last stride, theta / gamma (4608,) fp32, eps 1e-5, n_all "
            "9216, send / recv M words, rstd_out (M,) fp32 or None, n_out (M, 4608) bf16 or None"),
    _op("tp_fence", _T + "tp_bulk", "flag_peer flag epoch",
        "None: the epoch stored into the peer's call word, then a wait until this GPU's holds at least it", "prefill",
        mutates="flag_peer", doc="flag_peer / flag / epoch int64 (1,)"),
    _op("tp_release", _T + "tp_bulk", "flag_peer epoch",
        "None: the epoch stored into the peer's flag word of the site", "prefill", mutates="flag_peer",
        doc="flag_peer / epoch int64 (1,)"),
    _op("tp_prefill_join", _T + "tp_bulk", "part part_peer h flag epoch",
        "(T, B, 1536) fp32 contiguous h + f32(bf16(part + part_peer))", "prefill",
        doc="part / part_peer (B T, 1536) fp32 contiguous out-projection partials in (b, t) row order (part_peer read "
            "through the peer's mapping), h (T, B, 1536) fp32 with unit last stride, flag / epoch int64 (1,)"),
    _op("tp_prefill_gather", _T + "tp_bulk", "x part_peer flag epoch lo batch",
        "(T, batch, 1536) fp32 contiguous every row: this GPU's rows [lo, lo + n) and the peer's", "prefill",
        doc="x (T, n, 1536) fp32 with unit last stride, part_peer (>= T (batch - n), 1536) fp32 contiguous the peer's "
            "rows in (t, row) order, flag / epoch int64 (1,), lo 0 or batch - n"),
]
for _tp_spec in _TENSOR_PARALLEL_SPECS:
    if _tp_spec.name in OPS:
        raise RuntimeError(f"neuronspark.ops.registry: duplicate operator name {_tp_spec.name!r}")
    OPS[_tp_spec.name] = _tp_spec
_SPECS.extend(_TENSOR_PARALLEL_SPECS)
del _tp_spec
