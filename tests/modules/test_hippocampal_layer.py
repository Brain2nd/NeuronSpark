"""modules.hippocampal_layer: the hippocampal layer in its three regimes.

* layout: the 70 parameters and 11 persistent buffers of the NeuronSpark checkpoint (keys, shapes, element counts), the
  non-persistent RoPE tables, the launch values (spectra, anchors, zero-launched parameters, PLIF b_eta, identity
  short conv), an empty build drawing nothing, the uncalibrated-PKM refusal;
* checkpoint: every hippocampal layer of the checkpoint loads strictly (prefix ``backbone.layers.<i>.`` stripped),
  marks its PKM ready and holds the config-derived value-residual anchor;
* the sub-blocks on the kernel operators against the same sub-blocks on the reference operators (attention sub-block,
  FFN + PKM read on a shared input), forward and backward; the whole layer is reported;
* inference: a prefill split into two calls against one prefill (output and every carried state), the two decode
  routes (ring kernel / torch ring chain) against each other, a CUDA-graph replay of a decode step against the eager
  step; decode steps after a prefill and a prefill against the whole row are reported;
* the auxiliary-loss operands and the launch-calibration capture protocol.

Every assertion on computed values is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import math

import pytest
import torch

import neuronspark.modules.pool_attention as PA
from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.copy_detector import copy_stats_session, copy_stats_whole_row
from neuronspark.modules.induction import induction_match_matrices, induction_session_match
from neuronspark.modules.hippocampal_layer import HippocampalLayer
from neuronspark.modules.plif import W_CLAMP_MAX
from neuronspark.modules.pool_attention import value_residual_anchor
from neuronspark.utils.cache import CopyState, InductionState, InferenceParams, LexicalState
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import (assert_all_match, assert_match, compare_outputs, match_stats, rand_tensor,
                                       summarize)

pytestmark = pytest.mark.cuda

CFG = NeuronSparkConfig()
V = CFG.vocab_size
D = CFG.D
LEX = CFG.lexical_dim
# (layer index, rank among the hippocampal layers): the value-residual source and one later layer
LAYERS = [(3, 0), (7, 1)]
LAYER_IDS = [f"layer{l}" for l, _ in LAYERS]
W_RING = 2048
SMALL_VOCAB = 64                                       # token ids drawn from few ids: repeats exercise copy / induction


def _plif_keys(prefix, C):
    return {f"{prefix}.w": (C,), f"{prefix}.v_th": (C,), f"{prefix}.w_eta": (C,), f"{prefix}.b_eta": (C,),
            f"{prefix}.eta_V": (16, C), f"{prefix}.eta_U": (C, 16)}


EXPECTED_PARAMS = {
    "W_decay.weight": (256, 1536), "decay_A_log_delta": (256,),
    "pool_bias_gain_delta": (6,), "pool_dendritic_gain_delta": (6,), "pool_write_bias": (6,),
    "pool_glu_a": (128, 1536), "pool_glu_b": (128, 1536), "pool_glu_u": (256, 128),
    "pool_value_residual": (1536,), "pool_vr_gate": (6, 1536),
    "pool_xq_a": (256, 1536), **{f"pool_xq_b.{m}": (256, 64) for m in range(4)}, "pool_xq_gain_proj": (4, 1536),
    "pool_xq_out_in": (64, 1024), "pool_xq_out": (256, 64),
    "qkv_proj.weight": (4608, 1536), **_plif_keys("gate_neuron", 1536), "qkv_conv.weight": (4608, 1, 4),
    "attn_out_norm.weight": (1536,), "attn_out_gate.weight": (1536, 1536), "attn_out_proj.weight": (1536, 1536),
    "ffn_norm.weight": (1536,), **_plif_keys("input_neuron2", 1536),
    "snn_ffn.gate_proj.weight": (768, 1536), "snn_ffn.up_proj.weight": (768, 1536),
    "snn_ffn.down_proj.weight": (1536, 768), "snn_ffn.skip_proj.weight": (1536, 1536),
    **_plif_keys("snn_ffn.gate_neuron", 768), **_plif_keys("snn_ffn.up_neuron", 768),
    "ffn_out_proj.weight": (1536, 1536),
    "lex_q.weight": (64, 1536), "lex_k.weight": (64, 160), "lex_o.weight": (1536, 160),
    "lex_q_gain": (64,), "lex_k_gain": (64,), "lex_beta": (1,),
    "copy_novelty_w": (1536,), "copy_novelty_b": (1,), "copy_novelty_v": (1536,), "pool_ind_beta": (18,),
    "pkm_q.weight": (512, 1536), "pkm_g.weight": (768, 1536), "pkm_o.weight": (1536, 768),
    "pkm_values": (16384, 768), "pkm_subkeys": (4, 2, 128, 64), "pkm_norm_gain": (768,), "pkm_log_gain": (8,),
}
EXPECTED_BUFFERS = {
    "decay_A_log_init": (256,), "pool_bias_gain_spectrum_raw": (6,), "pool_dendritic_gain_spectrum_log": (6,),
    "pool_value_residual_anchor": (1536,), "pool_write_bias_spectrum": (6,), "pkm_copy_strength": (),
    "pkm_copy_tau": (), "pkm_calibrated": (), "pkm_launch_rms_res": (), "pkm_launch_rms_y": (),
    "pkm_launch_entropy": (),
}
ZERO_LAUNCHED = ("W_decay.weight", "decay_A_log_delta", "pool_bias_gain_delta", "pool_dendritic_gain_delta",
                 "pool_write_bias", "pool_glu_u", "pool_value_residual", "pool_vr_gate", "pool_xq_gain_proj",
                 "pool_xq_out", "attn_out_norm.weight", "snn_ffn.skip_proj.weight", "lex_q_gain", "lex_k_gain",
                 "copy_novelty_w", "copy_novelty_v", "pool_ind_beta", "pkm_norm_gain", "pkm_log_gain")


# ------------------------------------------------------------------------------------------------ helpers
def _layer(layer_idx, device, impl="kernel", init="launch"):
    return HippocampalLayer(CFG, layer_idx, device=device, impl=impl, init=init)


def _neurons(layer):
    return (layer.gate_neuron, layer.input_neuron2, layer.snn_ffn.gate_neuron, layer.snn_ffn.up_neuron)


def _exercised_state(layer_idx, device):
    """A launch layer moved off its zero-launched values (every term of the layer runs, as in a trained layer) with
    the PKM marked calibrated; returns its state dict."""
    layer = _layer(layer_idx, device)
    scales = {"W_decay.weight": 0.02, "decay_A_log_delta": 0.1, "pool_bias_gain_delta": 0.1,
              "pool_dendritic_gain_delta": 0.1, "pool_write_bias": 0.3, "pool_glu_u": 0.05,
              "pool_value_residual": 0.1, "pool_vr_gate": 0.02, "pool_xq_gain_proj": 0.02, "pool_xq_out": 0.05,
              "attn_out_norm.weight": 0.1, "snn_ffn.skip_proj.weight": 0.01, "lex_q_gain": 0.1, "lex_k_gain": 0.1,
              "copy_novelty_w": 0.02, "copy_novelty_v": 0.02, "pool_ind_beta": 0.5, "pkm_norm_gain": 0.1,
              "pkm_log_gain": 0.1}
    with torch.no_grad():
        for name, s in scales.items():
            p = layer.get_parameter(name)
            p.add_(s * torch.randn_like(p))
        for n in _neurons(layer):
            n.w_eta.add_(0.05 * torch.randn_like(n.w_eta))
            n.b_eta.add_(0.3 * torch.randn_like(n.b_eta))
            n.eta_U.add_(0.02 * torch.randn_like(n.eta_U))
        layer.pkm_calibrated.fill_(1.0)
        layer.pkm_copy_tau.fill_(0.9)
    return {k: v.detach().clone() for k, v in layer.state_dict().items()}


def _loaded(layer_idx, state, device, impl="kernel"):
    layer = _layer(layer_idx, device, impl=impl, init="empty")
    layer.load_state_dict(state, strict=True)
    assert layer.pkm_ready
    return layer


def _row_inputs(T, B, device, hippocampal_index, lex_table=None):
    """Token ids with repeats, the lexical vector of a fixed id table (a function of the token: the whole row and an
    inference call see the same vector), the B-major residual stream, the carry of a later layer."""
    ids = torch.randint(0, SMALL_VOCAB, (B, T), device=device)
    if lex_table is None:
        lex_table = rand_tensor((SMALL_VOCAB, LEX), torch.float32, device)
    h = rand_tensor((B, T, D), torch.float32, device).transpose(0, 1)
    carry = None if hippocampal_index == 0 else (0.5 * rand_tensor((T, B, CFG.D_value), torch.float32, device)).to(
        torch.bfloat16)
    return ids, lex_table, h, carry


def _lex(lex_table, ids):
    return lex_table[ids].transpose(0, 1).contiguous()                  # (T, B, 160) fp32


def _whole_row(layer, ids, lex_table, h, carry, g_out=None, g_carry=None, aux_out=None, calib=None):
    """Whole-row call (training regime); with ``g_out`` also the backward.  Returns (outputs, grads)."""
    lex = _lex(lex_table, ids)
    hr = h.detach().clone().requires_grad_(g_out is not None)
    cr = None if carry is None else carry.detach().clone().requires_grad_(g_out is not None)
    with regime("train"):
        ind = induction_match_matrices(ids, CFG.hippocampal_induction_bias_orders, pad=V)
        copy = copy_stats_whole_row(lex)
        out, carry_out = layer(hr, lex=lex, ind=ind, copy=copy, carry=cr, aux_out=aux_out, calib=calib)
    outs = {"h": out.detach(), "carry": carry_out.detach()}
    grads = {}
    if g_out is not None:
        roots, cots = [out], [g_out]
        if carry is None and g_carry is not None:
            roots.append(carry_out)
            cots.append(g_carry)
        torch.autograd.backward(roots, cots)
        grads["h"] = hr.grad.detach().clone()
        if cr is not None:
            grads["carry"] = cr.grad.detach().clone()
        for name, p in layer.named_parameters():
            if p.grad is not None:
                grads[name] = p.grad.detach().clone()
    return outs, grads


def _cache(layer, B, device, W=W_RING):
    st = layer.allocate_inference_cache(B, W)
    pos, ring_ptr = InferenceParams.allocate_counters(B, device)
    return InferenceParams(max_seqlen=W, max_batch_size=B, pos=pos, ring_ptr=ring_ptr,
                           dendritic_state_dtype=torch.bfloat16,
                           layers={layer.layer_idx: st},
                           lexical=LexicalState.allocate(B, 2, V, device),
                           induction=InductionState.allocate(B, W, max(CFG.hippocampal_induction_bias_orders), V,
                                                             device),
                           copy=CopyState.allocate(B, W, 96, device), announce=False)


def _infer_call(layer, ip, ids, lex_table, h, carry):
    """One inference call of the layer with the model-level session pieces of the call (induction keys ring, copy
    ring) and the counter advance the model performs after its last layer."""
    T = int(ids.shape[1])
    W = int(ip.max_seqlen)
    lex = _lex(lex_table, ids)
    with torch.no_grad(), regime("infer"):
        ind = induction_session_match(ids, ip.induction, ip.ring_ptr, W, CFG.hippocampal_induction_bias_orders)
        copy = copy_stats_session(lex, ip.copy, ip.ring_ptr, ip.pos, W)
        out, carry_out = layer(h, lex=lex, ind=ind, copy=copy, carry=carry, inference_params=ip)
        ip.pos.add_(T)
        ip.ring_ptr.add_(T)
    ip.seqlen_offset += T
    return out, carry_out


def _state_tensors(ip):
    """Every carried state tensor of the cache, by name.  ``pool_cum`` keeps +inf in the ring slots that hold no event
    (the cache's empty-slot value, ``neuronspark.utils.cache``; never a computed value): it enters as two tensors, the
    written clocks (the empty slots read as 0) and the empty-slot indicator (1.0 = empty), so the occupancy of the
    rings and every written clock are both compared by the rule.  A -inf or nan among the written clocks stays in the
    first tensor (the rule rejects it)."""
    out = {}
    for name, t in ip.tensors():
        if name.endswith(".pool_cum"):
            empty = t == float("inf")
            out[name] = torch.where(empty, torch.zeros_like(t), t)
            out[name + ".empty"] = empty.to(torch.float32)
        else:
            out[name] = t.clone()
    return out


def _rows(h):
    return h.contiguous()


# ------------------------------------------------------------------------------------------------ layout
@pytest.mark.parametrize("layer_idx,hippocampal_index", LAYERS, ids=LAYER_IDS)
def test_state_dict_layout(device, layer_idx, hippocampal_index):
    layer = _layer(layer_idx, device, init="empty")
    params = dict(layer.named_parameters())
    sd = layer.state_dict()
    buffers = {k: v for k, v in sd.items() if k not in params}
    assert {k: tuple(v.shape) for k, v in params.items()} == EXPECTED_PARAMS
    assert {k: tuple(v.shape) for k, v in buffers.items()} == EXPECTED_BUFFERS
    assert len(params) == 70 and len(buffers) == 11
    assert sum(p.numel() for p in params.values()) == CFG.hippocampal_layer_params
    assert sum(b.numel() for b in buffers.values()) == 256 + 3 * 6 + 1536 + 6
    assert all(p.dtype == torch.float32 for p in params.values())
    assert tuple(layer.rope_cos.shape) == (8192, 768) and "rope_cos" not in sd and "rope_sin" not in sd
    assert layer.hippocampal_index == hippocampal_index and layer.value_residual_first == (hippocampal_index == 0)


@pytest.mark.parametrize("layer_idx,hippocampal_index", [(3, 0), (15, 3)], ids=["layer3", "layer15"])
def test_launch_values(device, layer_idx, hippocampal_index):
    layer = _layer(layer_idx, device)
    for name in ZERO_LAUNCHED:
        p = layer.get_parameter(name).detach()
        assert_match(p, torch.zeros_like(p), f"{name}.launch")
    assert_match(layer.copy_novelty_b.detach(), torch.full((1,), 4.0, device=device), "copy_novelty_b")
    assert_match(layer.lex_beta.detach(), torch.full((1,), -2.0, device=device), "lex_beta")
    # spectra: eligibility gains 1 -> 0.03, match-gain factors 1 -> 0.37, write bias 0 -> 6, decay half-lives 1 -> 512
    gains = torch.exp(torch.linspace(0.0, math.log(0.03), 6, dtype=torch.float64)).to(torch.float32)
    assert_match(layer.pool_bias_gains(), gains.to(device), "bias_gain_launch")
    factors = torch.exp(torch.linspace(0.0, math.log(0.37), 6, dtype=torch.float64)).to(torch.float32)
    assert_match(layer.pool_query_gain(torch.float32), factors.to(device), "match_gain_launch")
    assert_match(layer.pool_write_bias_total(), torch.linspace(0.0, 6.0, 6, device=device), "write_bias_launch")
    assert_match(layer.decay_A_log_init, -torch.linspace(0.0, math.log(512.0), 256, device=device), "decay_A_log_init")
    # value-residual anchor
    if hippocampal_index == 0:
        expect = torch.zeros(CFG.D_value, device=device)
    else:
        a, b = CFG.hippocampal_pool_value_residual_anchor[hippocampal_index - 1]
        expect = torch.cat([torch.full((256,), a * (b / a) ** (h / 5), device=device) for h in range(6)])
    assert_match(layer.pool_value_residual_anchor, expect, "anchor")
    assert_match(value_residual_anchor((0.5, 2.0), 1536, 6, device),
                 torch.tensor([0.5, 0.6598, 0.8706, 1.1487, 1.5157, 2.0], device=device).repeat_interleave(256),
                 "anchor_values")
    # identity short conv, PLIF b_eta from w, PKM buffers
    w = layer.qkv_conv.weight.detach()
    ident = torch.zeros_like(w)
    ident[:, :, -1] = 1.0
    assert_match(w, ident, "qkv_conv_identity")
    for n in _neurons(layer):
        beta = torch.sigmoid(torch.clamp(n.w.detach(), max=W_CLAMP_MAX))
        omb = (1.0 - beta).clamp(1e-6, 1 - 1e-6)
        assert_match(n.b_eta.detach(), torch.log(omb / (1 - omb)), "b_eta_launch")
    assert float(layer.pkm_copy_strength) == 4.0 and float(layer.pkm_copy_tau) == 1.0
    assert float(layer.pkm_calibrated) == 0.0 and not layer.pkm_ready


def test_launch_is_seeded_and_empty_build_draws_nothing(device):
    torch.manual_seed(7)
    a = {k: v.detach().clone() for k, v in _layer(3, device).state_dict().items()}
    torch.manual_seed(7)
    b = {k: v.detach().clone() for k, v in _layer(3, device).state_dict().items()}
    assert_all_match(b, a, prefix="same_seed")
    torch.manual_seed(11)
    expected = torch.rand(256, device=device)
    torch.manual_seed(11)
    _layer(3, device, init="empty")
    assert_match(torch.rand(256, device=device), expected, "empty_build_stream")


def test_uncalibrated_pkm_refuses(device):
    layer = _layer(3, device)
    ids, table, h, carry = _row_inputs(64, 2, device, 0)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _whole_row(layer, ids, table, h, carry)


def test_forward_requires_the_regime_autocast(device):
    layer = _loaded(3, _exercised_state(3, device), device)
    ids, table, h, _ = _row_inputs(64, 2, device, 0)
    lex = _lex(table, ids)
    ind = induction_match_matrices(ids, CFG.hippocampal_induction_bias_orders, pad=V)
    copy = copy_stats_whole_row(lex)
    with pytest.raises(RuntimeError), torch.no_grad():
        layer(h, lex=lex, ind=ind, copy=copy, carry=None)


# ------------------------------------------------------------------------------------------------ checkpoint
def _ckpt_layer_state(ckpt_dir, layer_idx):
    from safetensors import safe_open

    path = ckpt_dir / "model.safetensors"
    prefix = f"backbone.layers.{layer_idx}."
    with safe_open(str(path), framework="pt", device="cpu") as f:
        out = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if not out:
        raise AssertionError(f"layer {layer_idx} not found in {path}")
    return out


@pytest.mark.ckpt
@pytest.mark.parametrize("layer_idx", CFG.hippocampal_layer_ids)
def test_checkpoint_layer_loads_strictly(ckpt_dir, device, layer_idx):
    state = _ckpt_layer_state(ckpt_dir, layer_idx)
    layer = _layer(layer_idx, device, init="empty")
    assert set(state) == set(layer.state_dict())
    layer.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    assert layer.pkm_ready
    fresh = _layer(layer_idx, device, init="empty")
    assert_match(layer.pool_value_residual_anchor, fresh.pool_value_residual_anchor, "anchor_vs_config")
    for k, v in state.items():
        assert_match(layer.state_dict()[k].cpu(), v, f"loaded.{k}")


# ------------------------------------------------------------------------------------------------ kernel vs reference
@pytest.mark.sm100
@pytest.mark.parametrize("layer_idx,hippocampal_index", LAYERS, ids=LAYER_IDS)
def test_attention_subblock_vs_reference_operators(device, layer_idx, hippocampal_index):
    """Steps 1-12 (lexical read, frozen norm, attention sub-block) on the kernel and on the reference operators,
    forward (residual, carry) and backward (every parameter of the sub-block, the input, the carry)."""
    T, B = 256, 2
    state = _exercised_state(layer_idx, device)
    ids, table, h, carry = _row_inputs(T, B, device, hippocampal_index)
    g_r = rand_tensor((T, B, D), torch.bfloat16, device)

    def run(impl):
        layer = _loaded(layer_idx, state, device, impl)
        lex = _lex(table, ids)
        hr = h.detach().clone().requires_grad_(True)
        cr = None if carry is None else carry.detach().clone().requires_grad_(True)
        with regime("train"):
            ind = induction_match_matrices(ids, CFG.hippocampal_induction_bias_orders, pad=V)
            h1 = layer._lexical_read_train(hr, lex)
            z = layer._ops.row_rmsnorm(h1, None, layer.norm_eps)
            r, c_out = layer._attention_train(h1, z, ind, cr)
        r.backward(g_r)
        out = {"r": r.detach(), "carry": c_out.detach(), "grad.h": hr.grad.detach().clone()}
        if cr is not None:
            out["grad.carry"] = cr.grad.detach().clone()
        for name, p in layer.named_parameters():
            if p.grad is not None:
                out["grad." + name] = p.grad.detach().clone()
        return out

    new, ref = run("kernel"), run("ref")
    # the forward and the input / carry gradients by the rule; the parameter gradients are sums over every position of
    # products that inherit each upstream operator's summation-order differences (the per-operator gradients are compared
    # in tests/ops), so they are reported here rather than asserted at matched ratio 1.0
    hard = [k for k in new if not k.startswith("grad.") or k in ("grad.h", "grad.carry")]
    assert_all_match({k: new[k] for k in hard}, {k: ref[k] for k in hard}, prefix=f"layer{layer_idx}.attention")
    soft = [match_stats(new[k], ref[k], f"layer{layer_idx}.attention.{k}") for k in new if k not in hard]
    print(f"[report] layer{layer_idx} attention parameter gradients vs the reference operators: {summarize(soft)}")


class _SharedUpstreamOps:
    """The reference operators except ``linear`` and ``row_rmsnorm`` (the kernels): the PKM half-queries, hence the
    top-k selection, are then the same on both sides of a comparison (the selection is discontinuous at near-ties)."""

    _KERNEL = ("linear", "row_rmsnorm")

    def __getattr__(self, name):
        from neuronspark import ops
        return getattr(ops.KERNELS if name in self._KERNEL else ops.REFS, name)


@pytest.mark.sm100
def test_ffn_and_pkm_vs_reference_operators(device):
    """Steps 16-18 on a shared FFN input: input_neuron2 -> spiking FFN -> ffn_out_proj and the PKM read, kernel vs
    reference operators (the layer's projections and norms shared, see ``_SharedUpstreamOps``), forward and
    backward."""
    T, B = 256, 2
    state = _exercised_state(3, device)
    h = rand_tensor((B, T, D), torch.float32, device).transpose(0, 1)
    r = rand_tensor((T, B, D), torch.bfloat16, device)
    g_ffn = rand_tensor((T, B, D), torch.bfloat16, device)
    g_y = rand_tensor((T, B, D), torch.bfloat16, device)

    def run(impl):
        layer = _loaded(3, state, device, impl)
        if impl == "ref":
            layer._ops = _SharedUpstreamOps()
        hr = h.detach().clone().requires_grad_(True)
        with regime("train"):
            x_hat = layer._ops.row_rmsnorm(hr, layer.ffn_norm.weight, layer.ffn_norm.eps)
            ffn = layer._ops.linear(layer.snn_ffn(layer.input_neuron2(x_hat)), layer.ffn_out_proj.weight)
            y = layer._pkm_read_train(x_hat, r)
        torch.autograd.backward([ffn, y], [g_ffn, g_y])
        out = {"ffn": ffn.detach(), "y": y.detach(), "grad.h": hr.grad.detach().clone()}
        for name, p in layer.named_parameters():
            if p.grad is not None:
                out["grad." + name] = p.grad.detach().clone()
        return out

    assert_all_match(run("kernel"), run("ref"), prefix="ffn_pkm")


@pytest.mark.sm100
@pytest.mark.parametrize("layer_idx,hippocampal_index", LAYERS, ids=LAYER_IDS)
def test_layer_vs_reference_operators_report(device, layer_idx, hippocampal_index):
    """The whole layer on the kernel and on the reference operators (reported: a PKM top-k near-tie upstream of the
    PKM read can select another slot on the two sides; the sub-block tests above assert)."""
    T, B = 256, 2
    state = _exercised_state(layer_idx, device)
    ids, table, h, carry = _row_inputs(T, B, device, hippocampal_index)
    g_out = rand_tensor((T, B, D), torch.float32, device)
    sides = {}
    for impl in ("kernel", "ref"):
        layer = _loaded(layer_idx, state, device, impl)
        outs, grads = _whole_row(layer, ids, table, h, carry, g_out=g_out)
        sides[impl] = {**outs, **{"grad." + k: v for k, v in grads.items()}}
    res = compare_outputs(sides["kernel"], sides["ref"], prefix=f"layer{layer_idx}")
    for r in res:
        print(r)
    print(summarize(res))
    assert all(bool(torch.isfinite(v).all()) for v in sides["kernel"].values() if v.is_floating_point())


# ------------------------------------------------------------------------------------------------ auxiliary operands / calibration
class _Calib:
    """A capture object of the calibration protocol (see neuronspark.modules.pkm)."""

    def __init__(self, mode):
        self.mode = mode
        self.records = []

    def pkm_mode(self, layer):
        return self.mode

    def record(self, layer, name, value):
        self.records.append((name, value))


@pytest.mark.sm100
def test_aux_out_operands(device):
    T, B = 256, 2
    layer = _loaded(3, _exercised_state(3, device), device)
    ids, table, h, carry = _row_inputs(T, B, device, 0)
    aux = []
    with torch.no_grad():
        _whole_row(layer, ids, table, h, carry, aux_out=aux)
    assert len(aux) == 1
    z, r, pi = aux[0]
    assert tuple(z.shape) == (T, B, D) and z.dtype == torch.float32
    assert tuple(r.shape) == (T, B, D) and r.dtype == torch.bfloat16
    assert tuple(pi.shape) == (T, B) and pi.dtype == torch.float32
    assert not (z.requires_grad or r.requires_grad or pi.requires_grad)


@pytest.mark.sm100
def test_calibration_capture_protocol(device):
    """An uncalibrated layer inside a calibration adds no PKM read and no read gain in every pass; the passes record
    the protocol's entries."""
    T, B = 256, 2
    state = _exercised_state(3, device)
    state["pkm_calibrated"] = torch.zeros((), device=device)
    ids, table, h, carry = _row_inputs(T, B, device, 0)
    outs = {}
    records = {}
    for mode in (None, "queries", "rms"):
        layer = _layer(3, device, init="empty")
        layer.load_state_dict(state, strict=True)
        assert not layer.pkm_ready
        calib = _Calib(mode)
        with torch.no_grad():
            outs[mode], _ = _whole_row(layer, ids, table, h, carry, calib=calib)
        records[mode] = {name: value for name, value in calib.records}
    assert records[None] == {}
    assert set(records["queries"]) == {"copy_max", "pkm_queries"}
    assert tuple(records["queries"]["copy_max"].shape) == (T * B,)
    assert tuple(records["queries"]["pkm_queries"].shape) == (T * B, 4, 2, 64)
    assert set(records["rms"]) == {"pkm_usage", "ms_y", "ms_res"}
    assert tuple(records["rms"]["ms_y"].shape) == (T * B,) and records["rms"]["ms_y"].dtype == torch.float64
    assert_all_match(outs["queries"], outs[None], prefix="queries_pass")
    assert_all_match(outs["rms"], outs[None], prefix="rms_pass")


# ------------------------------------------------------------------------------------------------ inference
@pytest.mark.parametrize("layer_idx,hippocampal_index", LAYERS, ids=LAYER_IDS)
@pytest.mark.parametrize("B", [1, 4])
def test_split_prefill_vs_one_prefill(device, layer_idx, hippocampal_index, B):
    """[prefill 768 | prefill 256] against one prefill of 1024 tokens: outputs and every carried state."""
    T = 1024
    state = _exercised_state(layer_idx, device)
    ids, table, h, carry = _row_inputs(T, B, device, hippocampal_index)
    h = _rows(h)

    def run(cuts):
        layer = _loaded(layer_idx, state, device)
        ip = _cache(layer, B, device)
        outs, carries = [], []
        for a, b in cuts:
            o, c = _infer_call(layer, ip, ids[:, a:b], table, _rows(h[a:b]), None if carry is None else carry[a:b])
            outs.append(o)
            carries.append(c)
        return {"h": torch.cat(outs, 0), "carry": torch.cat(carries, 0), **_state_tensors(ip)}

    one = run([(0, T)])
    split = run([(0, 768), (768, T)])
    assert_all_match(split, one, prefix=f"layer{layer_idx}.b{B}.split")


@pytest.mark.parametrize("B", [1, 32])
def test_decode_routes_agree(device, monkeypatch, B):
    """At a ring-kernel decode size the ring kernel against the torch ring chain + extra-query formation kernel (the
    route of the other sizes), on the same state."""
    T, steps = 64, 4
    state = _exercised_state(7, device)
    ids, table, h, carry = _row_inputs(T + steps, B, device, 1)
    h = _rows(h)

    def run(sizes):
        monkeypatch.setattr(PA, "RING_KERNEL_DECODE_SIZES", sizes)
        layer = _loaded(7, state, device)
        ip = _cache(layer, B, device)
        _infer_call(layer, ip, ids[:, :T], table, _rows(h[:T]), carry[:T])
        outs = []
        for t in range(T, T + steps):
            o, _ = _infer_call(layer, ip, ids[:, t:t + 1], table, _rows(h[t:t + 1]), carry[t:t + 1])
            outs.append(o)
        return {"h": torch.cat(outs, 0), **_state_tensors(ip)}

    kernel = run(frozenset((1, 32)))
    chain = run(frozenset())
    assert_all_match(kernel, chain, prefix=f"b{B}.ring_kernel_vs_chain")


@pytest.mark.parametrize("B", [1, 4])
def test_decode_cuda_graph_replay(device, B):
    """A captured decode step replayed from a state equals the eager step from that state (outputs and states)."""
    T = 64
    state = _exercised_state(7, device)
    ids, table, h, carry = _row_inputs(T + 1, B, device, 1)
    h = _rows(h)
    layer = _loaded(7, state, device)
    ip = _cache(layer, B, device)
    _infer_call(layer, ip, ids[:, :T], table, _rows(h[:T]), carry[:T])
    s_ids = ids[:, T:T + 1].clone()
    s_h = _rows(h[T:T + 1]).clone()
    s_c = carry[T:T + 1].clone()

    def step():
        out, _ = _infer_call(layer, ip, s_ids, table, s_h, s_c)
        return out

    snap = ip.snapshot()
    eager = step().clone()
    eager_state = _state_tensors(ip)
    ip.restore(snap)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
            ip.restore(snap)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step()
    ip.restore(snap)
    graph.replay()
    torch.cuda.synchronize()
    assert_all_match({"h": static_out, **_state_tensors(ip)}, {"h": eager, **eager_state}, prefix=f"b{B}.graph")


@pytest.mark.parametrize("B", [1, 8])
def test_decode_steps_vs_prefill_report(device, B):
    """Teacher-forced decode steps after a prefill against one prefill over the same tokens (reported: the decode
    kernels and the prefill kernels reduce in other orders)."""
    T, steps = 256, 8
    state = _exercised_state(7, device)
    ids, table, h, carry = _row_inputs(T + steps, B, device, 1)
    h = _rows(h)
    layer = _loaded(7, state, device)
    ip = _cache(layer, B, device)
    one, _ = _infer_call(layer, ip, ids, table, h, carry)
    layer = _loaded(7, state, device)
    ip = _cache(layer, B, device)
    _infer_call(layer, ip, ids[:, :T], table, _rows(h[:T]), carry[:T])
    outs = [_infer_call(layer, ip, ids[:, t:t + 1], table, _rows(h[t:t + 1]), carry[t:t + 1])[0]
            for t in range(T, T + steps)]
    res = match_stats(torch.cat(outs, 0), one[T:], f"b{B}.steps_vs_prefill")
    print(res)
    assert bool(torch.isfinite(torch.cat(outs, 0)).all())


@pytest.mark.sm100
def test_prefill_vs_whole_row_report(device):
    """A prefill from a zero state against the whole row on the same tokens (reported: the regimes differ in the TF32
    setting, the match-gain rounding and the kernels)."""
    T, B = 1024, 2
    state = _exercised_state(7, device)
    ids, table, h, carry = _row_inputs(T, B, device, 1)
    layer = _loaded(7, state, device)
    with torch.no_grad():
        whole, _ = _whole_row(layer, ids, table, h, carry)
    ip = _cache(layer, B, device)
    pre, _ = _infer_call(layer, ip, ids, table, _rows(h), carry)
    res = match_stats(pre, whole["h"], "prefill_vs_whole_row")
    print(res)
    assert bool(torch.isfinite(pre).all())
