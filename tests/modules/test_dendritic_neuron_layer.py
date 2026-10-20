"""DendriticNeuronLayer / DendriticNeurons in the three regimes.

* the whole row (training) and an inference session (prefill chunks and decode steps from the row-start state, for
  both dendritic-layer state dtypes) against the pure-torch oracle ``modules/dendritic_minimal.py`` (float64, bf16
  rounding at the kernels' tensor boundaries);
* the reference operator namespace (``impl="ref"``) against the kernels, and the regimes against each other (chunked
  prefill and decode steps against one prefill);
* the parameter layout, the launch values, the inference cache and the guards.

Parameters: the seeded launch with perturbed vectors.  Every comparison is the per-operator rule of
``neuronspark.utils.testing`` (matched ratio 1.0 for every output and every state tensor).
"""
from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.dendritic_neuron_layer import DendriticNeuronLayer, DendriticNeurons
from neuronspark.modules.dendritic_minimal import DENDRITIC_PARAM_NAMES, dendritic_neuron_layer_ref, dendritic_step_ref
from neuronspark.utils.cache import CopyState, InductionState, InferenceParams, LexicalState, DendriticState
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match, compare_outputs, rand_tensor, summarize

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

CFG = NeuronSparkConfig()
D = CFG.D
DENDRITIC_LAYERS = CFG.dendritic_layer_ids
# a session: prefill chunks (aligned, unaligned) and decode steps, from the row-start state
SESSION = (256, 37, 1, 1, 64, 1, 1)
STATE_DTYPES = ("bfloat16", "float32")


# ------------------------------------------------------------------------------------------------ parameters
def _launch_params(layer_idx: int, device, seed: int = 1234) -> Dict[str, torch.Tensor]:
    """The seeded launch of layer ``layer_idx`` with the launch-constant vectors perturbed (so that every parameter
    takes part in the forward and receives a gradient)."""
    m = DendriticNeurons(layer_idx=layer_idx, device=device, launch_seed=seed + layer_idx)
    p = {k: v.detach().clone() for k, v in m.named_parameters()}
    g = torch.Generator(device=device).manual_seed(seed + 17 * layer_idx)

    def noise(t):
        return torch.randn(t.shape, generator=g, device=device, dtype=torch.float32)

    p["pre_norm_gain"] += 0.1 * noise(p["pre_norm_gain"])
    p["conv_bias"] += 0.02 * noise(p["conv_bias"])
    p["D"] += 0.1 * noise(p["D"])
    p["gamma"] += 0.1 * noise(p["gamma"])
    p["rho"] += 0.3 * noise(p["rho"])
    return p


def _build_layer(params: Dict[str, torch.Tensor], layer_idx: int, device, impl: str = "kernel") -> DendriticNeuronLayer:
    layer = DendriticNeuronLayer(CFG, layer_idx, device=device, impl=impl, init="empty")
    layer.mixer_a.load_state_dict(params, strict=True)
    return layer


def _inference_params(b: int, layers: Dict[int, DendriticState], dendritic_dtype: str, device,
                      W: int = 512) -> InferenceParams:
    pos, ptr = InferenceParams.allocate_counters(b, device)
    return InferenceParams(
        max_seqlen=W, max_batch_size=b, pos=pos, ring_ptr=ptr, dendritic_state_dtype=dendritic_dtype, layers=layers,
        lexical=LexicalState.allocate(b, 2, CFG.vocab_size, device),
        induction=InductionState.allocate(b, W, 4, CFG.vocab_size, device),
        copy=CopyState.allocate(b, W, 96, device), announce=False)


def _session_inputs(b: int, lengths, device) -> List[torch.Tensor]:
    return [rand_tensor((T, b, D), torch.float32, device) for T in lengths]


def _run_session(params, hs: List[torch.Tensor], b: int, dendritic_dtype: str, device, impl: str = "kernel") -> dict:
    layer = _build_layer(params, DENDRITIC_LAYERS[0], device, impl=impl)
    st = layer.allocate_inference_cache(b, dendritic_state_dtype=dendritic_dtype, device=device)
    ip = _inference_params(b, {layer.layer_idx: st}, dendritic_dtype, device)
    out = {}
    with torch.no_grad(), regime("infer"):
        for i, h in enumerate(hs):
            hh, hn_next = layer(h, ip)
            assert hn_next is None
            out[f"h{i}"] = hh
            out[f"conv{i}"] = st.conv.clone()
            out[f"ssm{i}"] = st.ssm.clone()
    return out


# ------------------------------------------------------------------------------------------------ layout and launch
def test_parameter_layout(device):
    layer = DendriticNeuronLayer(CFG, 0, device=device)
    sd = layer.state_dict()
    assert sorted(sd) == sorted(f"mixer_a.{n}" for n in DENDRITIC_PARAM_NAMES)
    shapes = {n: tuple(t.shape) for n, t in layer.mixer_a.named_parameters()}
    assert shapes == {
        "pre_norm_gain": (1536,), "in_proj.weight": (18832, 1536), "conv_weight": (9472, 4), "conv_bias": (9472,),
        "dt_bias": (144,), "A_log": (144,), "D": (144,), "gamma": (9216,), "W_out": (1536, 9216), "rho": (9216,)}
    assert sum(t.numel() for t in layer.parameters()) == CFG.dendritic_layer_params
    assert all(t.dtype == torch.float32 for t in sd.values())
    assert "norm_ones" not in sd and tuple(layer.mixer_a.norm_ones.shape) == (9216,)
    theta = layer.mixer_a.theta()
    assert_match(theta, torch.full_like(theta, CFG.dendritic_theta0), "launch_theta")


# ------------------------------------------------------------------------------------------------ whole row
def test_whole_row_vs_oracle(device):
    p = _launch_params(DENDRITIC_LAYERS[0], device)
    layer = _build_layer(p, DENDRITIC_LAYERS[0], device)
    B, T = 2, 512
    h = rand_tensor((B, T, D), torch.float32, device).transpose(0, 1)
    with torch.no_grad():
        with regime("train"):
            out, hn_next = layer(h)
        assert hn_next is None
        upd = dendritic_neuron_layer_ref(p, h, dtype=torch.float64, round_like_kernels=True)
    assert_match(out, h.double() + upd, "whole_row_vs_oracle", check_dtype=False)


# ------------------------------------------------------------------------------------------------ inference session
@pytest.mark.parametrize("dendritic_dtype", STATE_DTYPES)
@pytest.mark.parametrize("b", (1, 2))
def test_session_vs_oracle(device, b, dendritic_dtype):
    p = _launch_params(DENDRITIC_LAYERS[0], device)
    hs = _session_inputs(b, SESSION, device)
    new = _run_session(p, hs, b, dendritic_dtype, device)
    ref_state = {
        "conv": torch.zeros((b, CFG.dendritic_conv_dim, CFG.dendritic_conv_width - 1), dtype=torch.float64,
                            device=device),
        "ssm": torch.zeros((b, CFG.dendritic_nheads, CFG.dendritic_head_dim, CFG.dendritic_state_size),
                           dtype=torch.bfloat16 if dendritic_dtype == "bfloat16" else torch.float64, device=device),
    }
    ref = {}
    with torch.no_grad():
        for i, h in enumerate(hs):
            if int(h.shape[0]) == 1:
                upd = dendritic_step_ref(p, h[0], ref_state, dtype=torch.float64, round_like_kernels=True)[None]
            else:
                upd = dendritic_neuron_layer_ref(p, h, ref_state, dtype=torch.float64, round_like_kernels=True)
            ref[f"h{i}"] = h.double() + upd
            ref[f"conv{i}"] = ref_state["conv"].clone()
            ref[f"ssm{i}"] = ref_state["ssm"].clone()
    assert_all_match(new, ref, prefix=f"session_oracle_b{b}_{dendritic_dtype}", check_dtype=False)


def test_prefill_and_decode_vs_one_prefill(device):
    """The regimes against each other (fp32 state): one prefill of 300 tokens vs a 256-token prefill followed by
    44 decode steps, and vs prefills of 256 + 44 tokens; every position's output and the final state."""
    b, T = 2, 300
    p = _launch_params(DENDRITIC_LAYERS[0], device)
    h = rand_tensor((T, b, D), torch.float32, device)
    one = _run_session(p, [h], b, "float32", device)
    steps = _run_session(p, [h[:256]] + [h[t:t + 1].contiguous() for t in range(256, T)], b, "float32", device)
    chunks = _run_session(p, [h[:256], h[256:].contiguous()], b, "float32", device)
    ref = {"h": one["h0"], "ssm": one["ssm0"], "conv": one["conv0"]}
    n = len(steps) // 3
    by_steps = {"h": torch.cat([steps[f"h{i}"] for i in range(n)], dim=0), "ssm": steps[f"ssm{n - 1}"],
                "conv": steps[f"conv{n - 1}"]}
    by_chunks = {"h": torch.cat([chunks["h0"], chunks["h1"]], dim=0), "ssm": chunks["ssm1"], "conv": chunks["conv1"]}
    results = compare_outputs(by_steps, ref, prefix="prefill_256_then_steps")
    results += compare_outputs(by_chunks, ref, prefix="prefill_256_then_44")
    s = summarize(results)
    assert s["status"] == "PASSED", "\n".join(str(r) for r in results if not r.passed)


def test_ref_namespace_matches_kernels(device):
    """``impl="ref"`` (the operators' pure-torch references, same call sequence) against the kernels, forward."""
    p = _launch_params(DENDRITIC_LAYERS[0], device)
    B, T = 1, 256
    h = rand_tensor((B, T, D), torch.float32, device).transpose(0, 1)
    k_layer = _build_layer(p, DENDRITIC_LAYERS[0], device, impl="kernel")
    r_layer = _build_layer(p, DENDRITIC_LAYERS[0], device, impl="ref")
    with torch.no_grad(), regime("train"):
        k_out, _ = k_layer(h)
        r_out, _ = r_layer(h)
    assert_match(k_out, r_out, "whole_row_kernel_vs_ref")
    hs = _session_inputs(2, (64, 1, 1), device)
    for dendritic_dtype in STATE_DTYPES:
        k = _run_session(p, hs, 2, dendritic_dtype, device, impl="kernel")
        r = _run_session(p, hs, 2, dendritic_dtype, device, impl="ref")
        assert_all_match(k, r, prefix=f"session_kernel_vs_ref_{dendritic_dtype}")


# ------------------------------------------------------------------------------------------------ cache and guards
def test_allocate_inference_cache(device):
    layer = DendriticNeuronLayer(CFG, 0, device=device, init="empty")
    st = layer.allocate_inference_cache(4)
    assert tuple(st.conv.shape) == (4, 9472, 3) and st.conv.dtype == torch.float32
    assert tuple(st.ssm.shape) == (4, 144, 64, 128) and st.ssm.dtype == torch.bfloat16    # the config default
    st32 = layer.allocate_inference_cache(4, dendritic_state_dtype="float32")
    assert st32.ssm.dtype == torch.float32
    with pytest.raises(ValueError):
        layer.allocate_inference_cache(4, dendritic_state_dtype="float16")


def test_guards(device):
    layer = _build_layer(_launch_params(DENDRITIC_LAYERS[0], device), DENDRITIC_LAYERS[0], device)
    with pytest.raises(RuntimeError):                       # the whole-row scan runs full chunks only
        with torch.no_grad(), regime("train"):
            layer(rand_tensor((255, 1, D), torch.float32, device))
    st = layer.allocate_inference_cache(1, dendritic_state_dtype="float32", device=device)
    ip = _inference_params(1, {layer.layer_idx: st}, "float32", device)
    with pytest.raises(RuntimeError):                       # inference calls run without autograd
        with regime("infer"):
            layer(rand_tensor((4, 1, D), torch.float32, device), ip)
    with pytest.raises(TypeError):                          # the stream is fp32
        with torch.no_grad(), regime("infer"):
            layer(rand_tensor((4, 1, D), torch.bfloat16, device), ip)
    with pytest.raises(ValueError):
        DendriticNeuronLayer(CFG, CFG.hippocampal_layer_ids[0], device=device, init="empty")
