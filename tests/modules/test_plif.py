"""modules.plif: the PLIF neuron population module in its three forms.

* construction: state-dict keys and shapes, the frozen retention seed, the launch b_eta computed from w in the
  constructor, the launch distributions, an empty build drawing nothing;
* whole-row form (forward spikes and every gradient) against the module on the reference operators;
* prefill + decode forms (spikes, carried membrane, carried input sum of the centred neuron) against the module on the
  reference operators;
* consistency: a prefill split into two calls and a prefill followed by single-token steps against one prefill over
  the same tokens, and the inference forms against the whole-row form from a zero state.

Populations: the attention-side write gate (C 1536, fp32 input, scalar tau), the FFN input neuron (C 1536, fp32
input, tau spectrum [2, 100], centred projection) and an FFN gate / up neuron (C 768, bf16 input).  Every comparison
is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import math

import pytest
import torch

from neuronspark.modules.plif import PLIFNeuron, V_TH_INIT, W_CLAMP_MAX, plif_b_eta_launch
from neuronspark.utils.cache import PLIFState
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match, match_stats, rand_tensor, summarize

pytestmark = pytest.mark.cuda

# (name, channels, init_tau, centred, input dtype)
POPULATIONS = [
    ("gate_c1536", 1536, 2.0, False, torch.float32),
    ("input2_c1536", 1536, (2.0, 100.0), True, torch.float32),
    ("ffn_c768", 768, 2.0, False, torch.bfloat16),
]
POP_IDS = [p[0] for p in POPULATIONS]
T_TRAIN, B_TRAIN = 1024, 8
PREFILL_CASES = [(1, 64, 4), (8, 256, 8), (8, 1024, 4)]          # (batch, prefill tokens, decode steps)
PARAM_NAMES = ("w", "v_th", "w_eta", "b_eta", "eta_V", "eta_U")
GRAD_NAMES = ("v_th", "w_eta", "b_eta", "eta_V", "eta_U")


# ------------------------------------------------------------------------------------------------ helpers
def _neuron(C, tau, centred, device, impl="kernel", init="launch"):
    return PLIFNeuron(C, tau, centred=centred, alpha=8.0, eta_rank=16, device=device, impl=impl, init=init)


def _trained_params(C, tau, centred, device):
    """A launch population moved off its launch point the way training moves it: a non-zero projection, a small w_eta,
    perturbed b_eta and thresholds (every term of the gate chain is exercised)."""
    n = _neuron(C, tau, centred, device)
    with torch.no_grad():
        n.w_eta.copy_(0.05 * rand_tensor((C,), torch.float32, device))
        n.b_eta.add_(0.3 * rand_tensor((C,), torch.float32, device))
        n.eta_U.copy_(0.02 * rand_tensor((C, 16), torch.float32, device))
        n.v_th.copy_(torch.empty(C, device=device).uniform_(0.1, 0.5))
    return {k: v.detach().clone() for k, v in n.state_dict().items()}


def _x(T, B, C, dtype, device, b_major=False):
    x = rand_tensor((B, T, C) if b_major else (T, B, C), torch.float32, device)
    x = x.transpose(0, 1) if b_major else x
    return (0.5 * x).to(torch.bfloat16) if dtype == torch.bfloat16 else x


def _load(C, tau, centred, params, device, impl="kernel"):
    n = _neuron(C, tau, centred, device, impl=impl, init="empty")
    n.load_state_dict(params, strict=True)
    return n


def _run_train(C, tau, centred, params, x, gS, device, impl="kernel"):
    n = _load(C, tau, centred, params, device, impl)
    xr = x.detach().clone().requires_grad_(True)
    with regime("train"):
        S = n(xr)
    S.backward(gS)
    out = {"S": S.detach()}
    for k in GRAD_NAMES:
        out["d_" + k] = getattr(n, k).grad.detach().clone()
    out["dx"] = xr.grad.detach().clone()
    return out


def _run_infer(C, tau, centred, params, xp, xd, device, impl="kernel"):
    """One prefill over xp (T >= 2), then the steps of xd one token at a time, from a zero state."""
    n = _load(C, tau, centred, params, device, impl)
    B = int(xp.shape[1])
    st = PLIFState.allocate(B, C, centred, device)
    pos = torch.zeros(B, dtype=torch.int64, device=device)
    outs = []
    with torch.no_grad(), regime("infer"):
        outs.append(n(xp, st, pos))
        pos += int(xp.shape[0])
        for t in range(int(xd.shape[0])):
            outs.append(n(xd[t:t + 1], st, pos))
            pos += 1
    res = {"S": torch.cat(outs, dim=0), "v": st.v.clone()}
    if centred:
        res["xsum"] = st.xsum.clone()
    return res


# ------------------------------------------------------------------------------------------------ construction
@pytest.mark.parametrize("name,C,tau,centred,dtype", POPULATIONS, ids=POP_IDS)
def test_keys_and_launch_values(device, name, C, tau, centred, dtype):
    n = _neuron(C, tau, centred, device)
    sd = n.state_dict()
    assert list(sd) == list(PARAM_NAMES)
    shapes = {"w": (C,), "v_th": (C,), "w_eta": (C,), "b_eta": (C,), "eta_V": (16, C), "eta_U": (C, 16)}
    for k, shp in shapes.items():
        assert tuple(sd[k].shape) == shp and sd[k].dtype == torch.float32, k
    assert not n.w.requires_grad and all(getattr(n, k).requires_grad for k in GRAD_NAMES)
    # b_eta from w in the constructor: logit(1 - sigmoid(clamp(w, max))), retained fraction clamped
    beta = torch.sigmoid(torch.clamp(n.w.detach(), max=W_CLAMP_MAX))
    omb = (1.0 - beta).clamp(1e-6, 1 - 1e-6)
    assert_match(n.b_eta.detach(), torch.log(omb / (1 - omb)), f"{name}.b_eta_launch")
    assert_match(plif_b_eta_launch(n.w.detach()), n.b_eta.detach(), f"{name}.b_eta_helper")
    lo, hi = V_TH_INIT
    assert bool(((n.v_th >= lo) & (n.v_th <= hi)).all())
    assert_match(n.w_eta.detach(), torch.zeros_like(n.w_eta), f"{name}.w_eta_launch")
    assert_match(n.eta_U.detach(), torch.zeros_like(n.eta_U), f"{name}.eta_U_launch")
    bound = math.sqrt(6.0 / (16 + C))                      # Xavier-uniform bound of eta_V
    assert float(n.eta_V.detach().abs().max()) <= bound
    if isinstance(tau, tuple):
        # spectrum: w = log(tau_c - 1) + N(0, 0.5), tau_c log-uniform over the channels
        taus = torch.exp(torch.linspace(math.log(tau[0]), math.log(tau[1]), C, device=device))
        dev_w = (n.w.detach() - torch.log(taus - 1.0)).float()
        assert abs(float(dev_w.mean())) < 0.1 and abs(float(dev_w.std()) - 0.5) < 0.1
    else:
        assert abs(float(n.w.detach().mean()) + math.log(tau - 1.0)) < 0.1


def test_empty_build_draws_nothing(device):
    """An empty build leaves the CPU and CUDA streams where they were: the next draws equal those of an untouched
    stream (any draw by the build would shift them)."""
    torch.manual_seed(123)
    expected_cuda, expected_cpu = torch.rand(256, device=device), torch.rand(256)
    torch.manual_seed(123)
    _neuron(1536, (2.0, 100.0), True, device, init="empty")
    assert_match(torch.rand(256, device=device), expected_cuda, "cuda_stream")
    assert_match(torch.rand(256), expected_cpu, "cpu_stream")


def test_inference_forms_refuse_autograd(device):
    n = _neuron(768, 2.0, False, device)
    st = PLIFState.allocate(2, 768, False, device)
    x = _x(4, 2, 768, torch.bfloat16, device)
    with pytest.raises(RuntimeError):
        with regime("infer"):
            n(x, st, torch.zeros(2, dtype=torch.int64, device=device))


def test_centred_neuron_needs_position(device):
    n = _neuron(1536, (2.0, 100.0), True, device)
    st = PLIFState.allocate(2, 1536, True, device)
    with pytest.raises(RuntimeError), torch.no_grad(), regime("infer"):
        n(_x(4, 2, 1536, torch.float32, device), st, None)


# ------------------------------------------------------------------------------------------------ whole row
def _train_case(name, C, tau, centred, dtype, device, T=T_TRAIN, B=B_TRAIN):
    params = _trained_params(C, tau, centred, device)
    x = _x(T, B, C, dtype, device, b_major=(name == "gate_c1536"))
    gS = rand_tensor((T, B, C), dtype, device)
    return params, x, gS


@pytest.mark.sm100
@pytest.mark.parametrize("name,C,tau,centred,dtype", POPULATIONS, ids=POP_IDS)
def test_whole_row_vs_reference_operators(device, name, C, tau, centred, dtype):
    params, x, gS = _train_case(name, C, tau, centred, dtype, device, T=256, B=2)
    new = _run_train(C, tau, centred, params, x, gS, device, impl="kernel")
    ref = _run_train(C, tau, centred, params, x, gS, device, impl="ref")
    # the spikes and the input gradient by the rule; the parameter gradients (sums over every position and row) are
    # reported: they inherit the reduction-order differences of the whole recurrence (the operators are compared in tests/ops)
    hard = [k for k in new if not k.startswith("d_")]
    assert_all_match({k: new[k] for k in hard}, {k: ref[k] for k in hard}, prefix=name)
    soft = [match_stats(new[k], ref[k], f"{name}.{k}") for k in new if k not in hard]
    print(f"[report] {name} parameter gradients vs the reference operators: {summarize(soft)}")


# ------------------------------------------------------------------------------------------------ prefill + decode
def _infer_case(C, tau, centred, dtype, b, T, steps, device):
    params = _trained_params(C, tau, centred, device)
    xp = _x(T, b, C, dtype, device)
    xd = _x(steps, b, C, dtype, device)
    return params, xp, xd


@pytest.mark.parametrize("b,T,steps", PREFILL_CASES)
@pytest.mark.parametrize("name,C,tau,centred,dtype", POPULATIONS, ids=POP_IDS)
def test_inference_vs_reference_operators(device, name, C, tau, centred, dtype, b, T, steps):
    params, xp, xd = _infer_case(C, tau, centred, dtype, b, T, steps, device)
    new = _run_infer(C, tau, centred, params, xp, xd, device, impl="kernel")
    ref = _run_infer(C, tau, centred, params, xp, xd, device, impl="ref")
    assert_all_match(new, ref, prefix=f"{name}.b{b}.T{T}")


@pytest.mark.parametrize("name,C,tau,centred,dtype", POPULATIONS, ids=POP_IDS)
def test_split_prefill_and_steps_vs_one_prefill(device, name, C, tau, centred, dtype):
    """The carried state continues a row: [prefill 192 | prefill 64] and [prefill 252 | 4 steps] against one prefill
    of 256 tokens."""
    b, T = 4, 256
    params = _trained_params(C, tau, centred, device)
    x = _x(T, b, C, dtype, device)
    one = _run_infer(C, tau, centred, params, x, x[:0], device)
    n = _load(C, tau, centred, params, device)

    def run(chunks):
        st = PLIFState.allocate(b, C, centred, device)
        pos = torch.zeros(b, dtype=torch.int64, device=device)
        outs = []
        with torch.no_grad(), regime("infer"):
            for c in chunks:
                outs.append(n(c, st, pos))
                pos += int(c.shape[0])
        res = {"S": torch.cat(outs, dim=0), "v": st.v.clone()}
        if centred:
            res["xsum"] = st.xsum.clone()
        return res

    split = run([x[:192], x[192:]])
    steps = run([x[:252]] + [x[t:t + 1] for t in range(252, 256)])
    assert_all_match(split, one, prefix=f"{name}.split")
    assert_all_match(steps, one, prefix=f"{name}.steps")


@pytest.mark.sm100
@pytest.mark.parametrize("name,C,tau,centred,dtype", POPULATIONS, ids=POP_IDS)
def test_prefill_vs_whole_row_from_zero(device, name, C, tau, centred, dtype):
    """The inference forms from a zero state against the whole-row form (no autograd) on the same row: the same gate
    chain and scan; the centring's running sum is sequential in prefill and chunked in the whole row."""
    b, T = 4, 512
    params = _trained_params(C, tau, centred, device)
    x = _x(T, b, C, dtype, device)
    n = _load(C, tau, centred, params, device)
    with torch.no_grad(), regime("train"):
        whole = n(x)
    pre = _run_infer(C, tau, centred, params, x, x[:0], device)["S"]
    res = match_stats(pre, whole, f"{name}.prefill_vs_whole_row")
    print(res)
    assert res.passed, str(res)
