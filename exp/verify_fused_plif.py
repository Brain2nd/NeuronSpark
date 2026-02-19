"""
验证 Fused PLIF Triton kernel 的正确性。
对比 Triton fused kernel vs 逐步 sequential 参考实现。
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
from spikingjelly.activation_based import surrogate

from atomic_ops.parallel_scan import (
    plif_parallel_forward,
    _HAS_TRITON,
)

device = 'cuda'
print(f"Triton available: {_HAS_TRITON}")
assert _HAS_TRITON, "Triton not available!"


def ref_plif_sequential(beta, u, v_th, v_init, surrogate_function):
    """Sequential reference implementation (ground truth)."""
    K = beta.shape[0]
    v = v_init.clone().float()
    spikes = []
    V_posts = []
    for k in range(K):
        v_pre = beta[k].float() * v + u[k].float()
        sp = (v_pre >= v_th[k].float()).float()
        v = v_pre - v_th[k].float() * sp
        spikes.append(surrogate_function(v_pre - v_th[k].float()))
        V_posts.append(v.clone())
    return torch.stack(spikes), torch.stack(V_posts)


def test_forward_correctness(K, batch, D, dtype, label):
    """Test 1: Forward numerical correctness vs sequential reference."""
    torch.manual_seed(42)
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype))
    u = torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1
    v_th_raw = torch.randn(K, batch, D, device=device, dtype=dtype)
    v_th = 0.1 + torch.abs(v_th_raw) * 0.5
    v_init = torch.randn(batch, D, device=device, dtype=dtype) * 0.1

    surr = surrogate.Sigmoid(alpha=4.0)

    # Triton fused
    spike_t, V_post_t, V_pre_t = plif_parallel_forward(
        beta, u, v_th, v_init, max_iter=3, surrogate_function=surr,
    )
    assert V_pre_t is None, "Fused path should return None for V_pre"

    # Sequential reference
    spike_ref, V_post_ref = ref_plif_sequential(beta, u, v_th, v_init, surr)

    # Compare (both should be identical since fused kernel IS sequential computation)
    spike_diff = (spike_t.float() - spike_ref.float()).abs().max().item()
    V_diff = (V_post_t.float() - V_post_ref.float()).abs().max().item()
    V_rel = V_diff / (V_post_ref.float().abs().max().item() + 1e-10)

    print(f"  [{label}] shape=({K},{batch},{D})")
    print(f"    spike max_diff={spike_diff:.2e}  V_post max_abs={V_diff:.2e} rel={V_rel:.2e}", end="")

    # Tolerance: fp32 accumulation in both, should be very close
    # bf16 inputs: Triton loads as bf16 then casts to fp32; ref casts to fp32 per step
    if dtype == torch.bfloat16:
        ok = V_rel < 0.01 and spike_diff < 0.01
    else:
        ok = V_diff < 1e-4 and spike_diff < 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradient_correctness(K, batch, D, label):
    """Test 2: Gradient correctness (fused vs torch.autograd.gradcheck-style)."""
    torch.manual_seed(42)

    # Fused Triton path
    beta_logits_t = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    u_t = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_th_raw_t = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_init_t = torch.randn(batch, D, device=device, dtype=torch.float32, requires_grad=True)

    surr = surrogate.Sigmoid(alpha=4.0)

    beta_t = torch.sigmoid(beta_logits_t)
    v_th_t = 0.1 + torch.abs(v_th_raw_t) * 0.5
    spike_t, V_post_t, _ = plif_parallel_forward(
        beta_t, u_t, v_th_t, v_init_t, max_iter=3, surrogate_function=surr,
    )
    loss_t = (spike_t.sum() + V_post_t.sum())
    loss_t.backward()

    grad_beta_logits_t = beta_logits_t.grad.clone()
    grad_u_t = u_t.grad.clone()
    grad_v_th_raw_t = v_th_raw_t.grad.clone()
    grad_v_init_t = v_init_t.grad.clone()

    # Sequential reference (using standard autograd)
    beta_logits_r = beta_logits_t.data.clone().requires_grad_(True)
    u_r = u_t.data.clone().requires_grad_(True)
    v_th_raw_r = v_th_raw_t.data.clone().requires_grad_(True)
    v_init_r = v_init_t.data.clone().requires_grad_(True)

    beta_r = torch.sigmoid(beta_logits_r)
    v_th_r = 0.1 + torch.abs(v_th_raw_r) * 0.5

    # Sequential with autograd
    v = v_init_r.clone()
    spikes_r = []
    V_posts_r = []
    for k in range(K):
        v_pre = beta_r[k] * v + u_r[k]
        sp_detached = (v_pre.detach() >= v_th_r[k].detach()).float()
        v = v_pre - v_th_r[k] * sp_detached
        spikes_r.append(surr(v_pre - v_th_r[k]))
        V_posts_r.append(v)
    spike_r = torch.stack(spikes_r)
    V_post_r = torch.stack(V_posts_r)
    loss_r = (spike_r.sum() + V_post_r.sum())
    loss_r.backward()

    diff_beta = (grad_beta_logits_t - beta_logits_r.grad).abs().max().item()
    diff_u = (grad_u_t - u_r.grad).abs().max().item()
    diff_v_th = (grad_v_th_raw_t - v_th_raw_r.grad).abs().max().item()
    diff_v_init = (grad_v_init_t - v_init_r.grad).abs().max().item()

    print(f"  [{label}] shape=({K},{batch},{D})")
    print(f"    grad_beta_logits max_diff={diff_beta:.2e}")
    print(f"    grad_u max_diff={diff_u:.2e}")
    print(f"    grad_v_th_raw max_diff={diff_v_th:.2e}")
    print(f"    grad_v_init max_diff={diff_v_init:.2e}")

    ok = diff_beta < 1e-3 and diff_u < 1e-3 and diff_v_th < 1e-3 and diff_v_init < 1e-3
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def test_no_grad():
    """Test 3: no_grad path works."""
    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)
    beta = torch.sigmoid(torch.randn(64, 2, 32, device=device))
    u = torch.randn(64, 2, 32, device=device) * 0.1
    v_th = 0.1 + torch.abs(torch.randn(64, 2, 32, device=device)) * 0.5
    v_init = torch.zeros(2, 32, device=device)

    with torch.no_grad():
        spike, V_post, _ = plif_parallel_forward(
            beta, u, v_th, v_init, max_iter=3, surrogate_function=surr,
        )

    assert spike.shape == (64, 2, 32)
    assert not spike.requires_grad
    print(f"  [no_grad] shape={spike.shape} PASS")
    return True


def test_spike_rate_sanity():
    """Test 4: Spike rate should be reasonable (not 0% or 100%)."""
    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)

    K, batch, D = 128, 2, 512
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device))
    u = torch.randn(K, batch, D, device=device) * 0.3
    v_th = 0.1 + torch.abs(torch.randn(K, batch, D, device=device)) * 0.3
    v_init = torch.zeros(batch, D, device=device)

    spike, V_post, _ = plif_parallel_forward(
        beta, u, v_th, v_init, max_iter=3, surrogate_function=surr,
    )
    rate = spike.mean().item()
    print(f"  [spike_rate] rate={rate:.3f}", end="")
    ok = 0.01 < rate < 0.99
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_plif_fixed_param():
    """Test 5: plif_fixed_param_forward with scalar beta/v_th dispatches to fused kernel."""
    from atomic_ops.parallel_scan import plif_fixed_param_forward

    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)
    K, batch, D = 128, 2, 768

    w = torch.tensor(0.5, device=device, requires_grad=True)
    beta = torch.sigmoid(w)
    u = torch.randn(K, batch, D, device=device, requires_grad=True)
    v_th = 0.3
    v_init = torch.zeros(batch, D, device=device)

    spike, V_post = plif_fixed_param_forward(
        beta, u, v_th, v_init, max_iter=3, surrogate_function=surr,
    )

    assert spike.shape == (K, batch, D)
    loss = spike.sum()
    loss.backward()

    has_w_grad = w.grad is not None and w.grad.abs().item() > 0
    has_u_grad = u.grad is not None and u.grad.abs().sum() > 0
    print(f"  [fixed_param] spike_rate={spike.mean():.3f} w_grad={has_w_grad} u_grad={has_u_grad}", end="")
    ok = has_w_grad and has_u_grad
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_real_training_shapes():
    """Test 6: Real training shapes — hidden (8192,2,6144) and output (8192,2,768)."""
    torch.manual_seed(42)
    surr = surrogate.Sigmoid(alpha=4.0)

    for label, D in [("hidden_bf16", 6144), ("output_bf16", 768)]:
        K, batch = 8192, 2
        beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=torch.bfloat16))
        u = torch.randn(K, batch, D, device=device, dtype=torch.bfloat16) * 0.1
        v_th = 0.1 + torch.abs(torch.randn(K, batch, D, device=device, dtype=torch.bfloat16)) * 0.5
        v_init = torch.zeros(batch, D, device=device, dtype=torch.bfloat16)

        spike, V_post, _ = plif_parallel_forward(
            beta, u, v_th, v_init, max_iter=3, surrogate_function=surr,
        )
        rate = spike.mean().item()
        print(f"  [{label}] shape=({K},{batch},{D}) spike_rate={rate:.3f} PASS")

    return True


# ============================================================
# Run all tests
# ============================================================
print("=" * 60)
print("Test 1: Forward numerical correctness (fused vs sequential)")
print("=" * 60)
all_ok = True

all_ok &= test_forward_correctness(64, 2, 32, torch.float32, "small_fp32")
all_ok &= test_forward_correctness(64, 2, 32, torch.bfloat16, "small_bf16")
all_ok &= test_forward_correctness(512, 2, 768, torch.float32, "medium_fp32")
all_ok &= test_forward_correctness(512, 2, 768, torch.bfloat16, "medium_bf16")

print()
print("=" * 60)
print("Test 2: Gradient correctness (fp32)")
print("=" * 60)
all_ok &= test_gradient_correctness(64, 2, 32, "small")
all_ok &= test_gradient_correctness(256, 2, 128, "medium")

print()
print("=" * 60)
print("Test 3: no_grad path")
print("=" * 60)
all_ok &= test_no_grad()

print()
print("=" * 60)
print("Test 4: Spike rate sanity")
print("=" * 60)
all_ok &= test_spike_rate_sanity()

print()
print("=" * 60)
print("Test 5: plif_fixed_param_forward (scalar expand)")
print("=" * 60)
all_ok &= test_plif_fixed_param()

print()
print("=" * 60)
print("Test 6: Real training shapes (smoke test)")
print("=" * 60)
all_ok &= test_real_training_shapes()

print()
print("=" * 60)
if all_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)
