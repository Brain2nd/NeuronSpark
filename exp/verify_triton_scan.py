"""
验证 Triton fused recurrence kernel 的正确性。
用真实训练形状：(8192, 2, 6144) 和 (8192, 2, 768)。
对比 Triton vs Hillis-Steele 参考实现。
"""
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
from atomic_ops.parallel_scan import (
    hillis_steele_scan,
    linear_recurrence,
    _HAS_TRITON,
)

device = 'cuda'
print(f"Triton available: {_HAS_TRITON}")
assert _HAS_TRITON, "Triton not available!"


def ref_linear_recurrence(beta, u, v_init):
    """Hillis-Steele reference (CPU-style, but on GPU for comparison)."""
    A, B = hillis_steele_scan(beta, u)
    return A * v_init.unsqueeze(0) + B


def test_forward_correctness(K, batch, D, dtype, label):
    """Test 1: Forward numerical correctness."""
    torch.manual_seed(42)
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype))
    u = torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1
    v_init = torch.randn(batch, D, device=device, dtype=dtype) * 0.1

    # Triton
    V_triton = linear_recurrence(beta, u, v_init)

    # Reference
    V_ref = ref_linear_recurrence(beta, u, v_init)

    diff = (V_triton - V_ref).abs().max().item()
    rel_diff = diff / (V_ref.abs().max().item() + 1e-10)
    print(f"  [{label}] shape=({K},{batch},{D}) max_abs_diff={diff:.2e} rel_diff={rel_diff:.2e}", end="")

    # Triton accumulates in fp32 internally, Hillis-Steele computes in bf16
    # So differences are expected — Triton is actually MORE accurate
    if dtype == torch.bfloat16:
        ok = rel_diff < 0.02  # 2% relative tolerance for bf16 vs fp32 accumulation
    else:
        ok = diff < 1e-5
    print(f" {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_correctness(K, batch, D, label):
    """Test 2: Gradient correctness (Triton vs Hillis-Steele)."""
    torch.manual_seed(42)

    # Triton path
    beta_logits_t = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    u_t = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_init_t = torch.randn(batch, D, device=device, dtype=torch.float32, requires_grad=True)

    beta_t = torch.sigmoid(beta_logits_t)
    V_t = linear_recurrence(beta_t, u_t, v_init_t)
    loss_t = V_t.sum()
    loss_t.backward()

    grad_beta_logits_t = beta_logits_t.grad.clone()
    grad_u_t = u_t.grad.clone()
    grad_v_init_t = v_init_t.grad.clone()

    # Reference path
    beta_logits_r = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    beta_logits_r.data.copy_(beta_logits_t.data)
    u_r = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    u_r.data.copy_(u_t.data)
    v_init_r = torch.randn(batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_init_r.data.copy_(v_init_t.data)

    beta_r = torch.sigmoid(beta_logits_r)
    V_r = ref_linear_recurrence(beta_r, u_r, v_init_r)
    loss_r = V_r.sum()
    loss_r.backward()

    diff_beta = (grad_beta_logits_t - beta_logits_r.grad).abs().max().item()
    diff_u = (grad_u_t - u_r.grad).abs().max().item()
    diff_v = (grad_v_init_t - v_init_r.grad).abs().max().item()

    print(f"  [{label}] shape=({K},{batch},{D})")
    print(f"    grad_beta_logits max_diff={diff_beta:.2e}")
    print(f"    grad_u max_diff={diff_u:.2e}")
    print(f"    grad_v_init max_diff={diff_v:.2e}")

    ok = diff_beta < 1e-3 and diff_u < 1e-3 and diff_v < 1e-3
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def test_no_grad():
    """Test 3: Verify no_grad path works (Phase 2 of plif_parallel_forward)."""
    torch.manual_seed(42)
    beta = torch.sigmoid(torch.randn(64, 2, 32, device=device))
    u = torch.randn(64, 2, 32, device=device) * 0.1
    v_init = torch.zeros(2, 32, device=device)

    with torch.no_grad():
        V = linear_recurrence(beta, u, v_init)

    assert V.shape == (64, 2, 32)
    assert not V.requires_grad
    print(f"  [no_grad] shape={V.shape} PASS")
    return True


def test_plif_forward():
    """Test 4: Full plif_parallel_forward with Triton backend."""
    from atomic_ops.parallel_scan import plif_parallel_forward
    from spikingjelly.activation_based import surrogate

    torch.manual_seed(42)
    K, batch, DN = 128, 2, 512  # Small for quick test
    # Use leaf tensors for gradient checking
    beta_logits = torch.randn(K, batch, DN, device=device, requires_grad=True)
    u = torch.randn(K, batch, DN, device=device, requires_grad=True)
    v_th_raw = torch.randn(K, batch, DN, device=device, requires_grad=True)
    v_init = torch.zeros(batch, DN, device=device)

    beta = torch.sigmoid(beta_logits)
    v_th = 0.1 + torch.abs(v_th_raw) * 0.5

    surr = surrogate.Sigmoid(alpha=4.0)
    spike, V_post, V_pre = plif_parallel_forward(beta, u, v_th, v_init, max_iter=3, surrogate_function=surr)

    assert spike.shape == (K, batch, DN)
    assert V_post.shape == (K, batch, DN)

    # Check backward works on leaf tensors
    loss = (spike.sum() + V_post.sum())
    loss.backward()

    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in [beta_logits, u, v_th_raw]
    )
    print(f"  [plif_forward] spike_rate={spike.mean():.3f} grads_ok={has_grad}", end="")
    print(f" {'PASS' if has_grad else 'FAIL'}")
    return has_grad


def test_fixed_param_forward():
    """Test 5: plif_fixed_param_forward with scalar beta/v_th (expanded + contiguous)."""
    from atomic_ops.parallel_scan import plif_fixed_param_forward
    from spikingjelly.activation_based import surrogate

    torch.manual_seed(42)
    K, batch, D = 128, 2, 768

    # Scalar beta (like output_neuron.w → sigmoid)
    # Create 0-dim leaf tensor properly
    w = torch.tensor(0.5, device=device, requires_grad=True)
    beta = torch.sigmoid(w)
    u = torch.randn(K, batch, D, device=device, requires_grad=True)
    v_th = 0.3  # Python float
    v_init = torch.zeros(batch, D, device=device)

    surr = surrogate.Sigmoid(alpha=4.0)
    spike, V_post = plif_fixed_param_forward(beta, u, v_th, v_init, max_iter=3, surrogate_function=surr)

    assert spike.shape == (K, batch, D)
    loss = spike.sum()
    loss.backward()

    has_w_grad = w.grad is not None and w.grad.abs().item() > 0
    has_u_grad = u.grad is not None and u.grad.abs().sum() > 0
    print(f"  [fixed_param] spike_rate={spike.mean():.3f} w_grad={has_w_grad} u_grad={has_u_grad}", end="")
    ok = has_w_grad and has_u_grad
    print(f" {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# Run all tests
# ============================================================
print("=" * 60)
print("Test 1: Forward numerical correctness")
print("=" * 60)
all_ok = True

# Small shape (quick)
all_ok &= test_forward_correctness(64, 2, 32, torch.float32, "small_fp32")
all_ok &= test_forward_correctness(64, 2, 32, torch.bfloat16, "small_bf16")

# Medium shape
all_ok &= test_forward_correctness(512, 2, 768, torch.float32, "medium_fp32")
all_ok &= test_forward_correctness(512, 2, 768, torch.bfloat16, "medium_bf16")

# Real training shapes (hidden neurons)
all_ok &= test_forward_correctness(8192, 2, 6144, torch.bfloat16, "real_hidden_bf16")

# Real training shapes (output neurons)
all_ok &= test_forward_correctness(8192, 2, 768, torch.bfloat16, "real_output_bf16")

print()
print("=" * 60)
print("Test 2: Gradient correctness (fp32)")
print("=" * 60)
# Use smaller shapes for gradient test (Hillis-Steele reference is memory-heavy)
all_ok &= test_backward_correctness(64, 2, 32, "small")
all_ok &= test_backward_correctness(256, 2, 128, "medium")

print()
print("=" * 60)
print("Test 3: no_grad path")
print("=" * 60)
all_ok &= test_no_grad()

print()
print("=" * 60)
print("Test 4: plif_parallel_forward (end-to-end)")
print("=" * 60)
all_ok &= test_plif_forward()

print()
print("=" * 60)
print("Test 5: plif_fixed_param_forward (scalar expand)")
print("=" * 60)
all_ok &= test_fixed_param_forward()

print()
print("=" * 60)
if all_ok:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)
