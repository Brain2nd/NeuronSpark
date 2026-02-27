"""
Recompute 内核 0-bit 误差验证 + 速度对比。

测试维度: K=16, batch=2, dim=256（快速验证）

验证项:
1. PLIF rowparam: standard vs recompute — spike, V_last, backward grads
2. RF PLIF: standard vs recompute — spike, V_last, W_last, backward grads
3. 速度对比: 100 次 forward+backward 计时
"""

import torch
import time
import sys

# ============================================================
# Test 1: PLIF rowparam — standard vs recompute
# ============================================================

def test_plif_rowparam_recompute():
    """验证 plif_rowparam_forward_recompute 与 plif_rowparam_forward_alpha 完全一致。"""
    from atomic_ops.parallel_scan import plif_rowparam_forward_alpha, plif_rowparam_forward_recompute

    K, batch, dim = 16, 2, 256
    alpha = 4.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(42)
    beta_row = torch.sigmoid(torch.randn(batch, dim, device=device))
    u = torch.randn(K, batch, dim, device=device, requires_grad=True)
    v_th_row = torch.abs(torch.randn(batch, dim, device=device)) + 0.1
    v_init = torch.randn(batch, dim, device=device)

    # --- Standard path ---
    u_std = u.detach().clone().requires_grad_(True)
    beta_std = beta_row.detach().clone().requires_grad_(True)
    vth_std = v_th_row.detach().clone().requires_grad_(True)
    vinit_std = v_init.detach().clone().requires_grad_(True)

    spike_std, V_post_std = plif_rowparam_forward_alpha(
        beta_std, u_std, vth_std, vinit_std, alpha,
    )
    loss_std = spike_std.sum()
    loss_std.backward()

    # --- Recompute path ---
    u_rec = u.detach().clone().requires_grad_(True)
    beta_rec = beta_row.detach().clone().requires_grad_(True)
    vth_rec = v_th_row.detach().clone().requires_grad_(True)
    vinit_rec = v_init.detach().clone().requires_grad_(True)

    spike_rec, V_last_rec = plif_rowparam_forward_recompute(
        beta_rec, u_rec, vth_rec, vinit_rec, alpha,
    )
    loss_rec = spike_rec.sum()
    loss_rec.backward()

    # --- Verify forward ---
    assert torch.equal(spike_std, spike_rec), "PLIF rowparam: spike mismatch!"
    assert torch.equal(V_post_std[-1], V_last_rec), \
        f"PLIF rowparam: V_last mismatch! max diff={torch.abs(V_post_std[-1] - V_last_rec).max().item()}"

    # --- Verify backward ---
    assert torch.allclose(u_std.grad, u_rec.grad, atol=0, rtol=0), \
        f"PLIF rowparam: u.grad mismatch! max diff={torch.abs(u_std.grad - u_rec.grad).max().item()}"
    assert torch.allclose(beta_std.grad, beta_rec.grad, atol=0, rtol=0), \
        f"PLIF rowparam: beta.grad mismatch! max diff={torch.abs(beta_std.grad - beta_rec.grad).max().item()}"
    assert torch.allclose(vth_std.grad, vth_rec.grad, atol=0, rtol=0), \
        f"PLIF rowparam: vth.grad mismatch! max diff={torch.abs(vth_std.grad - vth_rec.grad).max().item()}"
    assert torch.allclose(vinit_std.grad, vinit_rec.grad, atol=0, rtol=0), \
        f"PLIF rowparam: vinit.grad mismatch! max diff={torch.abs(vinit_std.grad - vinit_rec.grad).max().item()}"

    print("[PASS] PLIF rowparam recompute: 0-bit exact match (forward + backward)")


# ============================================================
# Test 2: RF PLIF — standard vs recompute
# ============================================================

def test_rf_plif_recompute():
    """验证 rf_plif_parallel_forward_recompute 与 rf_plif_parallel_forward 完全一致。"""
    from atomic_ops.parallel_scan import rf_plif_parallel_forward, rf_plif_parallel_forward_recompute
    from spikingjelly.activation_based import surrogate

    K, batch, dim = 16, 2, 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    surr = surrogate.Sigmoid(alpha=4.0)

    torch.manual_seed(42)
    beta = torch.sigmoid(torch.randn(K, batch, dim, device=device))
    omega = torch.abs(torch.randn(K, batch, dim, device=device)) * 0.3
    # Stability: sqrt(beta^2 + omega^2) <= 0.999
    r_sq = beta * beta + omega * omega
    scale = torch.where(r_sq > 0.999**2, 0.999 * torch.rsqrt(r_sq.clamp(min=1e-12)), torch.ones_like(r_sq))
    beta = beta * scale
    omega = omega * scale

    u = torch.randn(K, batch, dim, device=device)
    v_th = torch.abs(torch.randn(K, batch, dim, device=device)) + 0.1
    v_init = torch.randn(batch, dim, device=device)
    w_init = torch.randn(batch, dim, device=device)

    # --- Standard path ---
    beta_std = beta.detach().clone().requires_grad_(True)
    omega_std = omega.detach().clone().requires_grad_(True)
    u_std = u.detach().clone().requires_grad_(True)
    vth_std = v_th.detach().clone().requires_grad_(True)
    vinit_std = v_init.detach().clone().requires_grad_(True)
    winit_std = w_init.detach().clone().requires_grad_(True)

    spike_std, V_post_std, W_std = rf_plif_parallel_forward(
        beta_std, omega_std, u_std, vth_std, vinit_std, winit_std,
        surrogate_function=surr,
    )
    loss_std = spike_std.sum()
    loss_std.backward()

    # --- Recompute path ---
    beta_rec = beta.detach().clone().requires_grad_(True)
    omega_rec = omega.detach().clone().requires_grad_(True)
    u_rec = u.detach().clone().requires_grad_(True)
    vth_rec = v_th.detach().clone().requires_grad_(True)
    vinit_rec = v_init.detach().clone().requires_grad_(True)
    winit_rec = w_init.detach().clone().requires_grad_(True)

    spike_rec, V_last_rec, W_last_rec = rf_plif_parallel_forward_recompute(
        beta_rec, omega_rec, u_rec, vth_rec, vinit_rec, winit_rec,
        surrogate_function=surr,
    )
    loss_rec = spike_rec.sum()
    loss_rec.backward()

    # --- Verify forward ---
    assert torch.equal(spike_std, spike_rec), "RF PLIF: spike mismatch!"
    assert torch.equal(V_post_std[-1], V_last_rec), \
        f"RF PLIF: V_last mismatch! max diff={torch.abs(V_post_std[-1] - V_last_rec).max().item()}"
    assert torch.equal(W_std[-1], W_last_rec), \
        f"RF PLIF: W_last mismatch! max diff={torch.abs(W_std[-1] - W_last_rec).max().item()}"

    # --- Verify backward ---
    for name, g_std, g_rec in [
        ('beta', beta_std.grad, beta_rec.grad),
        ('omega', omega_std.grad, omega_rec.grad),
        ('u', u_std.grad, u_rec.grad),
        ('v_th', vth_std.grad, vth_rec.grad),
        ('v_init', vinit_std.grad, vinit_rec.grad),
        ('w_init', winit_std.grad, winit_rec.grad),
    ]:
        assert torch.allclose(g_std, g_rec, atol=0, rtol=0), \
            f"RF PLIF: {name}.grad mismatch! max diff={torch.abs(g_std - g_rec).max().item()}"

    print("[PASS] RF PLIF recompute: 0-bit exact match (forward + backward)")


# ============================================================
# Test 3: Speed comparison
# ============================================================

def benchmark_plif_rowparam():
    """速度对比: standard vs recompute PLIF rowparam forward+backward。"""
    if not torch.cuda.is_available():
        print("[SKIP] Benchmark requires CUDA")
        return

    from atomic_ops.parallel_scan import plif_rowparam_forward_alpha, plif_rowparam_forward_recompute

    K, batch, dim = 16, 3, 8192  # 接近生产维度
    alpha = 4.0
    N_ITER = 100
    WARMUP = 10
    device = 'cuda'

    torch.manual_seed(42)
    beta_row = torch.sigmoid(torch.randn(batch, dim, device=device))
    v_th_row = torch.abs(torch.randn(batch, dim, device=device)) + 0.1
    v_init = torch.randn(batch, dim, device=device)

    def bench(forward_fn, label):
        # Warmup
        for _ in range(WARMUP):
            u = torch.randn(K, batch, dim, device=device, requires_grad=True)
            spike, _ = forward_fn(beta_row, u, v_th_row, v_init, alpha)
            spike.sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(N_ITER):
            u = torch.randn(K, batch, dim, device=device, requires_grad=True)
            spike, _ = forward_fn(beta_row, u, v_th_row, v_init, alpha)
            spike.sum().backward()
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / N_ITER
        print(f"  {label}: {ms:.3f} ms/iter")

    print("\n[BENCHMARK] PLIF rowparam (K=16, B=3, D=8192):")
    bench(plif_rowparam_forward_alpha, "standard ")
    bench(plif_rowparam_forward_recompute, "recompute")


def benchmark_rf_plif():
    """速度对比: standard vs recompute RF PLIF forward+backward。"""
    if not torch.cuda.is_available():
        print("[SKIP] Benchmark requires CUDA")
        return

    from atomic_ops.parallel_scan import rf_plif_parallel_forward, rf_plif_parallel_forward_recompute
    from spikingjelly.activation_based import surrogate

    K, batch, dim = 16, 3, 8192  # 接近生产维度 (TK=8192 for single token block)
    N_ITER = 100
    WARMUP = 10
    device = 'cuda'
    surr = surrogate.Sigmoid(alpha=4.0)

    torch.manual_seed(42)
    v_init = torch.randn(batch, dim, device=device)
    w_init = torch.randn(batch, dim, device=device)

    def bench(forward_fn, label, returns_full):
        # Warmup
        for _ in range(WARMUP):
            beta = torch.sigmoid(torch.randn(K, batch, dim, device=device))
            omega = torch.abs(torch.randn(K, batch, dim, device=device)) * 0.2
            u = torch.randn(K, batch, dim, device=device, requires_grad=True)
            v_th = torch.abs(torch.randn(K, batch, dim, device=device)) + 0.1
            out = forward_fn(beta, omega, u, v_th, v_init, w_init, surrogate_function=surr)
            out[0].sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(N_ITER):
            beta = torch.sigmoid(torch.randn(K, batch, dim, device=device))
            omega = torch.abs(torch.randn(K, batch, dim, device=device)) * 0.2
            u = torch.randn(K, batch, dim, device=device, requires_grad=True)
            v_th = torch.abs(torch.randn(K, batch, dim, device=device)) + 0.1
            out = forward_fn(beta, omega, u, v_th, v_init, w_init, surrogate_function=surr)
            out[0].sum().backward()
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / N_ITER
        print(f"  {label}: {ms:.3f} ms/iter")

    print("\n[BENCHMARK] RF PLIF (K=16, B=3, D=8192):")
    bench(rf_plif_parallel_forward, "standard ", returns_full=True)
    bench(rf_plif_parallel_forward_recompute, "recompute", returns_full=False)


if __name__ == '__main__':
    print("=" * 60)
    print("Recompute 内核 0-bit 误差验证")
    print("=" * 60)

    test_plif_rowparam_recompute()
    test_rf_plif_recompute()

    print("\n" + "=" * 60)
    print("Speed Benchmark")
    print("=" * 60)

    benchmark_plif_rowparam()
    benchmark_rf_plif()

    print("\nAll tests passed!")
