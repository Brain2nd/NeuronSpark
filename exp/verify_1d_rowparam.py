"""
验证 1D row-param PLIF kernel 的正确性和性能。

优化内容：
  OLD: beta.expand(batch, D).contiguous() → plif_rowparam_forward((batch,D), ...)
  NEW: plif_rowparam_forward((D,), ...) → 1D kernel, cols % param_dim 索引

验证项目：
  1. 前向输出 bit-exact 对比（1D kernel vs expanded row-param kernel）
  2. 反向梯度对比（对 beta, v_th, u, v_init）
  3. 性能对比
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
from spikingjelly.activation_based import surrogate

device = 'cuda'
dtype = torch.bfloat16

D, K, seq_len, batch = 768, 16, 512, 2
TK = seq_len * K
D_ff = 2304


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, label, n_warmup=5, n_iter=30):
    for _ in range(n_warmup):
        fn()
    t0 = sync_time()
    for _ in range(n_iter):
        fn()
    elapsed = (sync_time() - t0) / n_iter * 1000
    print(f"  {label:55s} {elapsed:8.3f} ms")
    return elapsed


def test_correctness(dim, label):
    """测试给定维度下 1D kernel vs expanded kernel 的数值一致性。"""
    print(f"\n--- 正确性验证: {label} (dim={dim}) ---")

    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
    )

    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)

    torch.manual_seed(42)
    # 1D parameters
    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    # ====== OLD: expand to (batch, dim) then use row-param kernel ======
    beta_exp = beta_1d.unsqueeze(0).expand(batch, dim).contiguous()
    v_th_exp = v_th_1d.unsqueeze(0).expand(batch, dim).contiguous()

    # Need grad tracking
    beta_1d_old = beta_1d.clone().requires_grad_(True)
    v_th_1d_old = v_th_1d.clone().requires_grad_(True)
    u_old = u.clone().requires_grad_(True)
    v_init_old = v_init.clone().requires_grad_(True)

    beta_exp_old = beta_1d_old.unsqueeze(0).expand(batch, dim).contiguous()
    v_th_exp_old = v_th_1d_old.unsqueeze(0).expand(batch, dim).contiguous()

    spike_old, vpost_old = _TritonPLIFRowParamForward.apply(
        beta_exp_old, u_old, v_th_exp_old, v_init_old, alpha,
    )
    loss_old = spike_old.sum() + vpost_old.sum()
    loss_old.backward()

    # ====== NEW: 1D kernel directly ======
    beta_1d_new = beta_1d.clone().requires_grad_(True)
    v_th_1d_new = v_th_1d.clone().requires_grad_(True)
    u_new = u.clone().requires_grad_(True)
    v_init_new = v_init.clone().requires_grad_(True)

    spike_new, vpost_new = _TritonPLIFRowParam1DForward.apply(
        beta_1d_new, u_new, v_th_1d_new, v_init_new, alpha,
    )
    loss_new = spike_new.sum() + vpost_new.sum()
    loss_new.backward()

    # ====== 对比 ======
    all_ok = True

    # Forward
    fwd_pairs = [
        ("spike", spike_old, spike_new),
        ("V_post", vpost_old, vpost_new),
    ]
    for name, old, new in fwd_pairs:
        max_diff = (old - new).abs().max().item()
        status = "✓ EXACT" if max_diff == 0 else f"✗ max_diff={max_diff:.2e}"
        print(f"  [fwd] {name:15s} {status}")
        if max_diff > 1e-3:
            all_ok = False

    # Backward
    bwd_pairs = [
        ("grad_u", u_old.grad, u_new.grad),
        ("grad_v_init", v_init_old.grad, v_init_new.grad),
        # beta/v_th grad: old is (batch,dim) summed from expand, new is (dim,) from atomic_add
        # old 的 grad 通过 expand contiguous 再 backward 返回 (dim,)
        ("grad_beta", beta_1d_old.grad, beta_1d_new.grad),
        ("grad_v_th", v_th_1d_old.grad, v_th_1d_new.grad),
    ]
    for name, old, new in bwd_pairs:
        if old is None or new is None:
            print(f"  [bwd] {name:15s} ✗ grad is None (old={old is None}, new={new is None})")
            all_ok = False
            continue
        max_diff = (old - new).abs().max().item()
        rel_diff = max_diff / (old.abs().max().item() + 1e-10)
        status = "✓ EXACT" if max_diff == 0 else f"max_diff={max_diff:.2e} (rel={rel_diff:.2e})"
        ok_str = "✓" if rel_diff < 0.02 else "✗"
        print(f"  [bwd] {name:15s} {ok_str} {status}")
        if rel_diff > 0.02:
            all_ok = False

    print(f"  结论: {'PASS ✓' if all_ok else 'FAIL ✗'}")
    return all_ok


def test_dispatch():
    """测试 plif_rowparam_forward 的自动分发逻辑。"""
    print("\n--- 自动分发验证 ---")

    from atomic_ops.parallel_scan import plif_rowparam_forward
    surr = surrogate.Sigmoid(alpha=4.0)

    torch.manual_seed(42)
    dim = D

    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    # OLD: caller does expand+contiguous
    beta_exp = beta_1d.unsqueeze(0).expand(batch, dim).contiguous()
    v_th_exp = v_th_1d.unsqueeze(0).expand(batch, dim).contiguous()
    spike_old, vpost_old = plif_rowparam_forward(beta_exp, u, v_th_exp, v_init, surr)

    # NEW: caller passes 1D directly
    spike_new, vpost_new = plif_rowparam_forward(beta_1d, u, v_th_1d, v_init, surr)

    max_s = (spike_old - spike_new).abs().max().item()
    max_v = (vpost_old - vpost_new).abs().max().item()
    print(f"  spike max_diff: {max_s:.2e}  {'✓ EXACT' if max_s == 0 else '✗'}")
    print(f"  V_post max_diff: {max_v:.2e}  {'✓ EXACT' if max_v == 0 else '✗'}")
    ok = max_s == 0 and max_v == 0
    print(f"  结论: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def test_benchmark(dim, label):
    """对比 1D kernel vs expanded kernel 性能。"""
    print(f"\n--- 性能对比: {label} (dim={dim}) ---")

    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
        plif_rowparam_forward,
    )
    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    # ====== Forward only ======
    print("  [Forward]")

    def old_fwd():
        beta_exp = beta_1d.unsqueeze(0).expand(batch, dim).contiguous()
        v_th_exp = v_th_1d.unsqueeze(0).expand(batch, dim).contiguous()
        _TritonPLIFRowParamForward.apply(beta_exp, u, v_th_exp, v_init, alpha)

    def new_fwd():
        _TritonPLIFRowParam1DForward.apply(beta_1d, u, v_th_1d, v_init, alpha)

    t_old = bench(old_fwd, "OLD: expand+contiguous → row-param kernel")
    t_new = bench(new_fwd, "NEW: 1D kernel (no expand)")
    print(f"  {'Forward speedup':55s} {t_old/t_new:.2f}x")

    # ====== Forward + Backward ======
    print("  [Forward + Backward]")

    def old_fwd_bwd():
        b = beta_1d.clone().requires_grad_(True)
        vt = v_th_1d.clone().requires_grad_(True)
        u_ = u.clone().requires_grad_(True)
        vi = v_init.clone().requires_grad_(True)
        be = b.unsqueeze(0).expand(batch, dim).contiguous()
        ve = vt.unsqueeze(0).expand(batch, dim).contiguous()
        s, v = _TritonPLIFRowParamForward.apply(be, u_, ve, vi, alpha)
        (s.sum() + v.sum()).backward()

    def new_fwd_bwd():
        b = beta_1d.clone().requires_grad_(True)
        vt = v_th_1d.clone().requires_grad_(True)
        u_ = u.clone().requires_grad_(True)
        vi = v_init.clone().requires_grad_(True)
        s, v = _TritonPLIFRowParam1DForward.apply(b, u_, vt, vi, alpha)
        (s.sum() + v.sum()).backward()

    t_old_fb = bench(old_fwd_bwd, "OLD: expand+contiguous → row-param (fwd+bwd)")
    t_new_fb = bench(new_fwd_bwd, "NEW: 1D kernel (fwd+bwd)")
    print(f"  {'Fwd+Bwd speedup':55s} {t_old_fb/t_new_fb:.2f}x")


if __name__ == '__main__':
    print(f"Config: D={D}, D_ff={D_ff}, K={K}, seq={seq_len}, batch={batch}, TK={TK}")
    print(f"Device: {device}, dtype: {dtype}")

    # 正确性
    ok1 = test_correctness(D, "D=768 (input/output neurons)")
    ok2 = test_correctness(2 * D_ff, "2*D_ff=4608 (FFN gate+up merged)")
    ok3 = test_dispatch()

    # 性能
    test_benchmark(D, "D=768 (input/output neurons)")
    test_benchmark(2 * D_ff, "2*D_ff=4608 (FFN gate+up merged)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  D=768 correctness:  {'PASS ✓' if ok1 else 'FAIL ✗'}")
    print(f"  2*D_ff correctness: {'PASS ✓' if ok2 else 'FAIL ✗'}")
    print(f"  Auto-dispatch:      {'PASS ✓' if ok3 else 'FAIL ✗'}")
