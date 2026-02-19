"""
多规模基准测试: 1D row-param kernel vs expand+contiguous row-param kernel

测试不同 batch size 下：
  1. 单次调用性能对比
  2. 模拟完整 forward pass (每层 8 次 PLIF, 20 层 = 160 次调用)
  3. Forward + Backward 对比

用法：
  TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas python exp/bench_1d_rowparam_scale.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
from spikingjelly.activation_based import surrogate

device = 'cuda'
dtype = torch.bfloat16

K = 16
seq_len = 512
TK = seq_len * K


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
    print(f"  {label:60s} {elapsed:8.3f} ms")
    return elapsed


def test_single_call(dim, batch, label):
    """单次 PLIF 调用性能对比。"""
    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
    )
    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    print(f"\n--- {label} | dim={dim}, batch={batch}, TK={TK} ---")
    print(f"  num_cols = batch×dim = {batch*dim}")

    # Forward
    print("  [Forward]")

    def old_fwd():
        beta_exp = beta_1d.unsqueeze(0).expand(batch, dim).contiguous()
        v_th_exp = v_th_1d.unsqueeze(0).expand(batch, dim).contiguous()
        _TritonPLIFRowParamForward.apply(beta_exp, u, v_th_exp, v_init, alpha)

    def new_fwd():
        _TritonPLIFRowParam1DForward.apply(beta_1d, u, v_th_1d, v_init, alpha)

    t_old = bench(old_fwd, "OLD: expand+contiguous → row-param kernel")
    t_new = bench(new_fwd, "NEW: 1D kernel (no expand)")
    speedup_fwd = t_old / t_new
    print(f"  {'Forward speedup':60s} {speedup_fwd:.3f}x")

    # Forward + Backward
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

    t_old_fb = bench(old_fwd_bwd, "OLD: expand+contiguous (fwd+bwd)")
    t_new_fb = bench(new_fwd_bwd, "NEW: 1D kernel (fwd+bwd)")
    speedup_fb = t_old_fb / t_new_fb
    print(f"  {'Fwd+Bwd speedup':60s} {speedup_fb:.3f}x")

    return speedup_fwd, speedup_fb


def test_cumulative(dim, batch, n_calls, label):
    """模拟完整 forward pass: n_calls 次 PLIF 调用。"""
    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
    )
    alpha = 4.0

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    print(f"\n--- 累积测试: {label} | dim={dim}, batch={batch}, {n_calls} calls ---")

    def old_cumulative():
        for _ in range(n_calls):
            beta_exp = beta_1d.unsqueeze(0).expand(batch, dim).contiguous()
            v_th_exp = v_th_1d.unsqueeze(0).expand(batch, dim).contiguous()
            _TritonPLIFRowParamForward.apply(beta_exp, u, v_th_exp, v_init, alpha)

    def new_cumulative():
        for _ in range(n_calls):
            _TritonPLIFRowParam1DForward.apply(beta_1d, u, v_th_1d, v_init, alpha)

    t_old = bench(old_cumulative, f"OLD: {n_calls}x expand+contiguous", n_warmup=3, n_iter=10)
    t_new = bench(new_cumulative, f"NEW: {n_calls}x 1D kernel", n_warmup=3, n_iter=10)
    saved = t_old - t_new
    speedup = t_old / t_new
    print(f"  {'Cumulative speedup':60s} {speedup:.3f}x")
    print(f"  {'Time saved':60s} {saved:.1f} ms")
    return speedup, saved


def test_correctness(dim, batch, label):
    """验证 1D kernel 数值正确性。"""
    from atomic_ops.parallel_scan import (
        _TritonPLIFRowParamForward, _TritonPLIFRowParam1DForward,
    )
    surr = surrogate.Sigmoid(alpha=4.0)
    alpha = float(surr.alpha)

    torch.manual_seed(42)
    beta_1d = torch.sigmoid(torch.randn(dim, device=device, dtype=dtype))
    v_th_1d = 0.3 + torch.rand(dim, device=device, dtype=dtype) * 0.2
    u = torch.randn(TK, batch, dim, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, dim, device=device, dtype=dtype)

    # OLD
    beta_1d_old = beta_1d.clone().requires_grad_(True)
    v_th_1d_old = v_th_1d.clone().requires_grad_(True)
    u_old = u.clone().requires_grad_(True)
    v_init_old = v_init.clone().requires_grad_(True)
    beta_exp = beta_1d_old.unsqueeze(0).expand(batch, dim).contiguous()
    v_th_exp = v_th_1d_old.unsqueeze(0).expand(batch, dim).contiguous()
    spike_old, vpost_old = _TritonPLIFRowParamForward.apply(
        beta_exp, u_old, v_th_exp, v_init_old, alpha)
    (spike_old.sum() + vpost_old.sum()).backward()

    # NEW
    beta_1d_new = beta_1d.clone().requires_grad_(True)
    v_th_1d_new = v_th_1d.clone().requires_grad_(True)
    u_new = u.clone().requires_grad_(True)
    v_init_new = v_init.clone().requires_grad_(True)
    spike_new, vpost_new = _TritonPLIFRowParam1DForward.apply(
        beta_1d_new, u_new, v_th_1d_new, v_init_new, alpha)
    (spike_new.sum() + vpost_new.sum()).backward()

    all_ok = True
    pairs = [
        ("spike", spike_old, spike_new),
        ("V_post", vpost_old, vpost_new),
        ("grad_u", u_old.grad, u_new.grad),
        ("grad_v_init", v_init_old.grad, v_init_new.grad),
        ("grad_beta", beta_1d_old.grad, beta_1d_new.grad),
        ("grad_v_th", v_th_1d_old.grad, v_th_1d_new.grad),
    ]
    ok_str_all = []
    for name, old, new in pairs:
        if old is None or new is None:
            ok_str_all.append(f"{name}:NONE")
            all_ok = False
            continue
        max_diff = (old - new).abs().max().item()
        rel_diff = max_diff / (old.abs().max().item() + 1e-10)
        ok = max_diff == 0 or rel_diff < 0.02
        ok_str_all.append(f"{name}:{'OK' if ok else 'FAIL'}({rel_diff:.1e})")
        if not ok:
            all_ok = False

    status = "PASS" if all_ok else "FAIL"
    print(f"  正确性 batch={batch}: {status} | {', '.join(ok_str_all)}")
    return all_ok


if __name__ == '__main__':
    D = 768
    D_ff = 2304

    batch_sizes = [2, 4, 8, 16, 32]
    dims = [D, 2 * D_ff]  # 768 and 4608
    dim_labels = ["D=768", "2*D_ff=4608"]

    print("=" * 80)
    print("Multi-scale benchmark: 1D row-param kernel vs expand+contiguous")
    print(f"K={K}, seq={seq_len}, TK={TK}")
    print("=" * 80)

    # Phase 1: Correctness at all scales
    print("\n" + "=" * 80)
    print("Phase 1: 正确性验证")
    print("=" * 80)
    all_correct = True
    for dim, dl in zip(dims, dim_labels):
        for batch in batch_sizes:
            ok = test_correctness(dim, batch, f"{dl}, batch={batch}")
            all_correct = all_correct and ok

    print(f"\n正确性总结: {'ALL PASS' if all_correct else 'SOME FAILED'}")

    if not all_correct:
        print("正确性未通过，跳过性能测试")
        sys.exit(1)

    # Phase 2: Single-call performance at all scales
    print("\n" + "=" * 80)
    print("Phase 2: 单次调用性能对比")
    print("=" * 80)

    results = {}
    for dim, dl in zip(dims, dim_labels):
        for batch in batch_sizes:
            sp_fwd, sp_fb = test_single_call(dim, batch, f"{dl}")
            results[(dl, batch)] = (sp_fwd, sp_fb)

    # Phase 3: Cumulative (simulate full model forward)
    print("\n" + "=" * 80)
    print("Phase 3: 累积性能（模拟完整 forward: 20层 × 8次PLIF = 160次调用）")
    print("=" * 80)

    n_calls = 160  # 20 layers × 8 PLIF calls per layer
    cum_results = {}
    for dim, dl in zip(dims, dim_labels):
        for batch in batch_sizes:
            sp, saved = test_cumulative(dim, batch, n_calls, f"{dl}")
            cum_results[(dl, batch)] = (sp, saved)

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\n{'dim':>12s} {'batch':>6s} {'num_cols':>10s} {'fwd':>8s} {'fwd+bwd':>8s} {'160calls':>9s} {'saved(ms)':>10s}")
    print("-" * 75)
    for dim, dl in zip(dims, dim_labels):
        for batch in batch_sizes:
            nc = batch * dim
            sp_fwd, sp_fb = results[(dl, batch)]
            sp_cum, saved = cum_results[(dl, batch)]
            print(f"{dl:>12s} {batch:>6d} {nc:>10d} {sp_fwd:>7.3f}x {sp_fb:>7.3f}x {sp_cum:>8.3f}x {saved:>9.1f}")
