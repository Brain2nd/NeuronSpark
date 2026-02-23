"""
PLIF Scan 微基准: 隔离对比 TK=8192 顺序扫描 vs K=16 Per-Token 扫描
不含 gradient checkpointing，直接测量 PLIF kernel 的加速效果
"""

import time
import torch
from atomic_ops.parallel_scan import plif_rowparam_forward_alpha, plif_rowparam_forward_recompute


def bench_plif(label, fn, beta_row, u, v_th_row, v_init, alpha, n_iters=100, warmup=20):
    """Benchmark a PLIF forward+backward call."""
    # Warmup
    for _ in range(warmup):
        u_t = u.detach().requires_grad_(True)
        spike, _ = fn(beta_row, u_t, v_th_row, v_init, alpha)
        spike.sum().backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    bwd_times = []

    for _ in range(n_iters):
        u_t = u.detach().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spike, _ = fn(beta_row, u_t, v_th_row, v_init, alpha)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        spike.sum().backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)

    peak_mem = torch.cuda.max_memory_allocated()

    def tm(vals, pct=0.1):
        n = len(vals)
        k = int(n * pct)
        s = sorted(vals)
        return sum(s[k:-k]) / (n - 2 * k) * 1000

    fwd = tm(fwd_times)
    bwd = tm(bwd_times)
    print(f"  {label:<45}  fwd: {fwd:7.3f}ms  bwd: {bwd:7.3f}ms  total: {fwd+bwd:7.3f}ms  mem: {peak_mem/1024**2:.1f}MB")
    return fwd, bwd, peak_mem


def main():
    torch.manual_seed(42)
    E = 4          # experts
    T = 512        # tokens
    B = 4          # batch
    K = 16         # SNN steps per token
    D = 256        # dim (2*D_ff for GU)
    TK = T * K     # 8192
    alpha = 4.0
    n_iters = 100

    print(f"Config: E={E}, T={T}, B={B}, K={K}, D={D}, TK={TK}")
    print(f"Iterations: {n_iters}\n")

    # ============================================================
    # 场景 1: MoE PLIF (routed experts)
    # ============================================================
    print("=" * 100)
    print("  MoE Routed Expert PLIF Scan")
    print("=" * 100)

    # OLD: TK=8192 顺序扫描, rows=E*B=16
    beta_old = torch.rand(E * B, D, device='cuda').sigmoid()
    u_old = torch.randn(TK, E * B, D, device='cuda')
    vth_old = torch.rand(E * B, D, device='cuda') * 0.3
    vinit_old = torch.zeros(E * B, D, device='cuda')

    fwd_old, bwd_old, mem_old = bench_plif(
        "OLD: plif_alpha  K=8192, rows=16",
        plif_rowparam_forward_alpha,
        beta_old, u_old, vth_old, vinit_old, alpha, n_iters)

    torch.cuda.empty_cache()

    # NEW: K=16 per-token, rows=E*T*B=8192
    beta_new = torch.rand(E * T * B, D, device='cuda').sigmoid()
    u_new = torch.randn(K, E * T * B, D, device='cuda')
    vth_new = torch.rand(E * T * B, D, device='cuda') * 0.3
    vinit_new = torch.zeros(E * T * B, D, device='cuda')

    fwd_new, bwd_new, mem_new = bench_plif(
        "NEW: plif_recompute  K=16, rows=8192",
        plif_rowparam_forward_recompute,
        beta_new, u_new, vth_new, vinit_new, alpha, n_iters)

    torch.cuda.empty_cache()

    # NEW with standard kernel (isolate reshape effect from recompute effect)
    fwd_new2, bwd_new2, mem_new2 = bench_plif(
        "NEW (alpha kernel): plif_alpha  K=16, rows=8192",
        plif_rowparam_forward_alpha,
        beta_new, u_new, vth_new, vinit_new, alpha, n_iters)

    print()
    print(f"  Per-Token Reshape speedup (alpha kernel):")
    print(f"    Forward:  {fwd_old/fwd_new2:.2f}x")
    print(f"    Backward: {bwd_old/bwd_new2:.2f}x")
    print(f"    Total:    {(fwd_old+bwd_old)/(fwd_new2+bwd_new2):.2f}x")
    print()
    print(f"  Recompute kernel speedup (vs alpha kernel, same K=16 shape):")
    print(f"    Forward:  {fwd_new2/fwd_new:.2f}x")
    print(f"    Backward: {bwd_new2/bwd_new:.2f}x")
    print(f"    Total:    {(fwd_new2+bwd_new2)/(fwd_new+bwd_new):.2f}x")
    print(f"    Memory:   {mem_new2/1024**2:.1f} MB → {mem_new/1024**2:.1f} MB ({(mem_new-mem_new2)/1024**2:+.1f} MB)")
    print()
    print(f"  Combined (OLD → NEW recompute):")
    print(f"    Forward:  {fwd_old/fwd_new:.2f}x  ({fwd_old:.3f} → {fwd_new:.3f} ms)")
    print(f"    Backward: {bwd_old/bwd_new:.2f}x  ({bwd_old:.3f} → {bwd_new:.3f} ms)")
    print(f"    Total:    {(fwd_old+bwd_old)/(fwd_new+bwd_new):.2f}x  ({fwd_old+bwd_old:.3f} → {fwd_new+bwd_new:.3f} ms)")

    # ============================================================
    # 场景 2: Grid block 利用率分析
    # ============================================================
    print("\n" + "=" * 100)
    print("  GPU Grid Block Analysis")
    print("=" * 100)
    BLOCK = 128
    num_cols_old = E * B * D  # 16 * 256 = 4096
    num_cols_new = E * T * B * D  # 8192 * 256 = 2097152
    grid_old = (num_cols_old + BLOCK - 1) // BLOCK
    grid_new = (num_cols_new + BLOCK - 1) // BLOCK
    print(f"  OLD: num_cols={num_cols_old:,}, grid={grid_old:,} blocks, seq_steps={TK}")
    print(f"  NEW: num_cols={num_cols_new:,}, grid={grid_new:,} blocks, seq_steps={K}")
    print(f"  Grid ratio: {grid_new/grid_old:.0f}x more parallel blocks")
    print(f"  Step ratio: {TK/K:.0f}x fewer sequential steps")

    # ============================================================
    # 场景 3: 完整 MoE _batched_expert_forward (含 GEMM + reshape)
    # ============================================================
    print("\n" + "=" * 100)
    print("  Full MoE _batched_expert_forward (GEMM + PLIF + reshape)")
    print("=" * 100)

    from model import SNNLanguageModel
    from spikingjelly.activation_based import functional
    import types

    # Build model
    cfg = dict(
        vocab_size=6144, D=256, N=4, K=16, num_layers=1, D_ff=768,
        use_moe=True, num_experts=4, top_k=2,
        D_ff_shared=256, D_ff_expert=128,
    )
    torch.manual_seed(42)
    model = SNNLanguageModel(**cfg).cuda()

    # Get the MoE FFN from layer 0
    moe = model.layers[0].snn_ffn

    # Create realistic spike input
    spike_in = torch.randint(0, 2, (TK, B, 256), device='cuda', dtype=torch.bfloat16).float()

    # Import old method
    from bench_moe_opt import _batched_expert_forward_OLD

    # Benchmark OLD _batched_expert_forward
    def bench_moe_method(label, method, n=100, warmup=20):
        for _ in range(warmup):
            functional.reset_net(moe)
            out = method(spike_in)
            out.sum().backward()
            for p in moe.parameters():
                if p.grad is not None:
                    p.grad = None

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fwd_t, bwd_t = [], []
        for _ in range(n):
            functional.reset_net(moe)
            for p in moe.parameters():
                if p.grad is not None:
                    p.grad = None

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = method(spike_in)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out.sum().backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            fwd_t.append(t1 - t0)
            bwd_t.append(t2 - t1)

        peak = torch.cuda.max_memory_allocated()
        def tm(v):
            n = len(v); k = int(n*0.1); s = sorted(v)
            return sum(s[k:-k])/(n-2*k)*1000

        f, b = tm(fwd_t), tm(bwd_t)
        print(f"  {label:<45}  fwd: {f:7.3f}ms  bwd: {b:7.3f}ms  total: {f+b:7.3f}ms  mem: {peak/1024**2:.1f}MB")
        return f, b, peak

    # OLD
    old_method = types.MethodType(_batched_expert_forward_OLD, moe)
    f_old, b_old, m_old = bench_moe_method("OLD: _batched_expert_forward", old_method)

    torch.cuda.empty_cache()

    # NEW (default)
    f_new, b_new, m_new = bench_moe_method("NEW: _batched_expert_forward", moe._batched_expert_forward)

    print()
    print(f"  Speedup:")
    print(f"    Forward:  {f_old/f_new:.2f}x  ({f_old:.3f} → {f_new:.3f} ms)")
    print(f"    Backward: {b_old/b_new:.2f}x  ({b_old:.3f} → {b_new:.3f} ms)")
    print(f"    Total:    {(f_old+b_old)/(f_new+b_new):.2f}x  ({f_old+b_old:.3f} → {f_new+b_new:.3f} ms)")
    print(f"    Memory:   {m_old/1024**2:.1f} → {m_new/1024**2:.1f} MB ({(m_new-m_old)/1024**2:+.1f} MB)")


if __name__ == '__main__':
    main()
