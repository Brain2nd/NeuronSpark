"""
Benchmark: Triton sparse_row_gemm vs cuBLAS F.linear

测试 sparse_row_gemm 在实际稀疏度下是否比 dense F.linear 更快。
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

device = 'cuda'
dtype = torch.bfloat16

from atomic_ops.sparse_gemm import sparse_row_gemm, _HAS_TRITON

def make_spike_data(M, K, row_sparsity, elem_sparsity_in_active, device, dtype):
    """生成模拟 AND 门输出的稀疏 spike 数据。"""
    x = torch.zeros(M, K, device=device, dtype=dtype)
    n_active = int(M * (1 - row_sparsity))
    active_idx = torch.randperm(M, device=device)[:n_active]
    nnz_per_row = max(1, int(K * (1 - elem_sparsity_in_active)))
    for idx in active_idx:
        cols = torch.randperm(K, device=device)[:nnz_per_row]
        x[idx.item(), cols] = 1.0
    return x

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

print("=" * 70)
print("sparse_row_gemm vs F.linear Benchmark")
print("=" * 70)

scenarios = [
    # (name, M, K_in, N_out, row_sparsity, elem_sparsity_active_rows)
    ("SNNFFN AND→down_proj (72% row sparse)", 1024, 2304, 768, 0.72, 0.91),
    ("SNNFFN AND→down_proj (43% row sparse)", 1024, 2304, 768, 0.43, 0.94),
    ("SNNBlock input (43% row sparse)", 1024, 768, 6144, 0.43, 0.60),
    ("Full scale AND (seq=512,K=16,b=2)", 16384, 2304, 768, 0.72, 0.91),
    ("Full scale input (seq=512,K=16,b=2)", 16384, 768, 6144, 0.43, 0.60),
]

for name, M, K_in, N_out, row_sp, elem_sp_active in scenarios:
    print(f"\n{name}")
    print(f"  shape: ({M}, {K_in}) @ ({N_out}, {K_in})^T, row_sparse={row_sp:.0%}")

    torch.manual_seed(42)
    x = make_spike_data(M, K_in, row_sp, elem_sp_active, device, dtype)
    W = torch.randn(N_out, K_in, device=device, dtype=dtype) * 0.01
    row_mask = x.any(dim=-1)
    actual_row_sp = 1 - row_mask.float().mean().item()
    actual_elem_sp = (x == 0).float().mean().item()
    n_active = row_mask.sum().item()
    print(f"  actual: elem={actual_elem_sp:.1%}, row_zero={actual_row_sp:.1%}, active={n_active}/{M}")

    t_dense = bench(lambda: F.linear(x, W))
    t_sparse = bench(lambda: sparse_row_gemm(x, W, row_mask))

    ratio = t_dense / t_sparse
    print(f"  Dense:  {t_dense:.3f} ms")
    print(f"  Sparse: {t_sparse:.3f} ms")
    print(f"  Speedup: {ratio:.2f}x {'(faster)' if ratio > 1 else '(SLOWER)'}")

print()
print("=" * 70)
