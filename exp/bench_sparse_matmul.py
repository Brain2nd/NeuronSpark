"""
Benchmark: 行稀疏 matmul vs 密集 matmul

测试场景：
1. SNNBlock 输入投影: spike_in (TK*batch, D=768) → W (DN=6144, D=768)
   行稀疏度 ~43%
2. SNNFFN down_proj: gated (TK*batch, D_ff=2304) → W (D=768, D_ff=2304)
   行稀疏度 ~72%, 元素稀疏度 ~97%
3. SNNBlock W_out: s_hidden (TK*batch, DN=6144) → W (D=768, DN=6144)
   行稀疏度 ~30%

方法对比：
A. Dense: F.linear(full_input, W)
B. Row-skip: mask → gather → F.linear(active, W) → scatter
C. torch.sparse.mm (CSR format)
"""

import torch
import torch.nn.functional as F
import time


def make_sparse_spike(rows, cols, elem_sparsity, row_sparsity, device, dtype):
    """生成具有指定稀疏度的二值 spike 张量。"""
    t = torch.zeros(rows, cols, device=device, dtype=dtype)

    # 先确定哪些行有 spike
    n_active_rows = int(rows * (1 - row_sparsity))
    active_idx = torch.randperm(rows, device=device)[:n_active_rows]

    # 在 active 行中随机放置 spike
    # 需要的总 nnz = rows * cols * (1 - elem_sparsity)
    total_nnz = int(rows * cols * (1 - elem_sparsity))
    nnz_per_active_row = max(1, total_nnz // n_active_rows)

    for idx in active_idx:
        col_idx = torch.randperm(cols, device=device)[:nnz_per_active_row]
        t[idx.item(), col_idx] = 1.0

    actual_elem = (t == 0).sum().item() / t.numel()
    actual_row = (~t.any(dim=-1)).sum().item() / rows
    return t, actual_elem, actual_row


def bench_dense(x, W, warmup=5, iters=50):
    """Dense matmul baseline."""
    for _ in range(warmup):
        _ = F.linear(x, W)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = F.linear(x, W)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_rowskip(x, W, warmup=5, iters=50):
    """Row-skip: 只对非零行做 matmul。"""
    out_dim = W.shape[0]
    for _ in range(warmup):
        mask = x.any(dim=-1)
        active = x[mask]
        active_out = F.linear(active, W)
        output = torch.zeros(x.shape[0], out_dim, device=x.device, dtype=x.dtype)
        output[mask] = active_out
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        mask = x.any(dim=-1)
        active = x[mask]
        active_out = F.linear(active, W)
        output = torch.zeros(x.shape[0], out_dim, device=x.device, dtype=x.dtype)
        output[mask] = active_out
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_rowskip_precomputed(x, mask, W, warmup=5, iters=50):
    """Row-skip with precomputed mask (mask 已知的场景)."""
    out_dim = W.shape[0]
    active = x[mask]
    for _ in range(warmup):
        active_out = F.linear(active, W)
        output = torch.zeros(x.shape[0], out_dim, device=x.device, dtype=x.dtype)
        output[mask] = active_out
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        active = x[mask]
        active_out = F.linear(active, W)
        output = torch.zeros(x.shape[0], out_dim, device=x.device, dtype=x.dtype)
        output[mask] = active_out
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_sparse_csr(x, W, warmup=5, iters=50):
    """CSR sparse matmul."""
    out_dim = W.shape[0]
    x_csr = x.to_sparse_csr()
    W_t = W.t().contiguous()  # sparse.mm needs (sparse, dense)
    for _ in range(warmup):
        out = torch.sparse.mm(x_csr, W_t)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        x_csr = x.to_sparse_csr()
        out = torch.sparse.mm(x_csr, W_t)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def run_scenario(name, rows, in_dim, out_dim, elem_sp, row_sp, device, dtype):
    """运行一个测试场景。"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"  shape: ({rows}, {in_dim}) × ({out_dim}, {in_dim})")
    print(f"  target: elem_sparse={elem_sp:.0%}, row_sparse={row_sp:.0%}")

    W = torch.randn(out_dim, in_dim, device=device, dtype=dtype) * 0.01
    x, actual_elem, actual_row = make_sparse_spike(rows, in_dim, elem_sp, row_sp, device, dtype)
    print(f"  actual: elem_sparse={actual_elem:.1%}, row_sparse={actual_row:.1%}")
    print(f"  active_rows={int(rows*(1-actual_row))}, total_nnz={int((1-actual_elem)*rows*in_dim)}")

    mask = x.any(dim=-1)

    t_dense = bench_dense(x, W)
    t_rowskip = bench_rowskip(x, W)
    t_rowskip_pre = bench_rowskip_precomputed(x, mask, W)

    try:
        x_fp32 = x.float()
        W_fp32 = W.float()
        t_csr = bench_sparse_csr(x_fp32, W_fp32)
        csr_ok = True
    except Exception as e:
        t_csr = float('nan')
        csr_ok = False
        print(f"  CSR failed: {e}")

    print(f"\n  方法              时间(ms)   加速比")
    print(f"  {'─'*45}")
    print(f"  Dense             {t_dense:7.2f}      1.00x")
    print(f"  Row-skip          {t_rowskip:7.2f}      {t_dense/t_rowskip:.2f}x")
    print(f"  Row-skip(premask) {t_rowskip_pre:7.2f}      {t_dense/t_rowskip_pre:.2f}x")
    if csr_ok:
        print(f"  CSR sparse.mm     {t_csr:7.2f}      {t_dense/t_csr:.2f}x")

    # 显存对比
    dense_mem = rows * in_dim * 2  # bf16
    sparse_nnz = int((1 - actual_elem) * rows * in_dim)
    csr_mem = sparse_nnz * 6 + rows * 4  # value(2B) + col_idx(4B) + row_ptr
    print(f"\n  显存: dense={dense_mem/1e6:.1f}MB  CSR≈{csr_mem/1e6:.1f}MB  ({dense_mem/csr_mem:.1f}x)")


def main():
    device = 'cuda'
    dtype = torch.bfloat16

    print("Sparse Matmul Benchmark — GB10")
    print(f"dtype: {dtype}")

    TK_batch = 512 * 2  # TK=512(32*16), batch=2

    # 场景 1: SNNBlock 输入投影 (×6)
    run_scenario(
        "场景1: SNNBlock 输入投影 (spike_in @ W, ×6 matmul)",
        rows=TK_batch, in_dim=768, out_dim=6144,
        elem_sp=0.80, row_sp=0.43,
        device=device, dtype=dtype,
    )

    # 场景 2: SNNFFN down_proj (AND 门输出)
    run_scenario(
        "场景2: SNNFFN down_proj (gated_AND @ W)",
        rows=TK_batch, in_dim=2304, out_dim=768,
        elem_sp=0.97, row_sp=0.72,
        device=device, dtype=dtype,
    )

    # 场景 3: SNNBlock W_out (hidden spike)
    run_scenario(
        "场景3: SNNBlock W_out (s_hidden @ W)",
        rows=TK_batch, in_dim=6144, out_dim=768,
        elem_sp=0.80, row_sp=0.30,
        device=device, dtype=dtype,
    )

    # 场景 4: 全规模 — seq_len=512
    TK_batch_full = 512 * 16 * 2  # 8192*2=16384
    run_scenario(
        "场景4: 全规模 SNNBlock 输入投影 (seq=512, K=16)",
        rows=TK_batch_full, in_dim=768, out_dim=6144,
        elem_sp=0.80, row_sp=0.43,
        device=device, dtype=dtype,
    )

    run_scenario(
        "场景5: 全规模 SNNFFN down_proj (seq=512, K=16)",
        rows=TK_batch_full, in_dim=2304, out_dim=768,
        elem_sp=0.97, row_sp=0.72,
        device=device, dtype=dtype,
    )


if __name__ == '__main__':
    main()
