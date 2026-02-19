"""
Benchmark: Triton BLOCK size tuning for fused PLIF kernels.
Find optimal BLOCK and num_warps for each tensor shape.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import time
import triton

from atomic_ops.parallel_scan import (
    _fused_plif_fwd_kernel,
    _fused_plif_bwd_kernel,
    _fwd_recurrence_kernel,
    _bwd_recurrence_kernel,
)

device = 'cuda'


def bench_fwd_kernel(K, batch, D, dtype, BLOCK, num_warps, n_warmup=3, n_iter=15):
    """Benchmark fused PLIF forward kernel with specific BLOCK/num_warps."""
    torch.manual_seed(42)
    num_cols = batch * D
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype)).contiguous()
    u = (torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1).contiguous()
    v_th = (0.1 + torch.abs(torch.randn(K, batch, D, device=device, dtype=dtype)) * 0.5).contiguous()
    v_init = torch.zeros(batch, D, device=device, dtype=dtype).contiguous()
    spike = torch.empty_like(u)
    V_post = torch.empty_like(u)

    grid = ((num_cols + BLOCK - 1) // BLOCK,)
    n_programs = grid[0]

    try:
        for _ in range(n_warmup):
            _fused_plif_fwd_kernel[grid](
                beta, u, v_th, v_init, spike, V_post,
                K, num_cols, BLOCK=BLOCK, num_warps=num_warps,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _fused_plif_fwd_kernel[grid](
                beta, u, v_th, v_init, spike, V_post,
                K, num_cols, BLOCK=BLOCK, num_warps=num_warps,
            )
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / n_iter * 1000
        return t, n_programs
    except Exception as e:
        return None, n_programs


def bench_bwd_kernel(K, batch, D, dtype, BLOCK, num_warps, n_warmup=3, n_iter=15):
    """Benchmark fused PLIF backward kernel with specific BLOCK/num_warps."""
    torch.manual_seed(42)
    num_cols = batch * D
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype)).contiguous()
    v_th = (0.1 + torch.abs(torch.randn(K, batch, D, device=device, dtype=dtype)) * 0.5).contiguous()
    v_init = torch.zeros(batch, D, device=device, dtype=dtype).contiguous()
    V_post = (torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1).contiguous()
    spike = torch.where(torch.rand(K, batch, D, device=device) > 0.8, 1.0, 0.0).to(dtype).contiguous()
    grad_spike = torch.randn_like(spike).contiguous()
    grad_V_post = torch.randn_like(V_post).contiguous()
    grad_beta = torch.empty_like(beta)
    grad_u = torch.empty_like(beta)
    grad_v_th = torch.empty_like(v_th)
    grad_v_init = torch.empty_like(v_init)

    grid = ((num_cols + BLOCK - 1) // BLOCK,)

    try:
        for _ in range(n_warmup):
            _fused_plif_bwd_kernel[grid](
                beta, v_th, v_init, V_post, spike,
                grad_spike, grad_V_post,
                grad_beta, grad_u, grad_v_th, grad_v_init,
                K, num_cols, 4.0,
                BLOCK=BLOCK, num_warps=num_warps,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _fused_plif_bwd_kernel[grid](
                beta, v_th, v_init, V_post, spike,
                grad_spike, grad_V_post,
                grad_beta, grad_u, grad_v_th, grad_v_init,
                K, num_cols, 4.0,
                BLOCK=BLOCK, num_warps=num_warps,
            )
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / n_iter * 1000
        return t, grid[0]
    except Exception as e:
        return None, grid[0]


# Test configurations
block_configs = [
    (16, 1), (16, 2),
    (32, 1), (32, 2), (32, 4),
    (64, 2), (64, 4),
    (128, 4), (128, 8),
    (256, 4), (256, 8),
    (512, 8),
]

shapes = [
    ("hidden_DN=6144", 8192, 2, 6144, torch.bfloat16),
    ("output_D=768",   8192, 2, 768,  torch.bfloat16),
    ("ffn_D_ff=2304",  8192, 2, 2304, torch.bfloat16),
]

print("=" * 110)
print("Fused PLIF FORWARD kernel — BLOCK × num_warps tuning")
print("=" * 110)
for label, K, batch, D, dtype in shapes:
    print(f"\n  {label}  (K={K}, cols={batch*D}):")
    best_t = float('inf')
    best_cfg = None
    for BLOCK, nw in block_configs:
        t, n_prog = bench_fwd_kernel(K, batch, D, dtype, BLOCK, nw)
        if t is not None:
            marker = ""
            if t < best_t:
                best_t = t
                best_cfg = (BLOCK, nw)
                marker = " ★"
            print(f"    BLOCK={BLOCK:4d} warps={nw:2d}  programs={n_prog:5d}  {t:7.2f}ms{marker}")
        else:
            print(f"    BLOCK={BLOCK:4d} warps={nw:2d}  FAILED")
    print(f"    → Best: BLOCK={best_cfg[0]}, warps={best_cfg[1]}, {best_t:.2f}ms")

print()
print("=" * 110)
print("Fused PLIF BACKWARD kernel — BLOCK × num_warps tuning")
print("=" * 110)
for label, K, batch, D, dtype in shapes:
    print(f"\n  {label}  (K={K}, cols={batch*D}):")
    best_t = float('inf')
    best_cfg = None
    for BLOCK, nw in block_configs:
        t, n_prog = bench_bwd_kernel(K, batch, D, dtype, BLOCK, nw)
        if t is not None:
            marker = ""
            if t < best_t:
                best_t = t
                best_cfg = (BLOCK, nw)
                marker = " ★"
            print(f"    BLOCK={BLOCK:4d} warps={nw:2d}  programs={n_prog:5d}  {t:7.2f}ms{marker}")
        else:
            print(f"    BLOCK={BLOCK:4d} warps={nw:2d}  FAILED")
    print(f"    → Best: BLOCK={best_cfg[0]}, warps={best_cfg[1]}, {best_t:.2f}ms")

print()
print("Done.")
