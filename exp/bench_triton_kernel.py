"""
Triton fused kernel vs Hillis-Steele: kernel 级别性能对比。
使用真实训练的张量形状，不加载模型，不会 OOM。
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import time

from atomic_ops.parallel_scan import (
    hillis_steele_scan,
    linear_recurrence,
    _HAS_TRITON,
)

device = 'cuda'
assert _HAS_TRITON

def bench_linear_recurrence(K, batch, D, dtype, n_warmup=5, n_iter=20, label=""):
    """Benchmark linear_recurrence: Triton vs Hillis-Steele."""
    torch.manual_seed(42)
    beta = torch.sigmoid(torch.randn(K, batch, D, device=device, dtype=dtype))
    u = torch.randn(K, batch, D, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(batch, D, device=device, dtype=dtype)

    # ---- Triton fused kernel ----
    for _ in range(n_warmup):
        V = linear_recurrence(beta, u, v_init)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        V = linear_recurrence(beta, u, v_init)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / n_iter

    # ---- Hillis-Steele reference ----
    def hs_recurrence(beta, u, v_init):
        A, B = hillis_steele_scan(beta, u)
        return A * v_init.unsqueeze(0) + B

    for _ in range(n_warmup):
        V = hs_recurrence(beta, u, v_init)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        V = hs_recurrence(beta, u, v_init)
    torch.cuda.synchronize()
    t_hs = (time.perf_counter() - t0) / n_iter

    speedup = t_hs / t_triton
    tensor_mb = K * batch * D * (2 if dtype == torch.bfloat16 else 4) / 1e6
    print(f"  {label:25s}  shape=({K:5d},{batch},{D:5d})  tensor={tensor_mb:6.1f}MB"
          f"  Triton={t_triton*1000:7.2f}ms  H-S={t_hs*1000:7.2f}ms  speedup={speedup:.2f}x")

def bench_with_backward(K, batch, D, dtype, n_warmup=3, n_iter=10, label=""):
    """Benchmark forward + backward."""
    torch.manual_seed(42)
    beta_logits = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    u = torch.randn(K, batch, D, device=device, dtype=torch.float32, requires_grad=True)
    v_init = torch.zeros(batch, D, device=device, dtype=torch.float32, requires_grad=True)

    # ---- Triton fwd+bwd ----
    for _ in range(n_warmup):
        beta = torch.sigmoid(beta_logits)
        V = linear_recurrence(beta, u, v_init)
        V.sum().backward()
        beta_logits.grad = None; u.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        beta = torch.sigmoid(beta_logits)
        V = linear_recurrence(beta, u, v_init)
        V.sum().backward()
        beta_logits.grad = None; u.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / n_iter

    # ---- Hillis-Steele fwd+bwd ----
    def hs_recurrence(beta, u, v_init):
        A, B = hillis_steele_scan(beta, u)
        return A * v_init.unsqueeze(0) + B

    for _ in range(n_warmup):
        beta = torch.sigmoid(beta_logits)
        V = hs_recurrence(beta, u, v_init)
        V.sum().backward()
        beta_logits.grad = None; u.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        beta = torch.sigmoid(beta_logits)
        V = hs_recurrence(beta, u, v_init)
        V.sum().backward()
        beta_logits.grad = None; u.grad = None; v_init.grad = None
    torch.cuda.synchronize()
    t_hs = (time.perf_counter() - t0) / n_iter

    speedup = t_hs / t_triton
    print(f"  {label:25s}  shape=({K:5d},{batch},{D:5d})"
          f"  Triton={t_triton*1000:7.2f}ms  H-S={t_hs*1000:7.2f}ms  speedup={speedup:.2f}x")


print("=" * 90)
print("Forward only (bf16) — linear_recurrence")
print("=" * 90)
# Real training shapes
bench_linear_recurrence(8192, 2, 6144, torch.bfloat16, label="hidden (DN=6144)")
bench_linear_recurrence(8192, 2, 768,  torch.bfloat16, label="output (D=768)")
bench_linear_recurrence(8192, 2, 2304, torch.bfloat16, label="ffn_mid (D_ff=2304)")

# Varying K
bench_linear_recurrence(512,  2, 6144, torch.bfloat16, label="K=512 hidden")
bench_linear_recurrence(1024, 2, 6144, torch.bfloat16, label="K=1024 hidden")
bench_linear_recurrence(4096, 2, 6144, torch.bfloat16, label="K=4096 hidden")

print()
print("=" * 90)
print("Forward + Backward (fp32) — linear_recurrence + .backward()")
print("=" * 90)
# Use smaller D for fp32 to avoid OOM (fp32 = 2x memory)
bench_with_backward(8192, 2, 768,  torch.float32, label="output (D=768)")
bench_with_backward(8192, 2, 2304, torch.float32, label="ffn_mid (D_ff=2304)")

# Note: (8192, 2, 6144) in fp32 would be ~800MB per tensor — might be tight with backward
# The real training uses bf16, so fp32 benchmark at full size is less relevant
print()
print("Done.")
