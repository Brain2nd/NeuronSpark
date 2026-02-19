"""
Benchmark: torch.compile fused modulation vs separate activations.
Tests the _fused_modulation function in SNNBlock.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import torch.nn.functional as F
import time

device = 'cuda'
dtype = torch.bfloat16

D, N, K, seq_len, batch = 768, 8, 16, 512, 2
TK = seq_len * K
DN = D * N


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench_op(label, fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    t0 = sync_time()
    for _ in range(n_iter):
        fn()
    t = (sync_time() - t0) / n_iter
    print(f"  {label:45s} {t*1000:8.2f}ms")
    return t


print("=" * 70)
print("Fused modulation benchmark")
print(f"TK={TK}, batch={batch}, DN={DN}")
print("=" * 70)

# Setup tensors (simulate matmul outputs)
raw_beta = torch.randn(TK, batch, DN, device=device, dtype=dtype)
raw_alpha = torch.randn(TK, batch, DN, device=device, dtype=dtype)
raw_th = torch.randn(TK, batch, DN, device=device, dtype=dtype)
I_all = torch.randn(TK, batch, DN, device=device, dtype=dtype)
b_beta = torch.randn(DN, device=device, dtype=dtype)
b_alpha = torch.randn(DN, device=device, dtype=dtype)
b_th = torch.randn(DN, device=device, dtype=dtype)
v_th_min = 0.1

print("\n--- Separate ops (baseline) ---")
def separate_ops():
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th

t_separate = bench_op("3 activations + alpha*I (separate)", separate_ops)

print("\n--- torch.compile fused ---")

# Compile the fused function
@torch.compile(backend='inductor', fullgraph=True)
def fused_modulation(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th

# Warm up compilation (first call compiles, may be slow)
print("  [Compiling...]")
t_compile_start = time.perf_counter()
_ = fused_modulation(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all)
torch.cuda.synchronize()
t_compile = time.perf_counter() - t_compile_start
print(f"  [Compilation took {t_compile:.1f}s]")

t_fused = bench_op("3 activations + alpha*I (fused)", lambda: fused_modulation(
    raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all,
))

print(f"\n  Speedup: {t_separate/t_fused:.2f}x")

# Verify correctness
print("\n--- Correctness check ---")
r_sep = separate_ops()
r_fused = fused_modulation(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all)
for i, name in enumerate(['beta', 'u', 'v_th']):
    diff = (r_sep[i] - r_fused[i]).abs().max().item()
    print(f"  {name}: max_diff={diff:.2e} {'OK' if diff < 1e-3 else 'MISMATCH!'}")

# Benchmark with gradients (training scenario)
print("\n--- With gradients (forward+backward) ---")
raw_beta_g = raw_beta.clone().requires_grad_(True)
raw_alpha_g = raw_alpha.clone().requires_grad_(True)
raw_th_g = raw_th.clone().requires_grad_(True)
I_all_g = I_all.clone().requires_grad_(True)
b_beta_g = b_beta.clone().requires_grad_(True)
b_alpha_g = b_alpha.clone().requires_grad_(True)
b_th_g = b_th.clone().requires_grad_(True)

def separate_fwd_bwd():
    beta = torch.sigmoid(raw_beta_g + b_beta_g)
    alpha = F.softplus(raw_alpha_g + b_alpha_g)
    v_th = v_th_min + torch.abs(raw_th_g + b_th_g)
    u = alpha * I_all_g
    (beta.sum() + u.sum() + v_th.sum()).backward()

def fused_fwd_bwd():
    beta, u, v_th = fused_modulation(raw_beta_g, b_beta_g, raw_alpha_g, b_alpha_g,
                                      raw_th_g, b_th_g, v_th_min, I_all_g)
    (beta.sum() + u.sum() + v_th.sum()).backward()

t_sep_fb = bench_op("Separate fwd+bwd", separate_fwd_bwd)
t_fused_fb = bench_op("Fused fwd+bwd", fused_fwd_bwd)
print(f"\n  Speedup (fwd+bwd): {t_sep_fb/t_fused_fb:.2f}x")

print("\nDone.")
