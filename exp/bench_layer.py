"""
Layer-level benchmark: SNNBlock + SNNFFN forward + backward.
Measures total time per layer with real training shapes.
"""
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

import torch
import time

device = 'cuda'
dtype = torch.bfloat16


def bench_snn_block(D, N, K, seq_len, batch, n_warmup=3, n_iter=10):
    """Benchmark SNNBlock forward + backward."""
    from atomic_ops.snn_block import SNNBlock
    from spikingjelly.activation_based import functional

    TK = seq_len * K
    block = SNNBlock(D=D, N=N).to(device).to(dtype)

    spike_in = torch.randint(0, 2, (TK, batch, D), device=device, dtype=dtype)

    # Forward
    for _ in range(n_warmup):
        functional.reset_net(block)
        out = block.forward_parallel(spike_in)
        out.sum().backward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        functional.reset_net(block)
        out = block.forward_parallel(spike_in)
        out.sum().backward()
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t0) / n_iter

    # Forward only
    for _ in range(n_warmup):
        functional.reset_net(block)
        with torch.no_grad():
            out = block.forward_parallel(spike_in)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        functional.reset_net(block)
        with torch.no_grad():
            out = block.forward_parallel(spike_in)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) / n_iter

    return t_fwd, t_total


def bench_snn_ffn(D, D_ff, K, seq_len, batch, n_warmup=3, n_iter=10):
    """Benchmark SNNFFN forward + backward."""
    from atomic_ops.snn_ffn import SNNFFN
    from spikingjelly.activation_based import functional

    TK = seq_len * K
    ffn = SNNFFN(D=D, D_ff=D_ff).to(device).to(dtype)

    spike_in = torch.randint(0, 2, (TK, batch, D), device=device, dtype=dtype)

    # Forward + backward
    for _ in range(n_warmup):
        functional.reset_net(ffn)
        out = ffn.forward_parallel(spike_in)
        out.sum().backward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        functional.reset_net(ffn)
        out = ffn.forward_parallel(spike_in)
        out.sum().backward()
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t0) / n_iter

    # Forward only
    for _ in range(n_warmup):
        functional.reset_net(ffn)
        with torch.no_grad():
            out = ffn.forward_parallel(spike_in)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        functional.reset_net(ffn)
        with torch.no_grad():
            out = ffn.forward_parallel(spike_in)
    torch.cuda.synchronize()
    t_fwd = (time.perf_counter() - t0) / n_iter

    return t_fwd, t_total


# Real training config
D = 768
N = 8
K = 16
seq_len = 512
batch = 2
D_ff = 2304

print("=" * 80)
print(f"Layer-level benchmark (D={D}, N={N}, K={K}, seq_len={seq_len}, batch={batch})")
print(f"TK = {seq_len*K}, DN = {D*N}, D_ff = {D_ff}")
print("=" * 80)

print("\nSNNBlock:")
t_fwd, t_total = bench_snn_block(D, N, K, seq_len, batch)
print(f"  Forward:  {t_fwd*1000:.1f}ms")
print(f"  Fwd+Bwd:  {t_total*1000:.1f}ms")
print(f"  Backward: {(t_total - t_fwd)*1000:.1f}ms")

print("\nSNNFFN:")
t_fwd, t_total = bench_snn_ffn(D, D_ff, K, seq_len, batch)
print(f"  Forward:  {t_fwd*1000:.1f}ms")
print(f"  Fwd+Bwd:  {t_total*1000:.1f}ms")
print(f"  Backward: {(t_total - t_fwd)*1000:.1f}ms")

print(f"\nPer-layer total: {(bench_snn_block(D, N, K, seq_len, batch)[1] + bench_snn_ffn(D, D_ff, K, seq_len, batch)[1])*1000:.1f}ms")
print(f"20-layer estimate (fwd+bwd, no checkpoint): {(bench_snn_block(D, N, K, seq_len, batch)[1] + bench_snn_ffn(D, D_ff, K, seq_len, batch)[1])*20*1000:.1f}ms")

print("\nDone.")
