"""
三种 linear_recurrence 实现的性能对比:
  1. Legacy (Hillis-Steele, Python while + torch.cat)
  2. AccScan (accelerated-scan Triton kernel)
  3. Sequential (简单 for 循环展开)
"""
import torch
import time
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

from atomic_ops.parallel_scan import (
    _linear_recurrence_legacy,
    _linear_recurrence_acc,
    _linear_recurrence_sequential,
    plif_parallel_forward,
)
import atomic_ops.parallel_scan as ps
from spikingjelly.activation_based import surrogate

device = 'cuda'
sf = surrogate.Sigmoid(alpha=4.0)

def bench(fn, args, warmup=10, repeat=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000

# ============================================================
# Benchmark 1: linear_recurrence 单次调用
# ============================================================
print("=" * 85)
print("linear_recurrence 单次调用 (ms)")
print("=" * 85)
print(f"{'K':>4} {'B':>4} {'D':>6} | {'Legacy':>9} {'AccScan':>9} {'SeqLoop':>9} | {'best':>8} {'vs Leg':>8}")
print("-" * 85)

for K, B, D in [(8,1,512), (8,4,512), (8,8,512), (8,8,1536), (8,32,512)]:
    beta = torch.sigmoid(torch.randn(K, B, D, device=device))
    u = torch.randn(K, B, D, device=device)
    vi = torch.randn(B, D, device=device)

    t_leg = bench(_linear_recurrence_legacy, (beta, u, vi))
    t_acc = bench(_linear_recurrence_acc, (beta, u, vi))
    t_seq = bench(_linear_recurrence_sequential, (beta, u, vi))

    best = min(t_leg, t_acc, t_seq)
    best_name = ['Legacy', 'AccScan', 'SeqLoop'][[t_leg, t_acc, t_seq].index(best)]
    speedup = t_leg / best
    print(f"{K:4d} {B:4d} {D:6d} | {t_leg:7.3f}ms {t_acc:7.3f}ms {t_seq:7.3f}ms | {best_name:>8} {speedup:6.2f}x")

print()

# ============================================================
# Benchmark 2: plif_parallel_forward (fwd+bwd, 模拟单层训练)
# ============================================================
print("=" * 85)
print("plif_parallel_forward forward+backward (ms, 模拟单层训练)")
print("=" * 85)
print(f"{'K':>4} {'B':>4} {'D':>6} | {'Legacy':>9} {'AccScan':>9} {'SeqLoop':>9} | {'best':>8} {'vs Leg':>8}")
print("-" * 85)

# 保存原始 linear_recurrence
_orig_lr = ps.linear_recurrence

def make_plif_fn(lr_fn):
    """返回一个用指定 linear_recurrence 的 plif forward+backward 函数"""
    def fn(bl, u, vtl, vi):
        ps.linear_recurrence = lr_fn
        b = torch.sigmoid(bl)
        vt = torch.abs(vtl) + 0.5
        s, vp, _ = plif_parallel_forward(b, u, vt, vi, 3, sf)
        (s.sum() + vp.sum()).backward()
    return fn

for K, B, D in [(8,1,512), (8,4,512), (8,8,512), (8,8,1536), (8,32,512)]:
    bl = torch.randn(K, B, D, device=device, requires_grad=True)
    u = torch.randn(K, B, D, device=device, requires_grad=True)
    vtl = torch.randn(K, B, D, device=device, requires_grad=True)
    vi = torch.randn(B, D, device=device, requires_grad=True)

    t_leg = bench(make_plif_fn(_linear_recurrence_legacy), (bl, u, vtl, vi), warmup=5, repeat=50)
    t_acc = bench(make_plif_fn(_linear_recurrence_acc), (bl, u, vtl, vi), warmup=5, repeat=50)
    t_seq = bench(make_plif_fn(_linear_recurrence_sequential), (bl, u, vtl, vi), warmup=5, repeat=50)

    best = min(t_leg, t_acc, t_seq)
    best_name = ['Legacy', 'AccScan', 'SeqLoop'][[t_leg, t_acc, t_seq].index(best)]
    speedup = t_leg / best
    print(f"{K:4d} {B:4d} {D:6d} | {t_leg:7.3f}ms {t_acc:7.3f}ms {t_seq:7.3f}ms | {best_name:>8} {speedup:6.2f}x")

# 恢复
ps.linear_recurrence = _orig_lr
