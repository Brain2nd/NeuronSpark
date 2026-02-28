"""Phase 2 Benchmark: binary_residual torch.compile 融合优化效果。

对比:
  unfused:  fp16_decode → float add → fp16_encode（~20 独立 kernel launch）
  compiled: 内联 decode+add+encode → torch.compile inductor 融合为 2-3 kernel

测试配置: B=3, T=512, D=1024, K=16, TK=8192
"""

import torch
import time
import gc
from atomic_ops.fp16_codec import (
    fp16_encode,
    _binary_residual_naive,   # unfused baseline
    _binary_residual_fwd,     # 内联版（compile 前）
    binary_residual,          # 生产版（compile 后 + STE autograd）
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def _current_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


# ====== 配置 ======
B = 3
T = 512
D = 1024
K = 16
TK = T * K
WARMUP = 30
REPEATS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def _make_spikes():
    x_a = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)
    x_b = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)
    return fp16_encode(x_a, K), fp16_encode(x_b, K)


# torch.compile 版（不含 autograd wrapper，纯 forward）
_compiled = torch.compile(_binary_residual_fwd, backend='inductor', fullgraph=True)

VARIANTS = {
    'unfused (baseline)': _binary_residual_naive,
    'compiled (inductor)': _compiled,
}


def bench_correctness():
    """验证 compiled 结果与 unfused 完全一致。"""
    print("=" * 70)
    print("[0] 正确性验证")
    print("=" * 70)

    spike_a, spike_b = _make_spikes()
    ref = _binary_residual_naive(spike_a, spike_b)

    for name, fn in VARIANTS.items():
        if 'baseline' in name:
            continue
        out = fn(spike_a, spike_b)
        match = torch.equal(ref, out)
        diff = (ref - out).abs().max().item()
        status = "PASS" if match else f"FAIL (max_diff={diff})"
        print(f"  {name:30s} {status}")
    print()


def bench_forward():
    """Forward 速度对比。"""
    print("=" * 70)
    print(f"[1] Forward 速度 — ({TK}, {B}, {D}), {REPEATS} iters")
    print("=" * 70)

    spike_a, spike_b = _make_spikes()
    results = {}

    for name, fn in VARIANTS.items():
        for _ in range(WARMUP):
            fn(spike_a, spike_b)
        _sync()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            fn(spike_a, spike_b)
        _sync()
        t = (time.perf_counter() - t0) / REPEATS * 1000
        results[name] = t

    baseline = list(results.values())[0]
    for name, t in results.items():
        ratio = baseline / t
        bar = "█" * int(ratio * 20)
        print(f"  {name:30s} {t:8.3f} ms  {ratio:5.2f}x  {bar}")
    print()


def bench_fwd_bwd():
    """Forward + Backward 速度对比（含 autograd）。"""
    print("=" * 70)
    print(f"[2] Forward+Backward 速度 — ({TK}, {B}, {D})")
    print("=" * 70)

    x_a = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)
    x_b = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)

    def make_autograd_fn(raw_fn):
        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, sa, sb):
                return raw_fn(sa, sb)
            @staticmethod
            def backward(ctx, g):
                return g, g
        return _F.apply

    results = {}
    for name, fn in VARIANTS.items():
        af = make_autograd_fn(fn)
        for _ in range(WARMUP):
            sa = fp16_encode(x_a, K).requires_grad_(True)
            sb = fp16_encode(x_b, K).requires_grad_(True)
            af(sa, sb).sum().backward()
        _sync()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            sa = fp16_encode(x_a, K).requires_grad_(True)
            sb = fp16_encode(x_b, K).requires_grad_(True)
            af(sa, sb).sum().backward()
        _sync()
        t = (time.perf_counter() - t0) / REPEATS * 1000
        results[name] = t

    baseline = list(results.values())[0]
    for name, t in results.items():
        ratio = baseline / t
        bar = "█" * int(ratio * 20)
        print(f"  {name:30s} {t:8.3f} ms  {ratio:5.2f}x  {bar}")
    print()


def bench_memory():
    """Forward+Backward 显存对比。"""
    print("=" * 70)
    print(f"[3] Forward+Backward 显存 — ({TK}, {B}, {D})")
    print("=" * 70)

    if DEVICE == 'cpu':
        print("  (跳过)")
        print()
        return

    x_a = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)
    x_b = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE).clamp(-100, 100)

    def make_autograd_fn(raw_fn):
        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, sa, sb):
                return raw_fn(sa, sb)
            @staticmethod
            def backward(ctx, g):
                return g, g
        return _F.apply

    results = {}
    for name, fn in VARIANTS.items():
        af = make_autograd_fn(fn)
        for _ in range(WARMUP):
            sa = fp16_encode(x_a, K).requires_grad_(True)
            sb = fp16_encode(x_b, K).requires_grad_(True)
            af(sa, sb).sum().backward()
        _sync()

        _reset_memory()
        baseline_mem = _current_mb()
        sa = fp16_encode(x_a, K).requires_grad_(True)
        sb = fp16_encode(x_b, K).requires_grad_(True)
        af(sa, sb).sum().backward()
        _sync()
        peak = _peak_mb() - baseline_mem
        results[name] = peak
        del sa, sb

    ref_peak = list(results.values())[0]
    for name, peak in results.items():
        delta = peak - ref_peak
        print(f"  {name:30s} {peak:8.1f} MB  ({delta:+.1f} MB)")
    print()


def bench_model_estimate():
    """全模型开销估算 (20 层 × 2 = 40 calls/step)。"""
    print("=" * 70)
    print("[4] 全模型开销估算 (20 层 × 2 = 40 calls / step)")
    print("=" * 70)

    spike_a, spike_b = _make_spikes()
    results = {}

    for name, fn in VARIANTS.items():
        for _ in range(WARMUP):
            fn(spike_a, spike_b)
        _sync()

        t0 = time.perf_counter()
        for _ in range(REPEATS):
            fn(spike_a, spike_b)
        _sync()
        t = (time.perf_counter() - t0) / REPEATS * 1000
        results[name] = t

    n_calls = 40
    for name, t in results.items():
        total_fwd = t * n_calls
        total_fwdbwd = total_fwd * 3.5
        print(f"  {name:30s} {t:.3f} ms/call → fwd {total_fwd:.0f} ms, fwd+bwd ~{total_fwdbwd:.0f} ms")
    print()


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Config: B={B}, T={T}, D={D}, K={K}, TK={TK}")
    print(f"Warmup={WARMUP}, Repeats={REPEATS}")
    print()

    bench_correctness()
    bench_forward()
    bench_fwd_bwd()
    bench_memory()
    bench_model_estimate()

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
