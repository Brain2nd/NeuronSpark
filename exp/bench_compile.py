"""
torch.compile 对 parallel scan 的加速测试
不修改任何现有代码，不影响正在运行的训练
"""
import torch
import time
import sys
sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')

from atomic_ops.parallel_scan import hillis_steele_scan, linear_recurrence, plif_parallel_forward

# 用小显存，不干扰训练
device = 'cuda'
dtype = torch.float32

# 模拟实际规模: K=16, batch*seq=2*512=1024, D=768 中取 N=8 组
K = 16
B_seq = 1024
D = 768

# ============================================================
# Benchmark 1: hillis_steele_scan
# ============================================================
def bench_scan(fn, label, warmup=5, repeat=20):
    a = torch.rand(K, B_seq, D, device=device, dtype=dtype) * 0.5 + 0.3
    b = torch.randn(K, B_seq, D, device=device, dtype=dtype) * 0.1

    # warmup
    for _ in range(warmup):
        A, B = fn(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        A, B = fn(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000

    print(f"  {label:<30} {elapsed:>8.2f} ms")
    return elapsed

print("=" * 55)
print(f"Benchmark: hillis_steele_scan  K={K}, shape=({B_seq},{D})")
print("=" * 55)

t_orig = bench_scan(hillis_steele_scan, "原始")

scan_compiled = torch.compile(hillis_steele_scan)
t_comp = bench_scan(scan_compiled, "torch.compile")

print(f"  加速比: {t_orig/t_comp:.2f}x")

# ============================================================
# Benchmark 2: linear_recurrence
# ============================================================
def bench_linrec(fn, label, warmup=5, repeat=20):
    beta = torch.rand(K, B_seq, D, device=device, dtype=dtype) * 0.5 + 0.3
    u = torch.randn(K, B_seq, D, device=device, dtype=dtype) * 0.1
    v_init = torch.zeros(B_seq, D, device=device, dtype=dtype)

    for _ in range(warmup):
        V = fn(beta, u, v_init)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        V = fn(beta, u, v_init)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000

    print(f"  {label:<30} {elapsed:>8.2f} ms")
    return elapsed

print()
print("=" * 55)
print(f"Benchmark: linear_recurrence  K={K}, shape=({B_seq},{D})")
print("=" * 55)

t_orig = bench_linrec(linear_recurrence, "原始")

linrec_compiled = torch.compile(linear_recurrence)
t_comp = bench_linrec(linrec_compiled, "torch.compile")

print(f"  加速比: {t_orig/t_comp:.2f}x")

# ============================================================
# Benchmark 3: plif_parallel_forward (完整 spike 计算)
# ============================================================
def bench_plif(fn, label, warmup=5, repeat=20):
    from spikingjelly.activation_based import surrogate
    sf = surrogate.Sigmoid(alpha=4.0)

    beta = torch.rand(K, B_seq, D, device=device, dtype=dtype) * 0.5 + 0.3
    u = torch.randn(K, B_seq, D, device=device, dtype=dtype) * 0.1
    v_th = torch.ones(K, B_seq, D, device=device, dtype=dtype) * 0.3
    v_init = torch.zeros(B_seq, D, device=device, dtype=dtype)

    for _ in range(warmup):
        s, vp, vpre = fn(beta, u, v_th, v_init, max_iter=3, surrogate_function=sf)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        s, vp, vpre = fn(beta, u, v_th, v_init, max_iter=3, surrogate_function=sf)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000

    print(f"  {label:<30} {elapsed:>8.2f} ms")
    return elapsed

print()
print("=" * 55)
print(f"Benchmark: plif_parallel_forward  K={K}, shape=({B_seq},{D})")
print("=" * 55)

t_orig = bench_plif(plif_parallel_forward, "原始")

plif_compiled = torch.compile(plif_parallel_forward)
t_comp = bench_plif(plif_compiled, "torch.compile")

print(f"  加速比: {t_orig/t_comp:.2f}x")

# ============================================================
# Benchmark 4: 简单 for 循环对比 (sequential baseline)
# ============================================================
def sequential_recurrence(beta, u, v_init):
    K = beta.shape[0]
    V_list = []
    v = v_init
    for k in range(K):
        v = beta[k] * v + u[k]
        V_list.append(v)
    return torch.stack(V_list, dim=0)

print()
print("=" * 55)
print(f"Benchmark: sequential vs parallel  K={K}, shape=({B_seq},{D})")
print("=" * 55)

t_seq = bench_linrec(sequential_recurrence, "顺序 for 循环")
t_par = bench_linrec(linear_recurrence, "parallel scan")
seq_compiled = torch.compile(sequential_recurrence)
t_seq_comp = bench_linrec(seq_compiled, "顺序 for + compile")

print(f"  parallel/sequential: {t_seq/t_par:.2f}x")
print(f"  compiled seq/原始 seq: {t_seq/t_seq_comp:.2f}x")
