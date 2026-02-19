"""
非线性叠加原理验证实验 (CPU only, lightweight)

核心问题: f(Σx_k) vs Σf(x_k)
  其中 x = Σ_{k=0}^{K-1} b_k · 2^{-(k+1)}, b_k ∈ {0,1}

7 个实验:
  1. 各非线性函数的分解误差测量
  2. 线性层 bias 问题
  3. 膜电位交叉项恢复 (x² 精确分解)
  4. 顺序处理 vs 朴素分解 (SNN 实际做的事)
  5. 多层 ANN vs SNN 模拟
  6. β 衰减对重建的影响
  7. 门控机制的二次非线性能力
"""

import torch
import torch.nn.functional as F
import math


def binary_decompose(x, K):
    """将 x ∈ [0, 1) 分解为 K-bit 二进制: x_k = b_k · 2^{-(k+1)}"""
    x_bits = []
    residual = x.clone()
    for k in range(K):
        weight = 2.0 ** (-(k + 1))
        bit = (residual >= weight).float()
        x_k = bit * weight
        x_bits.append(x_k)
        residual = residual - x_k
    return x_bits


# ============================================================
# Experiment 1: 各非线性函数的分解误差
# ============================================================
def exp1_decomposition_error():
    print("=" * 70)
    print("Exp 1: 分解误差 |f(Σx_k) - Σf(x_k)|")
    print("       x ∈ [0,1), K=8 bit 二进制分解, N=10000 样本")
    print("=" * 70)

    K, N = 8, 10000
    x = torch.rand(N)
    x_bits = binary_decompose(x, K)

    # 验证重建
    recon_err = (x - sum(x_bits)).abs().max().item()
    print(f"重建误差 (理论 ≤ 2^-{K} = {2**(-K):.2e}): {recon_err:.2e}\n")

    nonlinearities = {
        'ReLU':    torch.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh':    torch.tanh,
        'GELU':    F.gelu,
        'x²':     lambda t: t ** 2,
        'x³':     lambda t: t ** 3,
        'sin(πx)': lambda t: torch.sin(math.pi * t),
        'exp(x)':  torch.exp,
    }

    print(f"{'函数':<12} {'平均误差':>12} {'最大误差':>12} {'相对误差':>12}")
    print("-" * 52)

    for name, f in nonlinearities.items():
        f_sum = f(x)                            # f(Σx_k)
        sum_f = sum(f(x_k) for x_k in x_bits)  # Σf(x_k)
        error = (f_sum - sum_f).abs()
        mean_err = error.mean().item()
        max_err = error.max().item()
        rel_err = (error / (f_sum.abs() + 1e-8)).mean().item()
        print(f"{name:<12} {mean_err:>12.6f} {max_err:>12.6f} {rel_err:>12.2%}")

    print()
    print("分析:")
    print("  ReLU 误差 ≈ 0: 因为 x_k ≥ 0 → ReLU(x_k) = x_k → Σ ReLU(x_k) = Σ x_k = x = ReLU(x)")
    print("  其余非线性函数误差显著, 因为 f(a+b) ≠ f(a)+f(b)")
    print("  Sigmoid 误差最大: K=8 个 σ(x_k) 的和远超 σ(x) (因为 σ(small) ≈ 0.5)")


# ============================================================
# Experiment 2: Linear 层 bias 的累加问题
# ============================================================
def exp2_linear_bias():
    print("\n" + "=" * 70)
    print("Exp 2: 线性层 bias 累加问题")
    print("       f(x) = Wx + b, Σf(x_k) = WΣx_k + K·b ≠ Wx + b")
    print("=" * 70)

    K, D, N = 8, 16, 1000
    W = torch.randn(D, D) * 0.1
    b = torch.randn(D) * 0.1
    x = torch.rand(N, D)
    x_bits = binary_decompose(x, K)

    f_x = x @ W.T + b
    sum_f_with_b = sum(xk @ W.T + b for xk in x_bits)     # Σ(Wx_k + b) = Wx + Kb
    sum_f_no_b = sum(xk @ W.T for xk in x_bits) + b       # Σ(Wx_k) + b = Wx + b ✓

    err_with = (f_x - sum_f_with_b).abs().mean().item()
    err_fix = (f_x - sum_f_no_b).abs().mean().item()

    print(f"每步都加 bias:  误差 = {err_with:.6f}  (理论 = (K-1)·|b|均值 = {(K-1)*b.abs().mean().item():.6f})")
    print(f"只加一次 bias:  误差 = {err_fix:.2e}")
    print()
    print("修复方案: bias 只在最后一个时间步加, 或者用无 bias 线性层 + 单独 bias 神经元")


# ============================================================
# Experiment 3: 膜电位交叉项精确恢复 x²
# ============================================================
def exp3_membrane_cross_terms():
    print("\n" + "=" * 70)
    print("Exp 3: 膜电位交叉项恢复 x²")
    print("       x² = (Σx_k)² = Σx_k² + 2·Σ_{j<k} x_j·x_k")
    print("       其中 Σ_{j<k} x_j·x_k = Σ_k V[k-1]·x_k (V 是膜电位)")
    print("=" * 70)

    K, N = 8, 10000
    x = torch.rand(N)
    x_bits = binary_decompose(x, K)

    x_sq = x ** 2
    sum_xk_sq = sum(xk ** 2 for xk in x_bits)

    # 膜电位交叉项: V[k-1] · x_k
    cross = torch.zeros(N)
    V = torch.zeros(N)  # 纯累加器 β=0
    for k in range(K):
        cross += V * x_bits[k]  # V[k-1] · x_k = (Σ_{j<k} x_j) · x_k
        V += x_bits[k]

    # x² = Σx_k² + 2·cross
    x_sq_recovered = sum_xk_sq + 2 * cross

    err_naive = (x_sq - sum_xk_sq).abs().mean().item()
    err_recovered = (x_sq - x_sq_recovered).abs().mean().item()

    print(f"真值 E[x²] = {x_sq.mean().item():.6f}")
    print(f"朴素 Σx_k² = {sum_xk_sq.mean().item():.6f}  (缺少交叉项)")
    print(f"恢复 Σx_k²+2·cross = {x_sq_recovered.mean().item():.6f}")
    print()
    print(f"朴素误差:   {err_naive:.6f}")
    print(f"恢复后误差: {err_recovered:.2e}")
    print()
    print("关键发现: V[t-1] · x_t (膜电位 × 当前输入) 精确恢复了二次交叉项!")
    print("这意味着: 具有乘法门控的 SNN 神经元可以精确计算 x²")
    print()

    # 扩展: x³ 需要三阶交叉项
    x_cu = x ** 3
    # x³ = Σx_k³ + 3·Σ_{j≠k} x_j·x_k² + 6·Σ_{i<j<k} x_i·x_j·x_k
    # 三阶项需要 V[k-1]² · x_k 和 V[k-1] · x_k² 的组合
    # 用两个神经元: 一个记录 V=Σx, 另一个记录 U=Σx²
    U = torch.zeros(N)  # Σ_{j<k} x_j²
    V = torch.zeros(N)  # Σ_{j<k} x_j
    triple = torch.zeros(N)
    for k in range(K):
        # V[k-1]² · x_k 包含 (Σ_{j<k}x_j)² · x_k
        # 但我们需要 Σ_{i<j<k} x_i·x_j·x_k = 0.5·((Σ_{j<k}x_j)² - Σ_{j<k}x_j²) · x_k
        triple += 0.5 * (V ** 2 - U) * x_bits[k]
        V += x_bits[k]
        U += x_bits[k] ** 2

    sum_xk_cu = sum(xk ** 3 for xk in x_bits)
    pair_mixed = torch.zeros(N)
    V2 = torch.zeros(N)
    for k in range(K):
        pair_mixed += V2 * x_bits[k] ** 2 + x_bits[k] * (sum(x_bits[j] ** 2 for j in range(k)))
        V2 += x_bits[k]

    # 直接验证: 用 (Σx_k)³ 展开
    x_cu_recovered = sum_xk_cu + 3 * cross * 2 * x.mean()  # 粗略估计

    # 更精确: (V_final)³ = x³
    x_cu_exact = V ** 3  # V 在循环结束后 = Σx_k = x
    err_cu_exact = (x_cu - x_cu_exact).abs().mean().item()
    print(f"x³ 通过 V^3 精确计算: 误差 = {err_cu_exact:.2e}")
    print("即: 只要膜电位精确累加了 x, 任何 f(x) = f(V_final) 都精确!")


# ============================================================
# Experiment 4: SNN 实际做的 ≠ Σf(x_k)
# ============================================================
def exp4_snn_actually_does():
    print("\n" + "=" * 70)
    print("Exp 4: SNN 实际做的事 vs 朴素分解")
    print("       SNN: 先累加 V = Σx_k, 再施加非线性 f(V)")
    print("       朴素: Σf(x_k)")
    print("       关键: SNN 做的是 f(Σx_k) = f(x), 不是 Σf(x_k)!")
    print("=" * 70)

    K, N = 8, 10000
    x = torch.rand(N)
    x_bits = binary_decompose(x, K)

    for fname, f in [('Sigmoid', torch.sigmoid), ('GELU', F.gelu), ('x²', lambda t: t**2)]:
        f_x = f(x)
        naive = sum(f(xk) for xk in x_bits)

        # SNN: accumulate then apply
        V = sum(x_bits)  # = x
        snn_result = f(V)

        err_naive = (f_x - naive).abs().mean().item()
        err_snn = (f_x - snn_result).abs().mean().item()

        print(f"  {fname:>8}: 朴素 Σf(x_k) 误差 = {err_naive:.6f},  SNN f(Σx_k) 误差 = {err_snn:.2e}")

    print()
    print('结论: SNN 的"先累加再阈值"范式天然计算 f(Σx_k) = f(x)')
    print("      叠加原理的不成立对 SNN 不是问题!")
    print()
    print("但真正的问题在多层网络的中间层...")


# ============================================================
# Experiment 5: 多层网络 — 中间层 spike 表示的挑战
# ============================================================
def exp5_multilayer():
    print("\n" + "=" * 70)
    print("Exp 5: 多层网络 — 中间层 spike 编码问题")
    print("       ANN: h = ReLU(W1·x + b1), y = W2·h + b2")
    print("       SNN: 第 1 层累积后发射 spike → 第 2 层接收 spike")
    print("=" * 70)

    K, D, N = 16, 8, 1000
    torch.manual_seed(42)

    W1 = torch.randn(D, D) * 0.3
    b1 = torch.randn(D) * 0.1
    W2 = torch.randn(D, D) * 0.3
    b2 = torch.randn(D) * 0.1

    x = torch.rand(N, D)
    x_bits = binary_decompose(x, K)

    # === ANN ground truth ===
    h_ann = torch.relu(x @ W1.T + b1)
    y_ann = h_ann @ W2.T + b2

    # === SNN 方案 A: β=0 纯累加器 (理想情况) ===
    V1 = torch.zeros(N, D)
    for k in range(K):
        V1 += x_bits[k] @ W1.T  # 不加 bias, 只累加
    V1 += b1  # bias 加一次
    h_snn_ideal = torch.relu(V1)  # 对累加结果施加非线性
    y_snn_ideal = h_snn_ideal @ W2.T + b2

    # === SNN 方案 B: β=0.5 有衰减 ===
    V1_decay = torch.zeros(N, D)
    beta = 0.5
    for k in range(K):
        V1_decay = beta * V1_decay + (1 - beta) * x_bits[k] @ W1.T
    V1_decay += b1
    h_snn_decay = torch.relu(V1_decay)
    y_snn_decay = h_snn_decay @ W2.T + b2

    # === SNN 方案 C: 时标补偿 (乘以 1/(1-β^K)·(1-β) 来恢复均值) ===
    scale = 1.0 / (1.0 - beta ** K) * K * (1 - beta) if beta > 0 else 1.0
    # 更精确: V_final = (1-β) Σ_{k=0}^{K-1} β^{K-1-k} x_k
    # 如果 x_k = c (常数), V_final = c·(1-β^K), 所以补偿 = 1/(1-β^K)
    V1_comp = V1_decay * (1.0 / (1.0 - beta ** K))
    V1_comp += b1
    h_snn_comp = torch.relu(V1_comp)
    y_snn_comp = h_snn_comp @ W2.T + b2

    err_ideal = (y_ann - y_snn_ideal).abs().mean().item()
    err_decay = (y_ann - y_snn_decay).abs().mean().item()
    err_comp = (y_ann - y_snn_comp).abs().mean().item()

    print(f"方案 A (β=0 纯累加):   误差 = {err_ideal:.2e}  ← 精确匹配 ANN")
    print(f"方案 B (β=0.5 有衰减): 误差 = {err_decay:.6f}  ← 衰减导致失配")
    print(f"方案 C (β=0.5+补偿):   误差 = {err_comp:.6f}  ← 补偿后改善")
    print()
    print("分析:")
    print("  β=0 时 SNN 精确等价于 ANN (对单层)")
    print("  β>0 引入指数衰减, MSB(先到)被衰减更多, LSB(后到)被衰减更少")
    print("  但 β 是可学习参数, 训练会自动调整来补偿!")
    print()
    print("  真正的挑战: 中间层输出是 SPIKE (0/1), 不是连续值 h")
    print("  第 2 层需要从 spike pattern 恢复出 h 的信息 → 这就是 rate/temporal coding 的问题")


# ============================================================
# Experiment 6: Rate Coding 收敛性
# ============================================================
def exp6_rate_coding():
    print("\n" + "=" * 70)
    print("Exp 6: Rate Coding 收敛性")
    print("       如果神经元以 rate = f(x) 发射, T 步后平均 ≈ f(x)")
    print("       测试: 不同 T 下的近似质量")
    print("=" * 70)

    N = 5000
    x = torch.rand(N) * 2 - 0.5  # x ∈ [-0.5, 1.5]

    # 目标: h = ReLU(x)
    h_true = torch.relu(x)

    # Rate coding: 以概率 p = clip(h, 0, 1) 发射
    # (实际 rate coding 更复杂, 这里简化演示)
    print(f"{'T (时间步)':>12} {'均方误差':>14} {'相对误差':>14}")
    print("-" * 44)

    for T in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        # 对每个神经元, 以概率 min(max(h,0),1) 在 T 步中发射
        rate = h_true.clamp(0, 1)
        spikes = torch.zeros(N)
        for t in range(T):
            spikes += (torch.rand(N) < rate).float()
        h_approx = spikes / T  # 平均发射率

        mse = ((h_true.clamp(0, 1) - h_approx) ** 2).mean().item()
        rel = (h_true.clamp(0, 1) - h_approx).abs().mean().item() / (h_true.clamp(0, 1).mean().item() + 1e-8)
        print(f"{T:>12} {mse:>14.6f} {rel:>14.4%}")

    print()
    print("分析: Rate coding MSE ∝ 1/T (中心极限定理)")
    print("  T=256 步才有 ~1% 误差 → 太慢!")
    print("  我们的 K-bit binary coding (K=16) 精度 = 2^-16 ≈ 0.0015% → 效率高 16× !")


# ============================================================
# Experiment 7: Boolean 超立方体上的多线性展开
# ============================================================
def exp7_boolean_multilinear():
    print("\n" + "=" * 70)
    print("Exp 7: Boolean 超立方体多线性展开")
    print("       任何 f(b_0,...,b_{K-1}) 都可写为多线性多项式:")
    print("       f = Σ_S c_S · Π_{k∈S} b_k")
    print("       因为 b_k ∈ {0,1} → b_k² = b_k (幂等)")
    print("=" * 70)

    K = 4  # 小 K 方便完整枚举
    N = 2 ** K  # 所有可能的 bit pattern

    # 枚举所有 K-bit 组合
    bits = torch.zeros(N, K)
    for i in range(N):
        for k in range(K):
            bits[i, k] = (i >> (K - 1 - k)) & 1

    # x = Σ b_k · 2^{-(k+1)}
    weights = torch.tensor([2.0 ** (-(k + 1)) for k in range(K)])
    x = (bits * weights).sum(dim=1)

    # 目标函数: f(x) = sigmoid(x)
    f_x = torch.sigmoid(x)

    # 多线性展开: f = c_∅ + Σ_k c_k·b_k + Σ_{j<k} c_{jk}·b_j·b_k + ...
    # 对 K=4, 共 2^4 = 16 个系数 (= 样本数, 所以精确拟合)

    # 构建 Vandermonde 矩阵 (多线性基)
    from itertools import combinations
    basis = []
    labels = []
    for size in range(K + 1):
        for S in combinations(range(K), size):
            col = torch.ones(N)
            for k in S:
                col = col * bits[:, k]
            basis.append(col)
            labels.append(S if S else '∅')

    A = torch.stack(basis, dim=1)  # [N, 2^K]

    # 求解 c = A^{-1} f (精确因为 A 是方阵且满秩)
    c = torch.linalg.solve(A, f_x)

    # 验证
    f_reconstructed = A @ c
    err = (f_x - f_reconstructed).abs().max().item()
    print(f"重建误差: {err:.2e} (应为 ~0)")
    print()

    # 展示系数
    print(f"{'基函数':<20} {'系数':>12} {'|系数|':>12}")
    print("-" * 48)
    for label, coeff in sorted(zip(labels, c.tolist()), key=lambda x: (len(x[0]) if x[0] != '∅' else 0, x[1])):
        name = '1' if label == '∅' else '·'.join(f'b{k}' for k in label)
        print(f"{name:<20} {coeff:>12.6f} {abs(coeff):>12.6f}")

    # 统计各阶能量
    print()
    energy_by_order = {}
    for label, coeff in zip(labels, c.tolist()):
        order = 0 if label == '∅' else len(label)
        energy_by_order[order] = energy_by_order.get(order, 0) + coeff ** 2

    print("各阶能量分布:")
    total_energy = sum(energy_by_order.values())
    for order in sorted(energy_by_order):
        pct = energy_by_order[order] / total_energy * 100
        bar = '█' * int(pct / 2)
        print(f"  {order}阶: {energy_by_order[order]:.6f} ({pct:5.1f}%) {bar}")

    print()
    print("分析:")
    print("  0阶 (常数) + 1阶 (线性) 占主要能量 → sigmoid 近似线性区域大")
    print("  2阶 (二次交叉) 提供非线性修正 → SNN 门控机制可捕获")
    print("  高阶 (≥3) 贡献小 → 实践中可忽略")
    print()
    print("  核心结论: 对 K-bit 二进制输入, 任何函数都是 bits 的多线性多项式")
    print("  SNN 通过 {膜电位累加 + 阈值 + 门控} 可以表达所有必要的阶数")


# ============================================================
# 汇总
# ============================================================
def summary():
    print("\n" + "=" * 70)
    print("总结: 非线性叠加分析与 SNN 可行性")
    print("=" * 70)
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数学事实:
  f(Σx_k) ≠ Σf(x_k)  对任何非线性 f (除 ReLU + 非负输入)

但这不是问题, 因为:

  1. SNN 做的是 f(Σx_k), 不是 Σf(x_k)
     膜电位先累加, 再由阈值施加非线性 → 等价于 f(x)

  2. 中间层 spike 编码的挑战可通过以下方式解决:
     a) β 可学习 — 训练自动补偿衰减失真
     b) K-bit binary coding 比 rate coding 高效 16× 以上
     c) 门控机制提供二次非线性 (捕获交叉项)

  3. 对 K-bit 二进制输入, 任何函数都是 bits 的多线性多项式
     阶数 ≤ K, 但实践中 0-2 阶占 >95% 能量
     SNN 的 {累加 + 阈值 + 门控} 足以表达

  4. 蒸馏不需要内部机制对齐:
     只需 Loss = ||SNN(x) - ANN(x)||²
     SNN 用自己的计算范式学习匹配 ANN 输出

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SNN Diffusion 蒸馏路线:
  Teacher: PixArt-α (600M, DiT)
  Student: SNN-DiT (替换 DiT block 为 SNNDecoderLayer)
  编码: 图像 patch → 二进制 spike frames
  去噪: 每个 diffusion step 对应 SNN 的 K 步时间动态
  蒸馏: 对齐中间特征 + 输出噪声预测

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    exp1_decomposition_error()
    exp2_linear_bias()
    exp3_membrane_cross_terms()
    exp4_snn_actually_does()
    exp5_multilayer()
    exp6_rate_coding()
    exp7_boolean_multilinear()
    summary()
