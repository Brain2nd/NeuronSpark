# Parallel Scan 优化计划

> 状态：当前方案，持续迭代
> 日期：2026-02-18

## 1. 当前瓶颈分析

### 1.1 性能现状
- **TPS (Tokens Per Second): 5**
- 1 epoch 预估: ~94 年（不可接受）
- 瓶颈: `atomic_ops/parallel_scan.py` 的 `hillis_steele_scan`

### 1.2 根因
`hillis_steele_scan` 使用 Python `while` 循环 + `torch.cat`：

```python
d = 1
while d < K:
    A_new_tail = A[d:] * A[:-d]
    B_new_tail = A[d:] * B[:-d] + B[d:]
    A = torch.cat([A[:d], A_new_tail], dim=0)
    B = torch.cat([B[:d], B_new_tail], dim=0)
    d *= 2
```

**问题**：
- K=8 → 3 次循环迭代，每次 2 个 `torch.cat` + 3 个逐元素操作 → **9 次 kernel launch**
- `plif_parallel_forward` 调用 3 次 `linear_recurrence`（Phase 1 + Phase 2 迭代 + Phase 3）
- 每次 `linear_recurrence` 调用 1 次 `hillis_steele_scan` → 9 次 launch
- Phase 2 不动点迭代: max_iter=3，每次迭代 1 次 `linear_recurrence`
- **单层单次前向**: 约 (1 + 3 + 1) × 9 = **45 次 kernel launch**
- **20 层 × 前向+反向**: 约 45 × 20 × 2 = **1800 次 kernel launch / token**
- 每次 launch 的 CPU-GPU 同步开销 ~10μs → 仅 launch 开销就 ~18ms/token

### 1.3 torch.compile 分析
- 会将 `while` 循环展开（K=8 固定），但 `torch.cat` 阻止跨迭代融合
- 最终仍产生多个小 kernel，无法融合为单 kernel
- 预期改善有限

## 2. 解决方案: accelerated-scan 库

### 2.1 库概述
- **仓库**: https://github.com/proger/accelerated-scan
- **版本**: 0.3.1 (2026-01-07)
- **安装**: `pip install accelerated-scan`
- **许可**: MIT

### 2.2 核心能力
解决一阶仿射递推（与我们的需求完全匹配）:
```
x[t] = a[t] * x[t-1] + b[t]
```

| 特性 | 状态 |
|------|------|
| DGX Spark (aarch64, GB10, sm_121) | ✓ 已验证 |
| CUDA 13.0 | ✓ 兼容 |
| float32 / bfloat16 | ✓ 支持 |
| autograd 反向传播 | ✓ 支持 (自定义 autograd.Function) |
| 任意序列长度 T | ✓ Triton 后端支持 |

### 2.3 后端选择

| 后端 | 推荐度 | 说明 |
|------|--------|------|
| `accelerated_scan.scalar` (Triton) | **首选** | GB10 上最快，T 无限制 |
| `accelerated_scan.warp` (CUDA C++) | 备选 | T 必须为 2 的幂且 ≥32 |
| `accelerated_scan.ref` (PyTorch) | 调试用 | 纯 PyTorch，可 torch.compile |

**GB10 实测加速比** (vs 顺序循环):

| B | C | T | Triton | 加速比 |
|---|---|---|--------|--------|
| 1 | 512 | 128 | 0.015ms | 131x |
| 1 | 512 | 2048 | 0.021ms | 1484x |
| 8 | 512 | 512 | 0.076ms | 106x |
| 8 | 1536 | 2048 | 1.418ms | 23x |

### 2.4 环境要求
```bash
pip install accelerated-scan
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```

### 2.5 API

```python
from accelerated_scan.scalar import scan

# gates (a): (B, C, T) — 衰减系数 beta
# tokens (b): (B, C, T) — 输入项 u
# 返回: (B, C, T) — 递推结果 V
V = scan(gates.contiguous(), tokens.contiguous())
```

**注意**: 初始状态固定为 0（`x[-1] = 0`），没有 `v_init` 参数。

## 3. 集成方案

### 3.1 形状适配
我们使用 SpikingJelly 的 `(K, B*D)` 或 `(K, batch, D)` 格式，
accelerated-scan 需要 `(B, C, T)` 格式：

```python
# 我们的格式: (K, batch, D) → accelerated-scan: (batch, D, K)
a_bck = a.permute(1, 2, 0).contiguous()  # (K, batch, D) → (batch, D, K)
b_bck = b.permute(1, 2, 0).contiguous()

V_bck = scan(a_bck, b_bck)  # (batch, D, K)

V = V_bck.permute(2, 0, 1)  # → (K, batch, D)
```

### 3.2 v_init 处理
accelerated-scan 固定 `x[-1] = 0`。我们需要 `V[-1] = v_init`。

利用仿射递推性质：
```
V[k] = A[k] * v_init + B[k]
```
其中 `(A, B)` 是前缀复合结果。

**方法**: 让 scan 计算 `B[k]`（v_init=0 时的轨迹），然后手动加上 `A[k] * v_init`：

```python
# 方法 1: 分两次 scan
A = scan(a, torch.ones_like(a))  # 前缀积 (v_init 的系数)
B = scan(a, b)                    # v_init=0 的递推结果
V = A * v_init + B                # 完整结果
```

或者更高效地：
```python
# 方法 2: 修改首项
# 令 b'[0] = a[0] * v_init + b[0], b'[k] = b[k] (k>0)
# 这样 scan(a, b') 直接得到正确的 V
b_mod = b.clone()
b_mod[:, :, 0] = b_mod[:, :, 0] + a[:, :, 0] * v_init  # v_init: (batch, D)
V = scan(a, b_mod)
```

方法 2 只需 **1 次 scan**，更高效。

### 3.3 替换 linear_recurrence

```python
# 旧代码 (parallel_scan.py)
def linear_recurrence(beta, u, v_init):
    A, B = hillis_steele_scan(beta, u)
    V = A * v_init.unsqueeze(0) + B
    return V

# 新代码
from accelerated_scan.scalar import scan as acc_scan

def linear_recurrence(beta, u, v_init):
    """beta: (K, *shape), u: (K, *shape), v_init: (*shape)"""
    K = beta.shape[0]
    # reshape to (B, C, T) — 把 K 放到最后
    # *shape 可能是 (batch, D) 或 (batch*D,)
    orig_shape = beta.shape[1:]  # (*shape)
    flat_dim = orig_shape.numel()

    a = beta.reshape(K, -1).permute(1, 0).unsqueeze(0).contiguous()   # (1, flat, K)
    b = u.reshape(K, -1).permute(1, 0).unsqueeze(0).contiguous()      # (1, flat, K)
    vi = v_init.reshape(-1)                                             # (flat,)

    # 修改首项以纳入 v_init
    b[:, :, 0] = b[:, :, 0] + a[:, :, 0] * vi.unsqueeze(0)

    V = acc_scan(a, b)  # (1, flat, K) — 单次 kernel!

    V = V.squeeze(0).permute(1, 0).reshape(K, *orig_shape)  # (K, *shape)
    return V
```

### 3.4 对 plif_parallel_forward 的影响

`plif_parallel_forward` 调用 `linear_recurrence` 的位置：
- **Phase 1**: 1 次（线性轨迹）
- **Phase 2**: max_iter-1 次（不动点迭代，在 `torch.no_grad()` 下）
- **Phase 3**: 1 次（有梯度的最终计算）

替换后每次 `linear_recurrence` 从 ~9 次 kernel launch → **1 次 kernel launch**。

**单层单次前向**:
- 旧: ~45 次 kernel launch
- 新: ~5 次 scan kernel + 少量逐元素操作 ≈ **10 次 kernel launch**
- **减少 ~4.5 倍 kernel launch**

**20 层前向+反向**:
- 旧: ~1800 次 kernel launch
- 新: ~400 次 kernel launch (每次 scan 的 backward 也是 1 次 kernel)

### 3.5 预期加速

保守估计（仅 kernel launch 开销减少）:
- 旧: 1800 launches × ~10μs = 18ms 纯开销
- 新: 400 launches × ~10μs = 4ms 纯开销
- **launch 开销减少 ~14ms/token**

但实际瓶颈不仅是 launch 开销，scan kernel 本身也比分散的小 kernel 更高效（数据在 GPU 寄存器/shared memory 中流转，不反复经过 global memory）。

**乐观估计**: TPS 从 5 提升到 **15-30**（3-6x 加速）。

**局限**: K=8 非常短，scan 本身的并行优势有限（log2(8)=3 层 vs K=8 顺序步）。真正的提升来自消除 Python 循环和 kernel launch 开销。

## 4. 进一步优化方向（后续迭代）

### 4.1 批量 Scan 融合
当前每层的 Phase 1/2/3 各自独立调用 scan。可以考虑：
- 将多个独立的 scan 拼接为一个大 tensor，单次 scan 处理
- 例如 Phase 2 的迭代 scan 和 delta_S scan 可以 concat 后一次处理

### 4.2 减少 Scan 次数
- Phase 2 不动点迭代: max_iter=3 → 尝试 max_iter=1（文献表明 1 次迭代通常足够）
- 如果 spike 很稀疏（火发率 <10%），甚至可以跳过 Phase 2

### 4.3 自定义 Triton Kernel
如果 accelerated-scan 的通用 scan 不够快（因为 K=8 太短），可以写一个专门针对小 K 的 fused kernel：
- 将 scan + spike + reset 融合为单个 Triton kernel
- 对 K=8，直接展开循环，在寄存器中完成所有计算
- 这是最终极的优化，但工作量大

### 4.4 torch.compile 包装
用 `torch.compile` 包装整个 `plif_parallel_forward`，让 TorchInductor 融合 scan 之间的逐元素操作（spike 判断、reset 计算等）。

## 5. 实施步骤

### Step 1: 安装与验证
```bash
conda activate SNN
pip install accelerated-scan
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
python -c "from accelerated_scan.scalar import scan; print('OK')"
```

### Step 2: 写集成代码
- 修改 `atomic_ops/parallel_scan.py`
- 保留旧的 `hillis_steele_scan` 作为 fallback
- 新增基于 accelerated-scan 的 `linear_recurrence`
- 确保 autograd 正确（梯度对比测试）

### Step 3: 正确性验证
- 对比新旧 `linear_recurrence` 输出（应 numerically close）
- 对比新旧 `plif_parallel_forward` 的 spike pattern 和梯度
- 用小模型跑几步训练，确认 loss 曲线无异常

### Step 4: 性能测试
- 运行 `exp/bench_compile.py`（已写好但未执行）
- 对比新旧 TPS

### Step 5: 修改 save_interval 后重新训练
- `save_interval` 从 1000 改为 200
- 用 `nohup` 或 `tmux` 启动训练，脱离 Claude Code session
- 监控 loss 曲线和 TPS

## 6. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| accelerated-scan 在 GB10 上有 bug | 先跑 correctness test，对比旧实现 |
| Triton 后端 non-commutative bug | 用 ref 后端做 ground truth 对比 |
| v_init 处理不正确 | 数学验证 + 数值对比 |
| K=8 太短，加速不明显 | 即使加速有限，消除 Python 循环本身就有价值 |
| 形状 permute 开销 | contiguous() 可能触发 copy，但只是 O(K×D) 的小操作 |
