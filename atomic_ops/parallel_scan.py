"""
Parallel Scan 工具函数：SNN 线性递推的高效并行求解（v7.1: 可微版本）

实现 Hillis-Steele 并行扫描算法，用于求解仿射递推：
  V[k] = a[k] * V[k-1] + b[k],  V[0] = v_init

以及带 spike 的完整 PLIF 神经元动力学：
  V_pre[k] = beta[k] * V_post[k-1] + u[k]
  s[k] = Θ(V_pre[k] - v_th[k])
  V_post[k] = V_pre[k] - v_th[k] * s[k]

v7.0 → v7.1 变更：
  - hillis_steele_scan: 消除原地操作，全部用 torch.cat 重建，autograd 兼容
  - plif_parallel_forward: 新增 surrogate_function 参数，不动点迭代 detach，
    输出 spike 使用 surrogate gradient
  - plif_fixed_param_forward: 接受 tensor beta/v_th，透传 surrogate_function

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

import torch


def hillis_steele_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hillis-Steele 并行前缀扫描：计算仿射映射序列的所有前缀复合。

    给定仿射映射 f_k(x) = a[k] * x + b[k], k = 0, ..., K-1，
    计算前缀复合 F_k = f_k ∘ f_{k-1} ∘ ... ∘ f_0，
    使得 V[k] = F_k(v_init) = A[k] * v_init + B[k]。

    复合规则: (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)

    实现使用 torch.cat 重建张量（无原地操作），完全兼容 autograd。

    Args:
        a: (K, *shape) — 乘性系数（如 β）
        b: (K, *shape) — 加性项（如 α·I）

    Returns:
        A: (K, *shape) — 前缀积 A[k] = ∏_{j=0}^{k} a[j]
        B: (K, *shape) — 前缀和 B[k] 使得 V[k] = A[k] * v_init + B[k]

    并行深度: O(log K)
    工作量: O(K * log K)
    """
    K = a.shape[0]
    A = a
    B = b

    d = 1
    while d < K:
        # 对 index >= d 的元素：与 d 步前的元素复合
        # (A[k], B[k]) = (A[k], B[k]) ∘ (A[k-d], B[k-d])
        # = (A[k] * A[k-d], A[k] * B[k-d] + B[k])
        A_new_tail = A[d:] * A[:-d]
        B_new_tail = A[d:] * B[:-d] + B[d:]

        # 重建完整张量（无原地修改）
        A = torch.cat([A[:d], A_new_tail], dim=0)
        B = torch.cat([B[:d], B_new_tail], dim=0)

        d *= 2

    return A, B


def linear_recurrence(beta: torch.Tensor, u: torch.Tensor, v_init: torch.Tensor) -> torch.Tensor:
    """
    求解线性递推: V[k] = beta[k] * V[k-1] + u[k], V[-1] = v_init

    使用 Hillis-Steele parallel scan。

    Args:
        beta: (K, *shape) — 衰减系数，值域 (0, 1)
        u:    (K, *shape) — 输入项
        v_init: (*shape) — 初始状态

    Returns:
        V: (K, *shape) — 所有 K 步的状态
    """
    A, B = hillis_steele_scan(beta, u)
    # V[k] = A[k] * v_init + B[k]
    V = A * v_init.unsqueeze(0) + B
    return V


def plif_parallel_forward(
    beta: torch.Tensor,
    u: torch.Tensor,
    v_th: torch.Tensor,
    v_init: torch.Tensor,
    max_iter: int = 3,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PLIF 神经元的并行前向传播（soft reset，surrogate gradient 兼容）。

    求解:
      V_pre[k] = beta[k] * V_post[k-1] + u[k]
      s[k] = Θ(V_pre[k] - v_th[k])
      V_post[k] = V_pre[k] - v_th[k] * s[k]

    方法:
      Phase 1: 线性轨迹 parallel scan（有梯度）
      Phase 2: spike 不动点迭代（detach，确定离散 spike pattern）
      Phase 3: 用 converged spike pattern 重算 V_post（有梯度），
               surrogate_function(V_pre - v_th) 生成可微 spike 输出

    Args:
        beta:  (K, *shape) — 衰减系数
        u:     (K, *shape) — 输入 α·I
        v_th:  (K, *shape) — 动态阈值
        v_init: (*shape) — 初始膜电位
        max_iter: spike 不动点迭代次数上限
        surrogate_function: SpikingJelly surrogate gradient 函数（如 surrogate.Sigmoid(alpha=4.0)）
                           None 时退化为硬阈值（无梯度）

    Returns:
        spike: (K, *shape) — spike 模式（有 surrogate gradient）
        V_post: (K, *shape) — 发放后膜电位
        V_pre: (K, *shape) — 发放前膜电位
    """
    # Phase 1: 线性轨迹 V_L (假设从不发放)
    V_L = linear_recurrence(beta, u, v_init)  # (K, *shape)

    # Phase 2: Spike 不动点迭代（全部 detach，不建立梯度图）
    # 目的：确定哪些神经元在哪些步发放（离散决策）
    with torch.no_grad():
        V_L_det = V_L.detach()
        beta_det = beta.detach()
        v_th_det = v_th.detach()
        v_init_det = v_init.detach() if isinstance(v_init, torch.Tensor) else v_init

        spike_pattern = (V_L_det >= v_th_det).float()

        for _ in range(max_iter - 1):
            # 计算 ΔS: ΔS[k] = beta[k] * ΔS[k-1] + v_th[k] * s[k]
            delta_S = linear_recurrence(
                beta_det, v_th_det * spike_pattern,
                torch.zeros_like(v_init_det) if isinstance(v_init_det, torch.Tensor)
                else torch.zeros_like(V_L_det[0]),
            )

            # ΔS_prev = ΔS[k-1]（位移一步）
            delta_S_prev = torch.zeros_like(delta_S)
            delta_S_prev[1:] = delta_S[:-1]

            # V_pre = V_L - beta * ΔS_prev
            V_pre_det = V_L_det - beta_det * delta_S_prev

            # 更新 spike
            spike_new = (V_pre_det >= v_th_det).float()

            # 收敛检查
            if torch.equal(spike_new, spike_pattern):
                break
            spike_pattern = spike_new

    # Phase 3: 用 converged spike pattern 重算 V_post（有完整梯度）
    # spike_pattern 是 detached 的，作为常数参与计算
    # 梯度通过 u, v_th, beta, v_init 流动
    u_eff = u - v_th * spike_pattern
    V_post = linear_recurrence(beta, u_eff, v_init)  # (K, *shape)

    # 重建 V_pre（有梯度，用于 surrogate gradient）
    V_post_prev = torch.zeros_like(V_post)
    if isinstance(v_init, torch.Tensor):
        V_post_prev[0] = v_init
    V_post_prev[1:] = V_post[:-1]
    V_pre = beta * V_post_prev + u

    # 生成可微 spike 输出
    if surrogate_function is not None:
        # forward: Heaviside(V_pre - v_th), backward: surrogate gradient
        spike = surrogate_function(V_pre - v_th)
    else:
        # 退化模式：硬阈值，无梯度
        spike = (V_pre >= v_th).float()

    return spike, V_post, V_pre


def plif_fixed_param_forward(
    beta,
    u: torch.Tensor,
    v_th,
    v_init: torch.Tensor,
    max_iter: int = 3,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    固定参数 PLIF 神经元的并行前向（如输出神经元、FFN 神经元）。

    ParametricLIFNode 方程: V[k] = beta * V[k-1] + (1-beta) * x[k]
    其中 beta = 1/(1+exp(w)), 可为 scalar tensor（保持梯度流向 w）。

    Args:
        beta: 衰减率 — scalar float、0-dim tensor 或 (K, *shape) tensor
        u: (K, *shape) — 输入（已乘以 (1-beta)）
        v_th: 阈值 — scalar float、0-dim tensor 或 (K, *shape) tensor
        v_init: (*shape) — 初始膜电位
        max_iter: spike 迭代次数
        surrogate_function: surrogate gradient 函数

    Returns:
        spike: (K, *shape) — spike 模式
        V_post: (K, *shape) — 发放后膜电位
    """
    K = u.shape[0]

    # 将 beta 扩展为与 u 同形的张量（保持梯度流）
    if isinstance(beta, torch.Tensor):
        if beta.dim() == 0:
            # 0-dim tensor: expand 不分配新内存，保持 autograd
            beta = beta.expand(K, *u.shape[1:])
    else:
        # Python float: 创建无梯度的常量张量
        beta = torch.full_like(u, beta)

    # 同样处理 v_th
    if isinstance(v_th, torch.Tensor):
        if v_th.dim() == 0:
            v_th = v_th.expand(K, *u.shape[1:])
    else:
        v_th = torch.full_like(u, v_th)

    spike, V_post, _ = plif_parallel_forward(
        beta, u, v_th, v_init, max_iter, surrogate_function,
    )
    return spike, V_post
