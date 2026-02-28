"""
Parallel Scan 工具函数：SNN 线性递推的高效并行求解（v7.4: Row-param PLIF kernel）

实现三层后端：
  1. Fused PLIF kernel（默认，CUDA + Sigmoid surrogate）：
     单 kernel 完成 PLIF 前向（scan + spike + soft reset）和反向（surrogate gradient）
     · per-element beta/v_th: _fused_plif_fwd_kernel / _fused_plif_bwd_kernel
     · row-param beta/v_th:  _fused_plif_fwd_rowparam_kernel / _fused_plif_bwd_rowparam_kernel
  2. Triton linear_recurrence（CUDA，非 Sigmoid 或无 surrogate）：
     列级并行 scan，O(K) 工作量，1 次 kernel launch
  3. Hillis-Steele parallel scan（CPU 回退）：O(K log K) 工作量

线性递推：
  V[k] = a[k] * V[k-1] + b[k],  V[-1] = v_init

PLIF 神经元动力学：
  V_pre[k] = beta[k] * V_post[k-1] + u[k]
  s[k] = Θ(V_pre[k] - v_th[k])
  V_post[k] = V_pre[k] - v_th[k] * s[k]

v7.3 → v7.4 变更：
  - 新增 row-param PLIF kernel：beta/v_th 恒定时加载到寄存器，省去 expand+contiguous
    · 前向：每步只读 u（beta/v_th 在寄存器中），内存读取减少 2/3
    · 反向：grad_beta/grad_v_th 在寄存器中累加（K 步归约），无需 per-step 存储
    · plif_fixed_param_forward 自动分发（scalar beta/v_th → row-param）
    · 新增公开接口 plif_rowparam_forward
  - plif_parallel_forward: 不变（per-element beta/v_th 场景）

v7.2 → v7.3 变更：
  - plif_parallel_forward: Fused PLIF kernel 替换 3-phase approach
  - 保留 v7.2 的 Triton linear_recurrence 用于非 PLIF 场景
  - 保留 3-phase fallback 用于 CPU/非 Sigmoid surrogate

v7.1 → v7.2 变更：
  - linear_recurrence: Triton fused kernel 替换 Hillis-Steele（CUDA）
  - plif_fixed_param_forward: expand 后 .contiguous()

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

import os
import torch


# ============================================================
# Triton fused recurrence kernels
# ============================================================

# DGX Spark (GB10, sm_121a): Triton 3.5.1 自带 ptxas 不支持 sm_121a，
# 需要使用系统 CUDA 13.0 的 ptxas
_SYSTEM_PTXAS = '/usr/local/cuda-13.0/bin/ptxas'
if os.path.exists(_SYSTEM_PTXAS) and 'TRITON_PTXAS_PATH' not in os.environ:
    os.environ['TRITON_PTXAS_PATH'] = _SYSTEM_PTXAS

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

if _HAS_TRITON:

    @triton.jit
    def _fwd_recurrence_kernel(
        A_ptr, B_ptr, INIT_ptr, OUT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Forward: V[k] = A[k]*V[k-1] + B[k], V[-1] = init.

        Grid: (ceil(num_cols / BLOCK),)
        Each program processes BLOCK columns across all K sequential steps.
        Accumulation in float32; storage in input dtype.
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v = a * v + b
            tl.store(OUT_ptr + off, v, mask=mask)

    @triton.jit
    def _bwd_recurrence_kernel(
        A_ptr, V_ptr, INIT_ptr, GRAD_OUT_ptr,
        GRAD_A_ptr, GRAD_B_ptr, GRAD_INIT_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Backward for V[k] = A[k]*V[k-1] + B[k].

        Reverse accumulation (k from K-1 down to 0):
          g = 0
          for k = K-1, ..., 0:
            g += grad_out[k]
            grad_B[k] = g
            grad_A[k] = g * V[k-1]   (V[-1] = init)
            g = g * A[k]
          grad_init = g
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        g = tl.zeros([BLOCK], dtype=tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            dV = tl.load(GRAD_OUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = g + dV

            tl.store(GRAD_B_ptr + off, g, mask=mask)

            if k > 0:
                v_prev = tl.load(
                    V_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(GRAD_A_ptr + off, g * v_prev, mask=mask)

            a = tl.load(A_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g = g * a

        tl.store(GRAD_INIT_ptr + cols, g, mask=mask)

    class _TritonLinearRecurrence(torch.autograd.Function):
        """Fused Triton linear recurrence: V[k] = A[k]*V[k-1] + B[k]."""

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, u, v_init):
            beta_c = beta.contiguous()
            u_c = u.contiguous()
            v_init_c = v_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()
            V = torch.empty_like(u_c)

            BLOCK = _TritonLinearRecurrence._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fwd_recurrence_kernel[grid](
                beta_c, u_c, v_init_c, V,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                ctx.save_for_backward(beta_c, V, v_init_c)
            ctx.K = K
            ctx.num_cols = num_cols

            return V

        @staticmethod
        def backward(ctx, grad_V):
            beta, V, v_init = ctx.saved_tensors
            grad_V_c = grad_V.contiguous()

            K = ctx.K
            num_cols = ctx.num_cols

            grad_beta = torch.empty_like(beta)
            grad_u = torch.empty_like(beta)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonLinearRecurrence._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _bwd_recurrence_kernel[grid](
                beta, V, v_init, grad_V_c,
                grad_beta, grad_u, grad_v_init,
                K, num_cols,
                BLOCK=BLOCK,
            )

            return grad_beta, grad_u, grad_v_init

    # ============================================================
    # Fused PLIF forward/backward kernels
    # ============================================================

    @triton.jit
    def _fused_plif_fwd_kernel(
        BETA_ptr, U_ptr, VTH_ptr, INIT_ptr,
        SPIKE_ptr, VPOST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Fused PLIF forward: single-pass sequential scan with inline spike + soft reset.

        Exact computation — sequential scan IS the ground truth.
        Replaces the 3-phase approach (linear scan + spike iteration + correction).

        Per column (parallel across batch*D):
          v = v_init
          for k = 0..K-1:
            v_pre = beta[k]*v + u[k]
            spike[k] = Θ(v_pre - v_th[k])
            v = v_pre - v_th[k]*spike[k]
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v + u
            spike = tl.where(v_pre >= vth, 1.0, 0.0)
            v = v_pre - vth * spike  # soft reset

            tl.store(SPIKE_ptr + off, spike, mask=mask)
            tl.store(VPOST_ptr + off, v, mask=mask)

    @triton.jit
    def _fused_plif_bwd_kernel(
        BETA_ptr, VTH_ptr, INIT_ptr, VPOST_ptr, SPIKE_ptr,
        GRAD_SPIKE_ptr, GRAD_VPOST_ptr,
        GRAD_BETA_ptr, GRAD_U_ptr, GRAD_VTH_ptr, GRAD_INIT_ptr,
        K, num_cols, ALPHA,
        BLOCK: tl.constexpr,
    ):
        """Fused PLIF backward: single reverse pass with Sigmoid surrogate gradient.

        优化: 寄存器缓存 v_post[k]，下一迭代复用为 v_prev，每步减少 1 次 global load。

        Reverse accumulation:
          acc = 0
          for k = K-1 downto 0:
            total_gV = grad_V_post[k] + acc
            sg = surrogate_grad(V_pre[k] - v_th[k])
            grad_v_pre = grad_spike[k]*sg + total_gV
            grad_beta[k] = grad_v_pre * V_post[k-1]
            grad_u[k] = grad_v_pre
            grad_v_th[k] = -grad_spike[k]*sg - total_gV*spike[k]
            acc = grad_v_pre * beta[k]
          grad_v_init = acc
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        acc = tl.zeros([BLOCK], dtype=tl.float32)

        # 预加载最后一步的 v_post — 进入循环时作为 "当前步" 使用
        cached_v = tl.load(
            VPOST_ptr + (K - 1) * num_cols + cols, mask=mask, other=0.0,
        ).to(tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v_post = cached_v  # 从寄存器缓存获取
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_s = tl.load(GRAD_SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g_V = tl.load(GRAD_VPOST_ptr + off, mask=mask, other=0.0).to(tl.float32)

            # V_post[k-1]
            if k > 0:
                v_prev = tl.load(
                    VPOST_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                cached_v = v_prev  # 缓存为下一迭代的 v_post
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # Sigmoid surrogate gradient
            x = v_post - vth * (1.0 - spike)  # = V_pre - v_th
            neg_ax = -ALPHA * x
            neg_ax = tl.where(neg_ax > 88.0, 88.0, neg_ax)  # prevent exp overflow
            sig = 1.0 / (1.0 + tl.exp(neg_ax))
            sg = ALPHA * sig * (1.0 - sig)

            total_gV = g_V + acc
            grad_v_pre = g_s * sg + total_gV

            tl.store(GRAD_BETA_ptr + off, grad_v_pre * v_prev, mask=mask)
            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)
            tl.store(GRAD_VTH_ptr + off, -g_s * sg - total_gV * spike, mask=mask)

            acc = grad_v_pre * beta

        tl.store(GRAD_INIT_ptr + cols, acc, mask=mask)

    # ============================================================
    # Fused PLIF kernels with row-parameter beta/v_th
    # (constant across K steps — e.g., ParametricLIFNode scalars)
    # ============================================================

    @triton.jit
    def _fused_plif_fwd_rowparam_kernel(
        BETA_ROW_ptr, U_ptr, VTH_ROW_ptr, INIT_ptr,
        SPIKE_ptr, VPOST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Fused PLIF forward with row-parameter beta and v_th.

        beta and v_th are (*shape) — constant across K steps, loaded once into registers.
        Reduces global memory reads from 3 per step (beta, u, v_th) to 1 (u only).
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vth = tl.load(VTH_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v + u
            spike = tl.where(v_pre >= vth, 1.0, 0.0)
            v = v_pre - vth * spike

            tl.store(SPIKE_ptr + off, spike, mask=mask)
            tl.store(VPOST_ptr + off, v, mask=mask)

    @triton.jit
    def _fused_plif_bwd_rowparam_kernel(
        BETA_ROW_ptr, VTH_ROW_ptr, INIT_ptr, VPOST_ptr, SPIKE_ptr,
        GRAD_SPIKE_ptr, GRAD_VPOST_ptr,
        GRAD_BETA_ROW_ptr, GRAD_U_ptr, GRAD_VTH_ROW_ptr, GRAD_INIT_ptr,
        K, num_cols, ALPHA,
        BLOCK: tl.constexpr,
    ):
        """Fused PLIF backward with row-parameter beta/v_th.

        优化: 寄存器缓存 v_post[k]，下一迭代复用为 v_prev，每步减少 1 次 global load。
        Gradients for beta and v_th are accumulated over K steps (reduction in registers).
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vth = tl.load(VTH_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_beta = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_vth = tl.zeros([BLOCK], dtype=tl.float32)

        # 预加载最后一步的 v_post — 进入循环时作为 "当前步" 使用
        cached_v = tl.load(
            VPOST_ptr + (K - 1) * num_cols + cols, mask=mask, other=0.0,
        ).to(tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            v_post = cached_v  # 从寄存器缓存获取（替代 global load）
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_s = tl.load(GRAD_SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g_V = tl.load(GRAD_VPOST_ptr + off, mask=mask, other=0.0).to(tl.float32)

            if k > 0:
                v_prev = tl.load(
                    VPOST_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                cached_v = v_prev  # 缓存为下一迭代的 v_post
            else:
                v_prev = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # Sigmoid surrogate gradient
            x = v_post - vth * (1.0 - spike)
            neg_ax = -ALPHA * x
            neg_ax = tl.where(neg_ax > 88.0, 88.0, neg_ax)
            sig = 1.0 / (1.0 + tl.exp(neg_ax))
            sg = ALPHA * sig * (1.0 - sig)

            total_gV = g_V + acc
            grad_v_pre = g_s * sg + total_gV

            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)

            # Accumulate gradients for row parameters (reduction over K in registers)
            acc_grad_beta += grad_v_pre * v_prev
            acc_grad_vth += -g_s * sg - total_gV * spike

            acc = grad_v_pre * beta

        tl.store(GRAD_INIT_ptr + cols, acc, mask=mask)
        tl.store(GRAD_BETA_ROW_ptr + cols, acc_grad_beta, mask=mask)
        tl.store(GRAD_VTH_ROW_ptr + cols, acc_grad_vth, mask=mask)

    class _TritonPLIFRowParamForward(torch.autograd.Function):
        """Fused Triton PLIF with row-parameter beta/v_th.

        For neurons with constant beta/v_th across K steps (ParametricLIFNode).
        Eliminates expand+contiguous for beta/v_th tensors, reduces memory I/O by ~40%.
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta_row, u, v_th_row, v_init, alpha):
            beta_row_c = beta_row.contiguous()
            u_c = u.contiguous()
            v_th_row_c = v_th_row.contiguous()
            v_init_c = v_init.contiguous()

            K = u_c.shape[0]
            num_cols = u_c[0].numel()

            spike = torch.empty_like(u_c)
            V_post = torch.empty_like(u_c)

            BLOCK = _TritonPLIFRowParamForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_rowparam_kernel[grid](
                beta_row_c, u_c, v_th_row_c, v_init_c,
                spike, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_row_c, v_th_row_c, v_init_c, V_post, spike)
            ctx.K = K
            ctx.num_cols = num_cols
            ctx.alpha = alpha

            return spike, V_post

        @staticmethod
        def backward(ctx, grad_spike, grad_V_post):
            beta_row, v_th_row, v_init, V_post, spike = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols
            alpha = ctx.alpha

            if grad_spike is None:
                grad_spike = torch.zeros_like(spike)
            if grad_V_post is None:
                grad_V_post = torch.zeros_like(V_post)

            grad_spike_c = grad_spike.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta_row = torch.empty_like(beta_row)
            grad_u = torch.empty_like(V_post)
            grad_v_th_row = torch.empty_like(v_th_row)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonPLIFRowParamForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_bwd_rowparam_kernel[grid](
                beta_row, v_th_row, v_init, V_post, spike,
                grad_spike_c, grad_V_post_c,
                grad_beta_row, grad_u, grad_v_th_row, grad_v_init,
                K, num_cols, float(alpha),
                BLOCK=BLOCK,
            )

            return grad_beta_row, grad_u, grad_v_th_row, grad_v_init, None


    # ============================================================
    # Fused PLIF row-param kernels with V_post recomputation (Mamba-style)
    # Forward: only stores spike (no V_post materialization)
    # Backward: recomputes V_post from u + beta + v_th + spike in registers
    # Optimal for short sequences (K≤32) — saves ~50% autograd memory
    # ============================================================

    @triton.jit
    def _fused_plif_fwd_rowparam_noV_kernel(
        BETA_ROW_ptr, U_ptr, VTH_ROW_ptr, INIT_ptr,
        SPIKE_ptr, V_LAST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Forward: only stores spike + V_last. V_post recomputed in backward."""
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vth = tl.load(VTH_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v + u
            spike = tl.where(v_pre >= vth, 1.0, 0.0)
            v = v_pre - vth * spike

            tl.store(SPIKE_ptr + off, spike, mask=mask)

        # Store final V_post (V[K-1]) for state update
        tl.store(V_LAST_ptr + cols, v, mask=mask)

    @triton.jit
    def _fused_plif_bwd_rowparam_recomp_kernel(
        BETA_ROW_ptr, VTH_ROW_ptr, INIT_ptr,
        U_ptr, SPIKE_ptr,
        GRAD_SPIKE_ptr,
        GRAD_BETA_ROW_ptr, GRAD_U_ptr, GRAD_VTH_ROW_ptr, GRAD_INIT_ptr,
        K, num_cols, ALPHA,
        BLOCK: tl.constexpr,
    ):
        """Backward with V_post recomputation (memory-optimized).

        Phase 1: Forward recompute V_post → reuse GRAD_U buffer
        Phase 2: Reverse step k reads GRAD_U[k-1] (=V_post[k-1]),
                 then overwrites GRAD_U[k] with grad_u[k].  No conflict.

        Saves 2 large tensors vs original:
          - grad_V_post_c  (always zero, replaced by constant 0)
          - V_post_temp     (reuses GRAD_U buffer)
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        beta = tl.load(BETA_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vth = tl.load(VTH_ROW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        v_init_val = tl.load(INIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Phase 1: 前向重计算 V_post → 写入 GRAD_U buffer (临时复用)
        v = v_init_val
        for k in range(K):
            off = k * num_cols + cols
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)
            v_pre = beta * v + u
            v = v_pre - vth * spike
            tl.store(GRAD_U_ptr + off, v, mask=mask)    # V_post temp

        # Phase 2: 标准反向 scan
        # GRAD_U[k] currently holds V_post[k]
        # Reverse step k reads [k-1] then overwrites [k] — no conflict
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_beta = tl.zeros([BLOCK], dtype=tl.float32)
        acc_grad_vth = tl.zeros([BLOCK], dtype=tl.float32)

        cached_v = tl.load(
            GRAD_U_ptr + (K - 1) * num_cols + cols, mask=mask, other=0.0,
        ).to(tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            v_post = cached_v
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_s = tl.load(GRAD_SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            if k > 0:
                v_prev = tl.load(
                    GRAD_U_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                cached_v = v_prev
            else:
                v_prev = v_init_val

            # Sigmoid surrogate gradient
            x = v_post - vth * (1.0 - spike)
            neg_ax = -ALPHA * x
            neg_ax = tl.where(neg_ax > 88.0, 88.0, neg_ax)
            sig = 1.0 / (1.0 + tl.exp(neg_ax))
            sg = ALPHA * sig * (1.0 - sig)

            # g_V = 0 (no external gradient flows into V_post)
            total_gV = acc
            grad_v_pre = g_s * sg + total_gV

            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)

            acc_grad_beta += grad_v_pre * v_prev
            acc_grad_vth += -g_s * sg - total_gV * spike

            acc = grad_v_pre * beta

        tl.store(GRAD_INIT_ptr + cols, acc, mask=mask)
        tl.store(GRAD_BETA_ROW_ptr + cols, acc_grad_beta, mask=mask)
        tl.store(GRAD_VTH_ROW_ptr + cols, acc_grad_vth, mask=mask)

    class _TritonPLIFRowParamRecompute(torch.autograd.Function):
        """Fused Triton PLIF with row-param beta/v_th and V_post recomputation.

        Same computation as _TritonPLIFRowParamForward, but does NOT store
        V_post across forward→backward gap. Instead, backward recomputes
        V_post from (u, beta, v_th, spike, v_init) — 16 steps of multiply-add
        in registers, near-zero cost for K≤32.

        Saves ~50% autograd memory (no V_post tensor in ctx.saved_tensors).
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta_row, u, v_th_row, v_init, alpha):
            beta_row_c = beta_row.contiguous()
            u_c = u.contiguous()
            v_th_row_c = v_th_row.contiguous()
            v_init_c = v_init.contiguous()

            K = u_c.shape[0]
            num_cols = u_c[0].numel()

            spike = torch.empty_like(u_c)
            V_last = torch.empty_like(v_init_c)

            BLOCK = _TritonPLIFRowParamRecompute._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_rowparam_noV_kernel[grid](
                beta_row_c, u_c, v_th_row_c, v_init_c,
                spike, V_last,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_row_c, v_th_row_c, v_init_c, spike, u_c)
            ctx.K = K
            ctx.num_cols = num_cols
            ctx.alpha = alpha

            return spike, V_last  # V_last = V_post[K-1] for state update

        @staticmethod
        def backward(ctx, grad_spike, _grad_V_last_unused):
            beta_row, v_th_row, v_init, spike, u = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols
            alpha = ctx.alpha

            if grad_spike is None:
                grad_spike = torch.zeros_like(spike)

            grad_spike_c = grad_spike.contiguous()

            grad_beta_row = torch.empty_like(beta_row)
            grad_u = torch.empty_like(u)
            grad_v_th_row = torch.empty_like(v_th_row)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonPLIFRowParamRecompute._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_bwd_rowparam_recomp_kernel[grid](
                beta_row, v_th_row, v_init,
                u, spike,
                grad_spike_c,
                grad_beta_row, grad_u, grad_v_th_row, grad_v_init,
                K, num_cols, float(alpha),
                BLOCK=BLOCK,
            )

            return grad_beta_row, grad_u, grad_v_th_row, grad_v_init, None

    class _TritonPLIFForward(torch.autograd.Function):
        """Fused Triton PLIF forward + backward.

        Single-pass sequential scan replaces the 3-phase approach:
          Phase 1 (linear scan) + Phase 2 (spike iteration) + Phase 3 (correction)
          → 1 fused kernel with inline spike detection + soft reset

        Advantages:
          - 1 kernel launch (vs 3-4 launches + ~10 element-wise ops)
          - Exact computation (no iteration convergence issues)
          - Less memory (no intermediate V_L, delta_S, delta_S_prev)
          - Higher precision (fp32 accumulation, no bf16 intermediate store/load)
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, u, v_th, v_init, alpha):
            beta_c = beta.contiguous()
            u_c = u.contiguous()
            v_th_c = v_th.contiguous()
            v_init_c = v_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()

            spike = torch.empty_like(u_c)
            V_post = torch.empty_like(u_c)

            BLOCK = _TritonPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_fwd_kernel[grid](
                beta_c, u_c, v_th_c, v_init_c,
                spike, V_post,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:4]):
                ctx.save_for_backward(beta_c, v_th_c, v_init_c, V_post, spike)
            ctx.K = K
            ctx.num_cols = num_cols
            ctx.alpha = alpha

            return spike, V_post

        @staticmethod
        def backward(ctx, grad_spike, grad_V_post):
            beta, v_th, v_init, V_post, spike = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols
            alpha = ctx.alpha

            if grad_spike is None:
                grad_spike = torch.zeros_like(spike)
            if grad_V_post is None:
                grad_V_post = torch.zeros_like(V_post)

            grad_spike_c = grad_spike.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta = torch.empty_like(beta)
            grad_u = torch.empty_like(beta)
            grad_v_th = torch.empty_like(v_th)
            grad_v_init = torch.empty_like(v_init)

            BLOCK = _TritonPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_plif_bwd_kernel[grid](
                beta, v_th, v_init, V_post, spike,
                grad_spike_c, grad_V_post_c,
                grad_beta, grad_u, grad_v_th, grad_v_init,
                K, num_cols, float(alpha),
                BLOCK=BLOCK,
            )

            return grad_beta, grad_u, grad_v_th, grad_v_init, None

    # ============================================================
    # Fused RF (Resonate-and-Fire) PLIF kernels
    # ============================================================
    # 二阶耦合动力学: V_pre = β·V_post_prev - ω·W_prev + u
    #                 W_new = ω·V_post_prev + β·W_prev
    # spike/reset 仅作用于 V, W 不 reset

    @triton.jit
    def _fused_rf_plif_fwd_kernel(
        BETA_ptr, OMEGA_ptr, U_ptr, VTH_ptr, VINIT_ptr, WINIT_ptr,
        SPIKE_ptr, VPOST_ptr, W_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """Fused Resonate-and-Fire PLIF forward.

        Per column:
          v, w = v_init, w_init
          for k = 0..K-1:
            v_pre = beta[k]*v - omega[k]*w + u[k]
            w_new = omega[k]*v + beta[k]*w
            spike[k] = Θ(v_pre - v_th[k])
            v = v_pre - v_th[k]*spike[k]
            w = w_new
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(VINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(WINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(OMEGA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v - omega * w + u
            w_new = omega * v + beta * w

            spike = tl.where(v_pre >= vth, 1.0, 0.0)
            v = v_pre - vth * spike  # soft reset V
            w = w_new               # W not reset

            tl.store(SPIKE_ptr + off, spike, mask=mask)
            tl.store(VPOST_ptr + off, v, mask=mask)
            tl.store(W_ptr + off, w, mask=mask)

    @triton.jit
    def _fused_rf_plif_bwd_kernel(
        BETA_ptr, OMEGA_ptr, VTH_ptr, VINIT_ptr, WINIT_ptr,
        VPOST_ptr, W_ptr, SPIKE_ptr,
        GRAD_SPIKE_ptr, GRAD_VPOST_ptr,
        GRAD_BETA_ptr, GRAD_OMEGA_ptr, GRAD_U_ptr, GRAD_VTH_ptr,
        GRAD_VINIT_ptr, GRAD_WINIT_ptr,
        K, num_cols, ALPHA,
        BLOCK: tl.constexpr,
    ):
        """Fused RF PLIF backward with dual accumulators (V, W chains).

        优化: 寄存器缓存 v_post[k]/w[k]，下一迭代复用为 v_prev/w_prev，
        每步减少 2 次 global memory load（从 10 降至 8）。

        Reverse accumulation:
          acc_v, acc_w = 0, 0
          for k = K-1 downto 0:
            total_gV = grad_V_post[k] + acc_v
            total_gW = acc_w
            grad_v_pre = grad_spike[k]*sg + total_gV
            grad_beta[k] = grad_v_pre*V_post[k-1] + total_gW*W[k-1]
            grad_omega[k] = grad_v_pre*(-W[k-1]) + total_gW*V_post[k-1]
            grad_u[k] = grad_v_pre
            grad_v_th[k] = -grad_spike[k]*sg - total_gV*spike[k]
            acc_v = grad_v_pre*beta[k] + total_gW*omega[k]
            acc_w = grad_v_pre*(-omega[k]) + total_gW*beta[k]
          grad_v_init = acc_v
          grad_w_init = acc_w
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        acc_v = tl.zeros([BLOCK], dtype=tl.float32)
        acc_w = tl.zeros([BLOCK], dtype=tl.float32)

        # 预加载最后一步的 v_post, w — 进入循环时作为 "当前步" 使用
        last_off = (K - 1) * num_cols + cols
        cached_v = tl.load(VPOST_ptr + last_off, mask=mask, other=0.0).to(tl.float32)
        cached_w = tl.load(W_ptr + last_off, mask=mask, other=0.0).to(tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(OMEGA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_s = tl.load(GRAD_SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)
            g_V = tl.load(GRAD_VPOST_ptr + off, mask=mask, other=0.0).to(tl.float32)

            # v_post[k] 和 w[k] 来自寄存器缓存（首次迭代从预加载获取）
            v_post = cached_v
            w_cur = cached_w

            # 加载 v_prev=v_post[k-1], w_prev=w[k-1]，同时缓存为下一迭代的 v_post, w
            if k > 0:
                v_prev = tl.load(
                    VPOST_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                w_prev = tl.load(
                    W_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                cached_v = v_prev  # 下一迭代 k-1 的 v_post
                cached_w = w_prev  # 下一迭代 k-1 的 w
            else:
                v_prev = tl.load(VINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                w_prev = tl.load(WINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

            # Sigmoid surrogate gradient
            x = v_post - vth * (1.0 - spike)  # = V_pre - v_th
            neg_ax = -ALPHA * x
            neg_ax = tl.where(neg_ax > 88.0, 88.0, neg_ax)
            sig = 1.0 / (1.0 + tl.exp(neg_ax))
            sg = ALPHA * sig * (1.0 - sig)

            total_gV = g_V + acc_v
            total_gW = acc_w
            grad_v_pre = g_s * sg + total_gV

            # Parameter gradients
            tl.store(GRAD_BETA_ptr + off, grad_v_pre * v_prev + total_gW * w_prev, mask=mask)
            tl.store(GRAD_OMEGA_ptr + off, grad_v_pre * (-w_prev) + total_gW * v_prev, mask=mask)
            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)
            tl.store(GRAD_VTH_ptr + off, -g_s * sg - total_gV * spike, mask=mask)

            # Propagate accumulators (transpose of forward transition matrix)
            acc_v = grad_v_pre * beta + total_gW * omega
            acc_w = grad_v_pre * (-omega) + total_gW * beta

        tl.store(GRAD_VINIT_ptr + cols, acc_v, mask=mask)
        tl.store(GRAD_WINIT_ptr + cols, acc_w, mask=mask)

    # ============================================================
    # RF PLIF recompute kernels (V_post/W not saved across fwd→bwd)
    # Forward: only stores spike + V_last + W_last
    # Backward: Phase 1 recomputes V_post/W, Phase 2 standard gradient
    # ============================================================

    @triton.jit
    def _fused_rf_plif_fwd_recompute_kernel(
        BETA_ptr, OMEGA_ptr, U_ptr, VTH_ptr, VINIT_ptr, WINIT_ptr,
        SPIKE_ptr, VLAST_ptr, WLAST_ptr,
        K, num_cols,
        BLOCK: tl.constexpr,
    ):
        """RF PLIF forward — only stores spike + V_last + W_last."""
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v = tl.load(VINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(WINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        for k in range(K):
            off = k * num_cols + cols
            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(OMEGA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v - omega * w + u
            w_new = omega * v + beta * w

            spike = tl.where(v_pre >= vth, 1.0, 0.0)
            v = v_pre - vth * spike
            w = w_new

            tl.store(SPIKE_ptr + off, spike, mask=mask)

        tl.store(VLAST_ptr + cols, v, mask=mask)
        tl.store(WLAST_ptr + cols, w, mask=mask)

    @triton.jit
    def _fused_rf_plif_bwd_recompute_kernel(
        BETA_ptr, OMEGA_ptr, VTH_ptr, VINIT_ptr, WINIT_ptr,
        U_ptr, SPIKE_ptr,
        GRAD_SPIKE_ptr,
        GRAD_BETA_ptr, GRAD_OMEGA_ptr, GRAD_U_ptr, GRAD_VTH_ptr,
        GRAD_VINIT_ptr, GRAD_WINIT_ptr,
        K, num_cols, ALPHA,
        BLOCK: tl.constexpr,
    ):
        """RF PLIF backward with V_post/W recomputation (memory-optimized).

        Phase 1: Forward recompute V_post/W → reuse GRAD_U / GRAD_VTH buffers
        Phase 2: Reverse gradient — read V_post[k-1]/W[k-1], then overwrite
                 with grad_u[k]/grad_v_th[k].  No conflict: reverse step k
                 reads [k-1] before writing [k].

        Saves 3 large tensors vs original:
          - grad_V_post_c  (always zero, replaced by constant 0)
          - V_post_temp     (reuses GRAD_U buffer)
          - W_temp          (reuses GRAD_VTH buffer)
        """
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < num_cols

        v_init_val = tl.load(VINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w_init_val = tl.load(WINIT_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Phase 1: Forward recompute V_post, W → GRAD_U / GRAD_VTH buffers
        v = v_init_val
        w = w_init_val
        for k in range(K):
            off = k * num_cols + cols
            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(OMEGA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_pre = beta * v - omega * w + u
            w_new = omega * v + beta * w
            v = v_pre - vth * spike
            w = w_new

            tl.store(GRAD_U_ptr + off, v, mask=mask)    # V_post temp
            tl.store(GRAD_VTH_ptr + off, w, mask=mask)   # W temp

        # Phase 2: Reverse gradient accumulation
        # GRAD_U[k] currently holds V_post[k], GRAD_VTH[k] holds W[k]
        # Reverse step k reads [k-1] then overwrites [k] — no conflict
        acc_v = tl.zeros([BLOCK], dtype=tl.float32)
        acc_w = tl.zeros([BLOCK], dtype=tl.float32)

        last_off = (K - 1) * num_cols + cols
        cached_v = tl.load(GRAD_U_ptr + last_off, mask=mask, other=0.0).to(tl.float32)
        cached_w = tl.load(GRAD_VTH_ptr + last_off, mask=mask, other=0.0).to(tl.float32)

        for k_rev in range(K):
            k = K - 1 - k_rev
            off = k * num_cols + cols

            beta = tl.load(BETA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            omega = tl.load(OMEGA_ptr + off, mask=mask, other=0.0).to(tl.float32)
            vth = tl.load(VTH_ptr + off, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            g_s = tl.load(GRAD_SPIKE_ptr + off, mask=mask, other=0.0).to(tl.float32)

            v_post = cached_v

            if k > 0:
                v_prev = tl.load(
                    GRAD_U_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                w_prev = tl.load(
                    GRAD_VTH_ptr + (k - 1) * num_cols + cols,
                    mask=mask, other=0.0,
                ).to(tl.float32)
                cached_v = v_prev
                cached_w = w_prev
            else:
                v_prev = v_init_val
                w_prev = w_init_val

            x = v_post - vth * (1.0 - spike)
            neg_ax = -ALPHA * x
            neg_ax = tl.where(neg_ax > 88.0, 88.0, neg_ax)
            sig = 1.0 / (1.0 + tl.exp(neg_ax))
            sg = ALPHA * sig * (1.0 - sig)

            # g_V = 0 (no external gradient flows into V_post)
            total_gV = acc_v
            total_gW = acc_w
            grad_v_pre = g_s * sg + total_gV

            tl.store(GRAD_BETA_ptr + off, grad_v_pre * v_prev + total_gW * w_prev, mask=mask)
            tl.store(GRAD_OMEGA_ptr + off, grad_v_pre * (-w_prev) + total_gW * v_prev, mask=mask)
            tl.store(GRAD_U_ptr + off, grad_v_pre, mask=mask)
            tl.store(GRAD_VTH_ptr + off, -g_s * sg - total_gV * spike, mask=mask)

            acc_v = grad_v_pre * beta + total_gW * omega
            acc_w = grad_v_pre * (-omega) + total_gW * beta

        tl.store(GRAD_VINIT_ptr + cols, acc_v, mask=mask)
        tl.store(GRAD_WINIT_ptr + cols, acc_w, mask=mask)

    class _TritonRFPLIFForward(torch.autograd.Function):
        """Fused Triton Resonate-and-Fire PLIF forward + backward."""

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, omega, u, v_th, v_init, w_init, alpha):
            beta_c = beta.contiguous()
            omega_c = omega.contiguous()
            u_c = u.contiguous()
            v_th_c = v_th.contiguous()
            v_init_c = v_init.contiguous()
            w_init_c = w_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()

            spike = torch.empty_like(u_c)
            V_post = torch.empty_like(u_c)
            W = torch.empty_like(u_c)

            BLOCK = _TritonRFPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_rf_plif_fwd_kernel[grid](
                beta_c, omega_c, u_c, v_th_c, v_init_c, w_init_c,
                spike, V_post, W,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:6]):
                ctx.save_for_backward(
                    beta_c, omega_c, v_th_c, v_init_c, w_init_c,
                    V_post, W, spike,
                )
            ctx.K = K
            ctx.num_cols = num_cols
            ctx.alpha = alpha

            return spike, V_post, W

        @staticmethod
        def backward(ctx, grad_spike, grad_V_post, grad_W):
            beta, omega, v_th, v_init, w_init, V_post, W, spike = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols
            alpha = ctx.alpha

            if grad_spike is None:
                grad_spike = torch.zeros_like(spike)
            if grad_V_post is None:
                grad_V_post = torch.zeros_like(V_post)

            grad_spike_c = grad_spike.contiguous()
            grad_V_post_c = grad_V_post.contiguous()

            grad_beta = torch.empty_like(beta)
            grad_omega = torch.empty_like(omega)
            grad_u = torch.empty_like(beta)
            grad_v_th = torch.empty_like(v_th)
            grad_v_init = torch.empty_like(v_init)
            grad_w_init = torch.empty_like(w_init)

            BLOCK = _TritonRFPLIFForward._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_rf_plif_bwd_kernel[grid](
                beta, omega, v_th, v_init, w_init,
                V_post, W, spike,
                grad_spike_c, grad_V_post_c,
                grad_beta, grad_omega, grad_u, grad_v_th,
                grad_v_init, grad_w_init,
                K, num_cols, float(alpha),
                BLOCK=BLOCK,
            )

            return grad_beta, grad_omega, grad_u, grad_v_th, grad_v_init, grad_w_init, None

    class _TritonRFPLIFRecompute(torch.autograd.Function):
        """RF PLIF with V_post/W recomputation (no persistent V_post/W in autograd).

        Same as _TritonRFPLIFForward but:
        - Forward: only stores spike + V_last + W_last (no full V_post/W)
        - Backward: recomputes V_post/W from (beta, omega, u, v_th, spike, v_init, w_init)
        - save_for_backward: 5 large (beta, omega, u, v_th, spike) + 2 small (v_init, w_init)
          原版: 6 large (beta, omega, v_th, V_post, W, spike) + 2 small
        - 返回 (spike, V_last, W_last) 而非 (spike, V_post, W)，V_post/W 输出也不再分配

        Net savings: ~1GB/layer (save_for_backward 少 1 大张量 + 输出少 2 大张量)
        @ D=1024,N=8,B=2,TK=8192，12 层共省 ~12GB
        """

        _BLOCK = 128

        @staticmethod
        def forward(ctx, beta, omega, u, v_th, v_init, w_init, alpha):
            beta_c = beta.contiguous()
            omega_c = omega.contiguous()
            u_c = u.contiguous()
            v_th_c = v_th.contiguous()
            v_init_c = v_init.contiguous()
            w_init_c = w_init.contiguous()

            K = beta_c.shape[0]
            num_cols = beta_c[0].numel()

            spike = torch.empty_like(u_c)
            V_last = torch.empty_like(v_init_c)
            W_last = torch.empty_like(w_init_c)

            BLOCK = _TritonRFPLIFRecompute._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_rf_plif_fwd_recompute_kernel[grid](
                beta_c, omega_c, u_c, v_th_c, v_init_c, w_init_c,
                spike, V_last, W_last,
                K, num_cols,
                BLOCK=BLOCK,
            )

            if any(ctx.needs_input_grad[:6]):
                ctx.save_for_backward(
                    beta_c, omega_c, u_c, v_th_c, v_init_c, w_init_c, spike,
                )
            ctx.K = K
            ctx.num_cols = num_cols
            ctx.alpha = alpha

            return spike, V_last, W_last

        @staticmethod
        def backward(ctx, grad_spike, _grad_V_last, _grad_W_last):
            beta, omega, u, v_th, v_init, w_init, spike = ctx.saved_tensors
            K = ctx.K
            num_cols = ctx.num_cols
            alpha = ctx.alpha

            if grad_spike is None:
                grad_spike = torch.zeros_like(spike)
            grad_spike_c = grad_spike.contiguous()

            grad_beta = torch.empty_like(beta)
            grad_omega = torch.empty_like(omega)
            grad_u = torch.empty_like(u)
            grad_v_th = torch.empty_like(v_th)
            grad_v_init = torch.empty_like(v_init)
            grad_w_init = torch.empty_like(w_init)

            BLOCK = _TritonRFPLIFRecompute._BLOCK
            grid = ((num_cols + BLOCK - 1) // BLOCK,)

            _fused_rf_plif_bwd_recompute_kernel[grid](
                beta, omega, v_th, v_init, w_init,
                u, spike,
                grad_spike_c,
                grad_beta, grad_omega, grad_u, grad_v_th,
                grad_v_init, grad_w_init,
                K, num_cols, float(alpha),
                BLOCK=BLOCK,
            )

            return grad_beta, grad_omega, grad_u, grad_v_th, grad_v_init, grad_w_init, None

# Hillis-Steele parallel prefix scan (CPU fallback)
# ============================================================

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
        A_new_tail = A[d:] * A[:-d]
        B_new_tail = A[d:] * B[:-d] + B[d:]

        A = torch.cat([A[:d], A_new_tail], dim=0)
        B = torch.cat([B[:d], B_new_tail], dim=0)

        d *= 2

    return A, B


# ============================================================
# Public API: linear_recurrence
# ============================================================

def linear_recurrence(beta: torch.Tensor, u: torch.Tensor, v_init: torch.Tensor) -> torch.Tensor:
    """
    求解线性递推: V[k] = beta[k] * V[k-1] + u[k], V[-1] = v_init

    CUDA 后端: Triton fused kernel（1 次 kernel launch，O(K) 工作量）
    CPU 后端:  Hillis-Steele parallel scan（O(K log K) 工作量）

    Args:
        beta: (K, *shape) — 衰减系数，值域 (0, 1)
        u:    (K, *shape) — 输入项
        v_init: (*shape) — 初始状态

    Returns:
        V: (K, *shape) — 所有 K 步的状态
    """
    if _HAS_TRITON and beta.is_cuda:
        return _TritonLinearRecurrence.apply(beta, u, v_init)
    # CPU fallback
    A, B = hillis_steele_scan(beta, u)
    V = A * v_init.unsqueeze(0) + B
    return V


# ============================================================
# PLIF parallel forward (with spike iteration)
# ============================================================

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
        V_pre: (K, *shape) — 发放前膜电位（fused path 返回 None）
    """
    # Fused Triton path: single-pass sequential scan (exact, no iteration)
    # Replaces 3-phase approach with 1 kernel launch — ~3x faster forward, ~5x faster backward
    if (_HAS_TRITON and beta.is_cuda and surrogate_function is not None
            and hasattr(surrogate_function, 'alpha')
            and type(surrogate_function).__name__ == 'Sigmoid'):
        alpha = float(surrogate_function.alpha)
        spike, V_post = _TritonPLIFForward.apply(beta, u, v_th, v_init, alpha)
        return spike, V_post, None

    # Fallback: 3-phase approach (CPU, non-Sigmoid surrogates, or no surrogate)
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


def plif_rowparam_forward(
    beta_row: torch.Tensor,
    u: torch.Tensor,
    v_th_row: torch.Tensor,
    v_init: torch.Tensor,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    行参数 PLIF 前向：beta 和 v_th 在 K 步中保持恒定。

    比 plif_parallel_forward 快 ~40%（省去 expand+contiguous，减少 2/3 显存读取）。
    用于 ParametricLIFNode（固定 beta/v_th）或合并多个固定参数神经元。

    Args:
        beta_row: (*shape) — 每列的衰减率（所有 K 步相同）
        u:        (K, *shape) — 每步输入
        v_th_row: (*shape) — 每列的阈值（所有 K 步相同）
        v_init:   (*shape) — 初始膜电位
        surrogate_function: surrogate gradient 函数

    Returns:
        spike:  (K, *shape) — spike 模式
        V_post: (K, *shape) — 发放后膜电位
    """
    if (_HAS_TRITON and u.is_cuda and surrogate_function is not None
            and hasattr(surrogate_function, 'alpha')
            and type(surrogate_function).__name__ == 'Sigmoid'):
        alpha = float(surrogate_function.alpha)
        spike, V_post = _TritonPLIFRowParamForward.apply(
            beta_row, u, v_th_row, v_init, alpha,
        )
        return spike, V_post

    # Fallback: expand to full (K, *shape) and use standard path
    K = u.shape[0]
    beta = beta_row.unsqueeze(0).expand(K, *u.shape[1:]).contiguous()
    v_th = v_th_row.unsqueeze(0).expand(K, *u.shape[1:]).contiguous()
    spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=surrogate_function)
    return spike, V_post


def plif_rowparam_forward_alpha(
    beta_row: torch.Tensor,
    u: torch.Tensor,
    v_th_row: torch.Tensor,
    v_init: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    行参数 PLIF 前向（直接传入 alpha 浮点值，跳过 surrogate 对象创建）。

    与 plif_rowparam_forward 功能一致，但避免：
    - 创建 surrogate.Sigmoid 对象
    - hasattr / type().__name__ 检查
    直接将 alpha 传给 Triton kernel，节省 ~0.03ms/call。

    Args:
        beta_row: (*shape) — 每列的衰减率
        u:        (K, *shape) — 每步输入
        v_th_row: (*shape) — 每列的阈值
        v_init:   (*shape) — 初始膜电位
        alpha:    surrogate gradient alpha 值（float）

    Returns:
        spike:  (K, *shape)
        V_post: (K, *shape)
    """
    if _HAS_TRITON and u.is_cuda:
        spike, V_post = _TritonPLIFRowParamForward.apply(
            beta_row, u, v_th_row, v_init, alpha,
        )
        return spike, V_post

    # Fallback: expand to full (K, *shape) and use standard path with surrogate object
    from spikingjelly.activation_based import surrogate as _surrogate
    surr = _surrogate.Sigmoid(alpha=alpha)
    K = u.shape[0]
    beta = beta_row.unsqueeze(0).expand(K, *u.shape[1:]).contiguous()
    v_th = v_th_row.unsqueeze(0).expand(K, *u.shape[1:]).contiguous()
    spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=surr)
    return spike, V_post



def plif_rowparam_forward_recompute(
    beta_row: torch.Tensor,
    u: torch.Tensor,
    v_th_row: torch.Tensor,
    v_init: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    行参数 PLIF 前向，Mamba-style V_post 重计算（不持有 V_post 在 autograd 图中）。

    与 plif_rowparam_forward_alpha 计算一致，但 forward 不分配/保存 V_post 张量，
    backward 从 (u, beta, v_th, spike, v_init) 重计算 V_post — K 步 multiply-add
    纯寄存器操作，K≤32 时几乎零开销。

    节省 ~50% autograd 显存（无 V_post 在 ctx.saved_tensors 中）。
    最适合短序列 (K≤32) 场景。

    Args:
        beta_row: (*shape) — 每列的衰减率
        u:        (K, *shape) — 每步输入
        v_th_row: (*shape) — 每列的阈值
        v_init:   (*shape) — 初始膜电位
        alpha:    surrogate gradient alpha 值（float）

    Returns:
        spike:  (K, *shape)
        V_last: (*shape) — 末步 V_post（用于状态更新）
    """
    if _HAS_TRITON and u.is_cuda:
        spike, V_last = _TritonPLIFRowParamRecompute.apply(
            beta_row, u, v_th_row, v_init, alpha,
        )
        return spike, V_last

    # Fallback: use standard path (with V_post materialization)
    spike, V_post = plif_rowparam_forward_alpha(beta_row, u, v_th_row, v_init, alpha)
    return spike, V_post[-1]


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

    v7.4: scalar/0-dim beta 和 v_th 使用 row-param 内核（无需 expand 到 (K, *shape)）。

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
    shape = u.shape[1:]

    # Row-param fast path: beta 和 v_th 都是 scalar/0-dim → 扩展为 (*shape) 行向量
    beta_is_scalar = isinstance(beta, torch.Tensor) and beta.dim() == 0
    beta_is_float = not isinstance(beta, torch.Tensor)
    vth_is_scalar = isinstance(v_th, torch.Tensor) and v_th.dim() == 0
    vth_is_float = not isinstance(v_th, torch.Tensor)

    if (beta_is_scalar or beta_is_float) and (vth_is_scalar or vth_is_float):
        # Build row vectors (*shape)
        if beta_is_scalar:
            beta_row = beta.expand(*shape).contiguous()
        else:
            beta_row = torch.full(shape, beta, device=u.device, dtype=u.dtype)
        if vth_is_scalar:
            v_th_row = v_th.expand(*shape).contiguous()
        else:
            v_th_row = torch.full(shape, v_th, device=u.device, dtype=u.dtype)
        return plif_rowparam_forward(beta_row, u, v_th_row, v_init, surrogate_function)

    # Full-tensor path: expand to (K, *shape) if needed
    if isinstance(beta, torch.Tensor):
        if beta.dim() == 0:
            beta = beta.expand(K, *shape).contiguous()
    else:
        beta = torch.full_like(u, beta)

    if isinstance(v_th, torch.Tensor):
        if v_th.dim() == 0:
            v_th = v_th.expand(K, *shape).contiguous()
    else:
        v_th = torch.full_like(u, v_th)

    spike, V_post, _ = plif_parallel_forward(
        beta, u, v_th, v_init, max_iter, surrogate_function,
    )
    return spike, V_post


# ============================================================
# RF PLIF parallel forward (Resonate-and-Fire)
# ============================================================

def rf_plif_parallel_forward(
    beta: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    v_th: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resonate-and-Fire PLIF 并行前向：二阶耦合 V/W 动力学。

    求解:
      V_pre[k] = β[k]·V_post[k-1] - ω[k]·W[k-1] + u[k]
      W[k]     = ω[k]·V_post[k-1] + β[k]·W[k-1]
      s[k]     = Θ(V_pre[k] - V_th[k])
      V_post[k]= V_pre[k] - V_th[k]·s[k]

    物理直觉：V = 位移, W = 速度, ω = 弹簧刚度。
    脉冲响应 V[t] ∝ r^t·cos(θt)，阻尼振荡（共振效应）。

    Args:
        beta:  (K, *shape) — 衰减/阻尼系数
        omega: (K, *shape) — 振荡频率
        u:     (K, *shape) — 输入电流 α·I
        v_th:  (K, *shape) — 动态阈值
        v_init: (*shape) — 初始膜电位 V[0]
        w_init: (*shape) — 初始振荡状态 W[0]
        surrogate_function: surrogate gradient 函数

    Returns:
        spike:  (K, *shape) — 二值 spike
        V_post: (K, *shape) — 发放后膜电位
        W:      (K, *shape) — 振荡状态
    """
    # Fused Triton path
    if (_HAS_TRITON and beta.is_cuda and surrogate_function is not None
            and hasattr(surrogate_function, 'alpha')
            and type(surrogate_function).__name__ == 'Sigmoid'):
        alpha = float(surrogate_function.alpha)
        spike, V_post, W = _TritonRFPLIFForward.apply(
            beta, omega, u, v_th, v_init, w_init, alpha,
        )
        return spike, V_post, W

    # CPU / non-Sigmoid fallback: sequential loop
    K = beta.shape[0]
    device = beta.device
    dtype = beta.dtype

    spike_list = []
    V_post_list = []
    W_list = []

    v = v_init
    w = w_init

    for k in range(K):
        v_pre = beta[k] * v - omega[k] * w + u[k]
        w_new = omega[k] * v + beta[k] * w

        if surrogate_function is not None:
            s = surrogate_function(v_pre - v_th[k])
        else:
            s = (v_pre >= v_th[k]).float()

        v = v_pre - v_th[k] * s
        w = w_new

        spike_list.append(s)
        V_post_list.append(v)
        W_list.append(w)

    return torch.stack(spike_list), torch.stack(V_post_list), torch.stack(W_list)


# ============================================================
# RF PLIF parallel forward with V_post/W recompute
# ============================================================

def rf_plif_parallel_forward_recompute(
    beta: torch.Tensor,
    omega: torch.Tensor,
    u: torch.Tensor,
    v_th: torch.Tensor,
    v_init: torch.Tensor,
    w_init: torch.Tensor,
    surrogate_function=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    RF PLIF 前向（V_post/W recompute，省显存）。

    与 rf_plif_parallel_forward 计算一致，但 forward 不保存完整 V_post/W 张量，
    backward 从 (beta, omega, u, v_th, spike, v_init, w_init) 重计算。

    save_for_backward: 5 large (beta, omega, u, v_th, spike) + 2 small (v/w_init)
    原版: 6 large (beta, omega, v_th, V_post, W, spike) + 2 small
    每层省 ~1GB (save 少 1 大张量 + 输出不再分配 V_post/W) @ D=1024,N=8,B=2,TK=8192

    Returns:
        spike:  (K, *shape) — 二值 spike
        V_last: (*shape) — 末步 V_post（用于状态更新）
        W_last: (*shape) — 末步 W（用于状态更新）
    """
    if (_HAS_TRITON and beta.is_cuda and surrogate_function is not None
            and hasattr(surrogate_function, 'alpha')
            and type(surrogate_function).__name__ == 'Sigmoid'):
        alpha = float(surrogate_function.alpha)
        return _TritonRFPLIFRecompute.apply(
            beta, omega, u, v_th, v_init, w_init, alpha,
        )

    # CPU / non-Sigmoid fallback
    spike, V_post, W = rf_plif_parallel_forward(
        beta, omega, u, v_th, v_init, w_init, surrogate_function,
    )
    return spike, V_post[-1], W[-1]

