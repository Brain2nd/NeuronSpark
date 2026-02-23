"""
MoE Combine Triton Kernel: Fused weighted sum for Mixture-of-Experts output.

Forward:
  out[n, d] = shared[n, d] + Σ_e experts[e, n, d] × weights[n, e]

Backward:
  grad_shared = grad_out                         (直接传递)
  grad_expert[e, n, d] = grad_out[n, d] × weight[n, e]
  grad_weight[n, e] = Σ_d grad_out[n, d] × expert[e, n, d]

替代 Python 循环 `for e: out += expert_out * weight`，
将 E 次 mul + E 次 add 合并为 1 次 Triton kernel。
"""

import os
import torch

# DGX Spark (GB10, sm_121a): 需要系统 CUDA 13.0 的 ptxas
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
    def _moe_combine_fwd_kernel(
        SHARED_ptr, EXPERTS_ptr, WEIGHTS_ptr, OUT_ptr,
        N, D,
        stride_en, stride_ed,
        E: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Forward: out[n, d] = shared[n, d] + Σ_e experts[e, n, d] × weights[n, e]

        Grid: (N, ceil(D / BLOCK_D))
        Each program processes 1 row, BLOCK_D columns.
        Inner loop over E (constexpr → compiler unrolls).
        """
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load shared[n, d]
        shared_off = pid_n * D + cols
        acc = tl.load(SHARED_ptr + shared_off, mask=mask, other=0.0).to(tl.float32)

        # Accumulate Σ_e experts[e, n, d] × weights[n, e]
        for e in tl.static_range(E):
            w = tl.load(WEIGHTS_ptr + pid_n * E + e).to(tl.float32)
            expert_off = e * stride_en + pid_n * stride_ed + cols
            exp_val = tl.load(EXPERTS_ptr + expert_off, mask=mask, other=0.0).to(tl.float32)
            acc += exp_val * w

        tl.store(OUT_ptr + shared_off, acc, mask=mask)

    @triton.jit
    def _moe_combine_bwd_expert_kernel(
        GRAD_OUT_ptr, WEIGHTS_ptr, GRAD_EXPERTS_ptr,
        N, D,
        stride_en, stride_ed,
        E: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Backward for experts: grad_expert[e, n, d] = grad_out[n, d] × weight[n, e]

        Grid: (N * E, ceil(D / BLOCK_D))
        Each program processes 1 (n, e) pair, BLOCK_D columns.
        """
        pid_ne = tl.program_id(0)
        pid_d = tl.program_id(1)
        n = pid_ne // E
        e = pid_ne % E
        cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load weight[n, e]
        w = tl.load(WEIGHTS_ptr + n * E + e).to(tl.float32)

        # Load grad_out[n, d]
        grad_off = n * D + cols
        g = tl.load(GRAD_OUT_ptr + grad_off, mask=mask, other=0.0).to(tl.float32)

        # Store grad_expert[e, n, d]
        expert_off = e * stride_en + n * stride_ed + cols
        tl.store(GRAD_EXPERTS_ptr + expert_off, g * w, mask=mask)

    @triton.jit
    def _moe_combine_bwd_weights_kernel(
        GRAD_OUT_ptr, EXPERTS_ptr, GRAD_WEIGHTS_ptr,
        N, D,
        stride_en, stride_ed,
        E: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Backward for weights: grad_weight[n, e] = Σ_d grad_out[n, d] × expert[e, n, d]

        Grid: (N * E,)
        Each program handles one (n, e) pair, loops over D/BLOCK_D chunks with register accumulation.
        """
        pid = tl.program_id(0)
        n = pid // E
        e = pid % E

        # Accumulate in BLOCK_D-sized vector, reduce to scalar at the end
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        num_blocks = tl.cdiv(D, BLOCK_D)
        for b in range(num_blocks):
            cols = b * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = cols < D

            grad_off = n * D + cols
            g = tl.load(GRAD_OUT_ptr + grad_off, mask=mask, other=0.0).to(tl.float32)

            expert_off = e * stride_en + n * stride_ed + cols
            exp_val = tl.load(EXPERTS_ptr + expert_off, mask=mask, other=0.0).to(tl.float32)

            acc += tl.where(mask, g * exp_val, 0.0)

        # Reduce BLOCK_D → scalar, then store
        result = tl.sum(acc)
        tl.store(GRAD_WEIGHTS_ptr + n * E + e, result)

    class MoECombine(torch.autograd.Function):
        """Fused MoE combine: shared + Σ_e experts[e] × weights[:, e]

        Args (forward):
            shared:  (N, D) — shared expert output
            experts: (E, N, D) — all routed expert outputs
            weights: (N, E) — routing weights per sample per expert
        """

        _BLOCK_D = 128

        @staticmethod
        def forward(ctx, shared, experts, weights):
            shared_c = shared.contiguous()
            experts_c = experts.contiguous()
            weights_c = weights.contiguous()

            E, N, D = experts_c.shape
            BLOCK_D = MoECombine._BLOCK_D

            out = torch.empty_like(shared_c)

            # experts strides: (E, N, D) → stride_en = N*D, stride_ed = D
            stride_en = N * D
            stride_ed = D

            grid = (N, (D + BLOCK_D - 1) // BLOCK_D)
            _moe_combine_fwd_kernel[grid](
                shared_c, experts_c, weights_c, out,
                N, D,
                stride_en, stride_ed,
                E=E, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(experts_c, weights_c)
            ctx.E = E
            ctx.N = N
            ctx.D = D

            return out

        @staticmethod
        def backward(ctx, grad_out):
            experts, weights = ctx.saved_tensors
            E, N, D = ctx.E, ctx.N, ctx.D
            BLOCK_D = MoECombine._BLOCK_D

            grad_out_c = grad_out.contiguous()

            stride_en = N * D
            stride_ed = D

            # grad_shared = grad_out (直接传递)
            grad_shared = grad_out_c

            # grad_experts: (E, N, D)
            grad_experts = torch.empty_like(experts)
            grid_exp = (N * E, (D + BLOCK_D - 1) // BLOCK_D)
            _moe_combine_bwd_expert_kernel[grid_exp](
                grad_out_c, weights, grad_experts,
                N, D,
                stride_en, stride_ed,
                E=E, BLOCK_D=BLOCK_D,
            )

            # grad_weights: (N, E)
            grad_weights = torch.empty_like(weights)
            grid_w = (N * E,)
            _moe_combine_bwd_weights_kernel[grid_w](
                grad_out_c, experts, grad_weights,
                N, D,
                stride_en, stride_ed,
                E=E, BLOCK_D=BLOCK_D,
            )

            return grad_shared, grad_experts, grad_weights


def moe_combine(shared: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Fused MoE output combine: shared + Σ_e experts[e] × weights[:, e]

    Args:
        shared:  (N, D) — shared expert output
        experts: (E, N, D) — all routed expert outputs
        weights: (N, E) — routing weights

    Returns:
        out: (N, D) — combined output
    """
    if _HAS_TRITON and shared.is_cuda:
        return MoECombine.apply(shared, experts, weights)
    # CPU fallback
    return shared + torch.einsum('end,ne->nd', experts, weights)
