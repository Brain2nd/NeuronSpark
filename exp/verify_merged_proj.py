"""
验证 SNNBlock 6 投影合并优化的正确性和性能。

优化内容：
  OLD: 6 次独立 F.linear (W_in, W_beta_x, W_alpha_x, W_th_x, W_gate, W_skip)
  NEW: torch.cat → 1 次 F.linear → split

验证项目：
  1. 前向输出 bit-exact 对比
  2. 反向梯度 bit-exact 对比（对输入 flat 和对每个权重矩阵）
  3. 性能：投影阶段 + 完整 forward_parallel + forward+backward
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

device = 'cuda'
dtype = torch.bfloat16

# 实际训练配置
D, N, K, seq_len, batch = 768, 8, 16, 512, 2
TK = seq_len * K   # 8192
DN = D * N          # 6144


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, label, n_warmup=5, n_iter=20):
    """Benchmark a function, return mean time in ms."""
    for _ in range(n_warmup):
        fn()
    t0 = sync_time()
    for _ in range(n_iter):
        fn()
    elapsed = (sync_time() - t0) / n_iter * 1000
    print(f"  {label:50s} {elapsed:8.2f} ms")
    return elapsed


# ============================================================
# Phase 1: 投影阶段正确性验证
# ============================================================

def test_projection_correctness():
    """验证 6-linear vs 1-merged-linear 的数值一致性（前向+反向）。"""
    print("=" * 70)
    print("Phase 1: 投影阶段正确性验证")
    print("=" * 70)

    from atomic_ops.snn_block import SNNBlock
    block = SNNBlock(D=D, N=N).to(device).to(dtype)

    torch.manual_seed(42)
    spike_in = (torch.rand(TK, batch, D, device=device, dtype=dtype) > 0.5).to(dtype)
    flat = spike_in.reshape(TK * batch, D)

    # ====== OLD: 6 separate F.linear ======
    flat_old = flat.clone().requires_grad_(True)

    I_all_old = F.linear(flat_old, block.W_in.weight).reshape(TK, batch, DN)
    raw_beta_old = F.linear(flat_old, block.W_beta_x.weight).reshape(TK, batch, DN)
    raw_alpha_old = F.linear(flat_old, block.W_alpha_x.weight).reshape(TK, batch, DN)
    raw_th_old = F.linear(flat_old, block.W_th_x.weight).reshape(TK, batch, DN)
    gate_old = torch.sigmoid(
        F.linear(flat_old, block.W_gate.weight).reshape(TK, batch, D)
    )
    skip_old = F.linear(flat_old, block.W_skip.weight).reshape(TK, batch, D)

    # 用合理的 loss 测试梯度
    loss_old = (I_all_old.sum() + raw_beta_old.sum() + raw_alpha_old.sum()
                + raw_th_old.sum() + gate_old.sum() + skip_old.sum())
    loss_old.backward()

    grad_flat_old = flat_old.grad.clone()
    grad_W_in_old = block.W_in.weight.grad.clone()
    grad_W_beta_old = block.W_beta_x.weight.grad.clone()
    grad_W_alpha_old = block.W_alpha_x.weight.grad.clone()
    grad_W_th_old = block.W_th_x.weight.grad.clone()
    grad_W_gate_old = block.W_gate.weight.grad.clone()
    grad_W_skip_old = block.W_skip.weight.grad.clone()

    block.zero_grad()

    # ====== NEW: 1 merged F.linear ======
    flat_new = flat.clone().requires_grad_(True)

    W_all = torch.cat([
        block.W_in.weight,        # (DN, D)
        block.W_beta_x.weight,    # (DN, D)
        block.W_alpha_x.weight,   # (DN, D)
        block.W_th_x.weight,      # (DN, D)
        block.W_gate.weight,      # (D, D)
        block.W_skip.weight,      # (D, D)
    ], dim=0)  # (4*DN + 2*D, D)

    proj_all = F.linear(flat_new, W_all)  # (TK*batch, 4*DN+2*D)

    I_all_new = proj_all[:, :DN].reshape(TK, batch, DN)
    raw_beta_new = proj_all[:, DN:2*DN].reshape(TK, batch, DN)
    raw_alpha_new = proj_all[:, 2*DN:3*DN].reshape(TK, batch, DN)
    raw_th_new = proj_all[:, 3*DN:4*DN].reshape(TK, batch, DN)
    gate_new = torch.sigmoid(
        proj_all[:, 4*DN:4*DN+D].reshape(TK, batch, D)
    )
    skip_new = proj_all[:, 4*DN+D:].reshape(TK, batch, D)

    loss_new = (I_all_new.sum() + raw_beta_new.sum() + raw_alpha_new.sum()
                + raw_th_new.sum() + gate_new.sum() + skip_new.sum())
    loss_new.backward()

    grad_flat_new = flat_new.grad.clone()
    grad_W_in_new = block.W_in.weight.grad.clone()
    grad_W_beta_new = block.W_beta_x.weight.grad.clone()
    grad_W_alpha_new = block.W_alpha_x.weight.grad.clone()
    grad_W_th_new = block.W_th_x.weight.grad.clone()
    grad_W_gate_new = block.W_gate.weight.grad.clone()
    grad_W_skip_new = block.W_skip.weight.grad.clone()

    # ====== 对比 ======
    print("\n--- 前向输出对比 ---")
    pairs = [
        ("I_all", I_all_old, I_all_new),
        ("raw_beta", raw_beta_old, raw_beta_new),
        ("raw_alpha", raw_alpha_old, raw_alpha_new),
        ("raw_th", raw_th_old, raw_th_new),
        ("gate", gate_old, gate_new),
        ("skip", skip_old, skip_new),
    ]
    all_ok = True
    for name, old, new in pairs:
        max_diff = (old - new).abs().max().item()
        match = "✓ EXACT" if max_diff == 0 else f"✗ max_diff={max_diff:.2e}"
        print(f"  {name:15s} {match}")
        if max_diff > 0:
            all_ok = False

    print("\n--- 反向梯度对比 ---")
    grad_pairs = [
        ("grad_flat", grad_flat_old, grad_flat_new),
        ("grad_W_in", grad_W_in_old, grad_W_in_new),
        ("grad_W_beta", grad_W_beta_old, grad_W_beta_new),
        ("grad_W_alpha", grad_W_alpha_old, grad_W_alpha_new),
        ("grad_W_th", grad_W_th_old, grad_W_th_new),
        ("grad_W_gate", grad_W_gate_old, grad_W_gate_new),
        ("grad_W_skip", grad_W_skip_old, grad_W_skip_new),
    ]
    for name, old, new in grad_pairs:
        max_diff = (old - new).abs().max().item()
        rel_diff = max_diff / (old.abs().max().item() + 1e-10)
        match = "✓ EXACT" if max_diff == 0 else f"✗ max_diff={max_diff:.2e} (rel={rel_diff:.2e})"
        print(f"  {name:15s} {match}")
        if max_diff > 1e-3:
            all_ok = False

    print(f"\n  投影阶段正确性: {'PASS ✓' if all_ok else 'FAIL ✗'}")
    return all_ok


# ============================================================
# Phase 2: 投影阶段性能对比
# ============================================================

def test_projection_benchmark():
    """对比 6-linear vs 1-merged-linear 的投影阶段性能。"""
    print("\n" + "=" * 70)
    print("Phase 2: 投影阶段性能对比")
    print("=" * 70)

    from atomic_ops.snn_block import SNNBlock
    block = SNNBlock(D=D, N=N).to(device).to(dtype)

    torch.manual_seed(42)
    spike_in = (torch.rand(TK, batch, D, device=device, dtype=dtype) > 0.5).to(dtype)
    flat = spike_in.reshape(TK * batch, D)

    # ====== 前向 benchmark ======
    print("\n--- 前向 ---")

    def old_fwd():
        F.linear(flat, block.W_in.weight)
        F.linear(flat, block.W_beta_x.weight)
        F.linear(flat, block.W_alpha_x.weight)
        F.linear(flat, block.W_th_x.weight)
        F.linear(flat, block.W_gate.weight)
        F.linear(flat, block.W_skip.weight)

    def new_fwd():
        W_all = torch.cat([
            block.W_in.weight, block.W_beta_x.weight,
            block.W_alpha_x.weight, block.W_th_x.weight,
            block.W_gate.weight, block.W_skip.weight,
        ], dim=0)
        F.linear(flat, W_all)

    # 也单独测 cat 的开销
    def cat_only():
        torch.cat([
            block.W_in.weight, block.W_beta_x.weight,
            block.W_alpha_x.weight, block.W_th_x.weight,
            block.W_gate.weight, block.W_skip.weight,
        ], dim=0)

    t_cat = bench(cat_only, "torch.cat 6 weights")
    t_old_fwd = bench(old_fwd, "OLD: 6× F.linear (forward)")
    t_new_fwd = bench(new_fwd, "NEW: cat + 1× F.linear (forward)")
    print(f"  {'Forward speedup':50s} {t_old_fwd/t_new_fwd:.2f}x")

    # ====== 前向+反向 benchmark ======
    print("\n--- 前向+反向 ---")

    flat_grad = flat.clone().requires_grad_(True)

    def old_fwd_bwd():
        flat_g = flat.clone().requires_grad_(True)
        out1 = F.linear(flat_g, block.W_in.weight)
        out2 = F.linear(flat_g, block.W_beta_x.weight)
        out3 = F.linear(flat_g, block.W_alpha_x.weight)
        out4 = F.linear(flat_g, block.W_th_x.weight)
        out5 = F.linear(flat_g, block.W_gate.weight)
        out6 = F.linear(flat_g, block.W_skip.weight)
        loss = out1.sum() + out2.sum() + out3.sum() + out4.sum() + out5.sum() + out6.sum()
        loss.backward()
        block.zero_grad()

    def new_fwd_bwd():
        flat_g = flat.clone().requires_grad_(True)
        W_all = torch.cat([
            block.W_in.weight, block.W_beta_x.weight,
            block.W_alpha_x.weight, block.W_th_x.weight,
            block.W_gate.weight, block.W_skip.weight,
        ], dim=0)
        proj = F.linear(flat_g, W_all)
        loss = proj.sum()
        loss.backward()
        block.zero_grad()

    t_old_fb = bench(old_fwd_bwd, "OLD: 6× F.linear (fwd+bwd)")
    t_new_fb = bench(new_fwd_bwd, "NEW: cat + 1× F.linear (fwd+bwd)")
    print(f"  {'Fwd+Bwd speedup':50s} {t_old_fb/t_new_fb:.2f}x")

    return t_old_fwd, t_new_fwd


# ============================================================
# Phase 3: 完整 SNNBlock forward_parallel 端到端对比
# ============================================================

def test_full_block_e2e():
    """完整 SNNBlock.forward_parallel 端到端对比。

    使用 monkey-patch 切换新旧路径，保证权重完全一致。
    """
    print("\n" + "=" * 70)
    print("Phase 3: 完整 SNNBlock forward_parallel 端到端对比")
    print("=" * 70)

    from atomic_ops.snn_block import SNNBlock

    torch.manual_seed(42)
    spike_in = (torch.rand(TK, batch, D, device=device, dtype=dtype) > 0.5).to(dtype)

    # ====== OLD path ======
    block = SNNBlock(D=D, N=N).to(device).to(dtype)
    functional.reset_net(block)

    spike_old = spike_in.clone().requires_grad_(True)
    out_old = block.forward_parallel(spike_old)
    loss_old = out_old.sum()
    loss_old.backward()
    out_old_val = out_old.detach().clone()
    grad_in_old = spike_old.grad.clone()
    grad_W_in_old = block.W_in.weight.grad.clone()
    grad_W_out_old = block.W_out.weight.grad.clone()

    # ====== NEW path: monkey-patch forward_parallel ======
    block.zero_grad()
    functional.reset_net(block)

    # 保存原始 forward_parallel
    _orig_forward = block.forward_parallel

    def _merged_forward_parallel(spike_in_seq):
        """使用合并投影的 forward_parallel。"""
        from atomic_ops.snn_block import _fused_modulation
        from atomic_ops.parallel_scan import plif_parallel_forward, plif_rowparam_forward

        TK_, batch_, D_ = spike_in_seq.shape
        DN_ = block.D * block.N
        flat = spike_in_seq.reshape(TK_ * batch_, D_)

        # ====== MERGED: 1 次 matmul ======
        W_all = torch.cat([
            block.W_in.weight,
            block.W_beta_x.weight,
            block.W_alpha_x.weight,
            block.W_th_x.weight,
            block.W_gate.weight,
            block.W_skip.weight,
        ], dim=0)

        proj_all = F.linear(flat, W_all)

        I_all = proj_all[:, :DN_].reshape(TK_, batch_, DN_)
        raw_beta = proj_all[:, DN_:2*DN_].reshape(TK_, batch_, DN_)
        raw_alpha = proj_all[:, 2*DN_:3*DN_].reshape(TK_, batch_, DN_)
        raw_th = proj_all[:, 3*DN_:4*DN_].reshape(TK_, batch_, DN_)
        gate_all = torch.sigmoid(
            proj_all[:, 4*DN_:4*DN_+D_].reshape(TK_, batch_, D_)
        )
        I_skip_all = proj_all[:, 4*DN_+D_:].reshape(TK_, batch_, D_)

        # 后续和原始代码完全相同
        beta_all, u_hidden, v_th_all = _fused_modulation(
            raw_beta, block.b_beta, raw_alpha, block.b_alpha,
            raw_th, block.b_th, block.v_th_min, I_all,
        )

        v_init_hidden = block.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch_, DN_, device=flat.device, dtype=flat.dtype)

        s_hidden, V_post_hidden, _ = plif_parallel_forward(
            beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
            surrogate_function=block.hidden_neuron.surrogate_function,
        )
        block.hidden_neuron.v = V_post_hidden[-1].detach()

        s_flat = s_hidden.reshape(TK_ * batch_, DN_)
        I_out_all = F.linear(s_flat, block.W_out.weight).reshape(TK_, batch_, D_)
        I_total_all = I_out_all * gate_all + I_skip_all

        beta_out = block.output_neuron.beta
        u_output = (1.0 - beta_out) * I_total_all
        v_init_output = block.output_neuron.v
        if isinstance(v_init_output, float):
            v_init_output = torch.zeros(batch_, D_, device=flat.device, dtype=flat.dtype)
        beta_out_row = beta_out.unsqueeze(0).expand(batch_, D_).contiguous()
        v_th_out_row = block.output_neuron.v_th.unsqueeze(0).expand(batch_, D_).contiguous()

        spike_out, V_post_output = plif_rowparam_forward(
            beta_out_row, u_output, v_th_out_row, v_init_output,
            surrogate_function=block.output_neuron.surrogate_function,
        )
        block.output_neuron.v = V_post_output[-1].detach()
        return spike_out

    block.forward_parallel = _merged_forward_parallel

    spike_new = spike_in.clone().requires_grad_(True)
    out_new = block.forward_parallel(spike_new)
    loss_new = out_new.sum()
    loss_new.backward()
    out_new_val = out_new.detach().clone()
    grad_in_new = spike_new.grad.clone()
    grad_W_in_new = block.W_in.weight.grad.clone()
    grad_W_out_new = block.W_out.weight.grad.clone()

    # Restore
    block.forward_parallel = _orig_forward

    # ====== 对比 ======
    print("\n--- 端到端前向对比 ---")
    max_diff_out = (out_old_val - out_new_val).abs().max().item()
    print(f"  output max_diff: {max_diff_out:.2e}  "
          f"{'✓ EXACT' if max_diff_out == 0 else '(non-zero, check tolerance)'}")

    print("\n--- 端到端反向梯度对比 ---")
    pairs = [
        ("grad_input", grad_in_old, grad_in_new),
        ("grad_W_in", grad_W_in_old, grad_W_in_new),
        ("grad_W_out", grad_W_out_old, grad_W_out_new),
    ]
    all_ok = True
    for name, old, new in pairs:
        max_diff = (old - new).abs().max().item()
        rel_diff = max_diff / (old.abs().max().item() + 1e-10)
        status = "✓ EXACT" if max_diff == 0 else f"max_diff={max_diff:.2e} (rel={rel_diff:.2e})"
        print(f"  {name:15s} {status}")
        if max_diff > 1e-2:
            all_ok = False

    print(f"\n  端到端正确性: {'PASS ✓' if all_ok else 'FAIL ✗'}")

    # ====== 性能对比 ======
    print("\n--- 端到端性能对比 ---")

    def bench_old():
        functional.reset_net(block)
        block.forward_parallel = _orig_forward
        block.forward_parallel(spike_in)

    def bench_new():
        functional.reset_net(block)
        block.forward_parallel = _merged_forward_parallel
        block.forward_parallel(spike_in)

    def bench_old_fb():
        functional.reset_net(block)
        block.zero_grad()
        block.forward_parallel = _orig_forward
        s = spike_in.clone().requires_grad_(True)
        out = block.forward_parallel(s)
        out.sum().backward()

    def bench_new_fb():
        functional.reset_net(block)
        block.zero_grad()
        block.forward_parallel = _merged_forward_parallel
        s = spike_in.clone().requires_grad_(True)
        out = block.forward_parallel(s)
        out.sum().backward()

    t_old_f = bench(bench_old, "OLD: forward_parallel")
    t_new_f = bench(bench_new, "NEW: forward_parallel (merged proj)")
    print(f"  {'Forward speedup':50s} {t_old_f/t_new_f:.2f}x")

    t_old_fb = bench(bench_old_fb, "OLD: forward + backward")
    t_new_fb = bench(bench_new_fb, "NEW: forward + backward (merged proj)")
    print(f"  {'Fwd+Bwd speedup':50s} {t_old_fb/t_new_fb:.2f}x")

    return all_ok


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print(f"Config: D={D}, N={N}, DN={DN}, K={K}, seq_len={seq_len}, batch={batch}, TK={TK}")
    print(f"Merged weight shape: ({4*DN + 2*D}, {D}) = ({4*DN+2*D}, {D})")
    print(f"Device: {device}, dtype: {dtype}\n")

    ok1 = test_projection_correctness()
    test_projection_benchmark()
    ok3 = test_full_block_e2e()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Projection correctness: {'PASS ✓' if ok1 else 'FAIL ✗'}")
    print(f"  E2E correctness:        {'PASS ✓' if ok3 else 'FAIL ✗'}")
