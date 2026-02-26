"""
Benchmark: PLIF+SEW fusion (P1) + gate·skip·scale fusion (P2)

对比 SNNBlock.forward_parallel 的 fused vs unfused 实现。
使用 CUDA events 精确测量 forward + backward wall time。

用法:
    python bench_sew_fusion.py
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from spikingjelly.activation_based import functional


# ============================================================
# Unfused baseline: 手动还原原始实现
# ============================================================

def _forward_unfused(block, spike_in_seq):
    """原始 forward_parallel（unfused: 分离的 gate·skip, 分离的 SEW add）"""
    from atomic_ops.parallel_scan import plif_rowparam_forward, rf_plif_parallel_forward
    from atomic_ops.snn_block_sew import _fused_modulation_rf

    TK, batch, D = spike_in_seq.shape
    DN = block.D * block.N

    # Phase 1: 批量投影
    flat = spike_in_seq.reshape(TK * batch, D)
    W_dn5 = torch.cat([
        block.W_in.weight, block.W_beta_x.weight, block.W_alpha_x.weight,
        block.W_th_x.weight, block.W_omega_x.weight,
    ], dim=0)
    proj_dn5 = F.linear(flat, W_dn5)
    I_all, raw_beta, raw_alpha, raw_th, raw_omega = proj_dn5.split(DN, dim=-1)
    I_all = I_all.reshape(TK, batch, DN)
    raw_beta = raw_beta.reshape(TK, batch, DN)
    raw_alpha = raw_alpha.reshape(TK, batch, DN)
    raw_th = raw_th.reshape(TK, batch, DN)
    raw_omega = raw_omega.reshape(TK, batch, DN)

    W_d2 = torch.cat([block.W_gate.weight, block.W_skip.weight], dim=0)
    proj_d2 = F.linear(flat, W_d2).reshape(TK, batch, 2 * D)
    gate_all = torch.sigmoid(proj_d2[:, :, :D])
    I_skip_all = proj_d2[:, :, D:]

    # Phase 1b
    beta_all, u_hidden, v_th_all, omega_all = _fused_modulation_rf(
        raw_beta, block.b_beta, raw_alpha, block.b_alpha,
        raw_th, block.b_th, block.v_th_min, I_all,
        raw_omega, block.b_omega,
    )

    # Phase 2: RF PLIF
    v_init_hidden = block.hidden_neuron.v
    if isinstance(v_init_hidden, float):
        v_init_hidden = block.hidden_neuron.expand_v_init(batch, flat.device, flat.dtype)
    w_init_hidden = block.hidden_neuron.w
    if isinstance(w_init_hidden, float):
        w_init_hidden = block.hidden_neuron.expand_w_init(batch, flat.device, flat.dtype)

    s_hidden, V_post_hidden, W_hidden = rf_plif_parallel_forward(
        beta_all, omega_all, u_hidden, v_th_all,
        v_init_hidden, w_init_hidden,
        surrogate_function=block.hidden_neuron.get_surrogate(u_hidden),
    )
    block.hidden_neuron.v = V_post_hidden[-1].detach()
    block.hidden_neuron.w = W_hidden[-1].detach()

    # Phase 4: UNFUSED gate·skip + scale
    s_flat = s_hidden.reshape(TK * batch, DN)
    I_out_all = F.linear(s_flat, block.W_out.weight).reshape(TK, batch, D)
    I_total_all = I_out_all * gate_all + I_skip_all
    beta_out = block.output_neuron.beta
    u_output = (1.0 - beta_out) * I_total_all

    # Phase 5: UNFUSED PLIF
    v_init_output = block.output_neuron.v
    if isinstance(v_init_output, float):
        v_init_output = block.output_neuron.expand_v_init(batch, flat.device, flat.dtype)

    beta_out_row = beta_out.unsqueeze(0).expand(batch, D).contiguous()
    v_th_out_row = block.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

    spike_out, V_post_output = plif_rowparam_forward(
        beta_out_row, u_output, v_th_out_row, v_init_output,
        surrogate_function=block.output_neuron.get_surrogate(u_output),
    )
    block.output_neuron.v = V_post_output[-1].detach()

    # Phase 6: UNFUSED SEW add
    return spike_out + spike_in_seq


# ============================================================
# Benchmark runner
# ============================================================

def benchmark(fn, block, spike_in, warmup=5, repeats=20):
    """CUDA events 精确测量 forward + backward。"""
    for _ in range(warmup):
        functional.reset_net(block)
        out = fn(block, spike_in)
        out.sum().backward()
        block.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    fwd_times, bwd_times = [], []
    for _ in range(repeats):
        functional.reset_net(block)
        torch.cuda.synchronize()

        s_f = torch.cuda.Event(enable_timing=True)
        e_f = torch.cuda.Event(enable_timing=True)
        s_b = torch.cuda.Event(enable_timing=True)
        e_b = torch.cuda.Event(enable_timing=True)

        s_f.record()
        out = fn(block, spike_in)
        e_f.record()

        loss = out.sum()
        s_b.record()
        loss.backward()
        e_b.record()

        torch.cuda.synchronize()
        fwd_times.append(s_f.elapsed_time(e_f))
        bwd_times.append(s_b.elapsed_time(e_b))
        block.zero_grad(set_to_none=True)

    def median(arr):
        s = sorted(arr)
        return s[len(s) // 2]

    fwd = median(fwd_times)
    bwd = median(bwd_times)
    return {'fwd_ms': fwd, 'bwd_ms': bwd, 'total_ms': fwd + bwd}


def main():
    device = 'cuda'
    dtype = torch.float32

    configs = [
        (256, 8, 16, 4, 512, "D=256 N=8 K=16 B=4 S=512"),
        (512, 8, 16, 4, 512, "D=512 N=8 K=16 B=4 S=512"),
        (1024, 8, 16, 2, 512, "D=1024 N=8 K=16 B=2 S=512"),
        (256, 8, 16, 8, 256, "D=256 N=8 K=16 B=8 S=256"),
    ]

    print("=" * 80)
    print("Benchmark: PLIF+SEW fusion (P1) + gate·skip·scale fusion (P2)")
    print("=" * 80)

    for D, N, K, batch, seq_len, label in configs:
        TK = seq_len * K

        from atomic_ops.snn_block_sew import SNNBlock
        block = SNNBlock(D=D, N=N, v_th_min=0.1, output_v_threshold=0.3).to(device, dtype)

        spike_in = (torch.rand(TK, batch, D, device=device, dtype=dtype) > 0.85).float()
        spike_in.requires_grad_(True)

        print(f"\n--- {label} (TK={TK}) ---")

        # Fused
        print("  Fused (P1+P2)...", end=" ", flush=True)
        fused = benchmark(lambda b, s: b.forward_parallel(s), block, spike_in)
        print(f"fwd={fused['fwd_ms']:.2f}ms  bwd={fused['bwd_ms']:.2f}ms  total={fused['total_ms']:.2f}ms")

        functional.reset_net(block)

        # Unfused
        print("  Unfused...", end=" ", flush=True)
        unfused = benchmark(_forward_unfused, block, spike_in)
        print(f"fwd={unfused['fwd_ms']:.2f}ms  bwd={unfused['bwd_ms']:.2f}ms  total={unfused['total_ms']:.2f}ms")

        # Speedup
        sp_f = unfused['fwd_ms'] / fused['fwd_ms']
        sp_b = unfused['bwd_ms'] / fused['bwd_ms']
        sp_t = unfused['total_ms'] / fused['total_ms']
        print(f"  Speedup: fwd={sp_f:.3f}x  bwd={sp_b:.3f}x  total={sp_t:.3f}x")

        # Correctness
        functional.reset_net(block)
        out_f = block.forward_parallel(spike_in)
        functional.reset_net(block)
        out_u = _forward_unfused(block, spike_in)
        diff = (out_f - out_u).abs().max().item()
        print(f"  Correctness: max_diff={diff:.2e} {'PASS' if diff < 1e-4 else 'FAIL'}")

        del block, spike_in
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
