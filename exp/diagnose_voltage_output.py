"""
验证关键修复：用 V_post（膜电压）代替 spike 作为 W_out 输入。

原理：
  当前: s_hidden (binary 0/1) → W_out → gate → output_neuron → spike_out
    ∂spike/∂β = surrogate'(V-v_th) · V_post[t-1] ≈ 0 (大部分时刻)

  修改: V_post (continuous) → W_out → gate → continuous_out
    ∂output/∂β = W_out · (∂V_post/∂β) = W_out · V_post[t-1] · (1 - surrogate'·v_th) ≈ 0.7·W_out·V_post[t-1]
    梯度在所有时刻都存在，不仅仅在阈值附近

  这消除了 hidden_neuron surrogate 对 β 梯度的阻断效应。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


class VoltageOutputPatch:
    """猴子补丁：用 V_post 代替 spike 作为 W_out 的输入"""

    @staticmethod
    def patch_forward_parallel(block, spike_in_seq):
        from atomic_ops.parallel_scan import plif_parallel_forward
        from atomic_ops.snn_block import _fused_modulation

        TK, batch, D = spike_in_seq.shape
        DN = block.D * block.N

        flat = spike_in_seq.reshape(TK * batch, D)

        I_all = F.linear(flat, block.W_in.weight).reshape(TK, batch, DN)
        raw_beta = F.linear(flat, block.W_beta_x.weight).reshape(TK, batch, DN)
        raw_alpha = F.linear(flat, block.W_alpha_x.weight).reshape(TK, batch, DN)
        raw_th = F.linear(flat, block.W_th_x.weight).reshape(TK, batch, DN)
        gate_all = torch.sigmoid(
            F.linear(flat, block.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, block.W_skip.weight).reshape(TK, batch, D)

        beta_all, u_hidden, v_th_all = _fused_modulation(
            raw_beta, block.b_beta, raw_alpha, block.b_alpha,
            raw_th, block.b_th, block.v_th_min, I_all,
        )

        v_init_hidden = block.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch, DN, device=flat.device, dtype=flat.dtype)

        s_hidden, V_post_hidden, _ = plif_parallel_forward(
            beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
            surrogate_function=block.hidden_neuron.surrogate_function,
        )
        block.hidden_neuron.v = V_post_hidden[-1].detach()

        # ====== 关键修改：用 V_post 代替 spike ======
        # 原: I_out = W_out · s_hidden (binary, surrogate gradient阻断β)
        # 新: I_out = W_out · V_post_hidden (continuous, gradient直通β)
        v_flat = V_post_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(v_flat, block.W_out.weight).reshape(TK, batch, D)
        I_total_all = I_out_all * gate_all + I_skip_all

        # 不经过 output_neuron，直接返回连续值
        return I_total_all


def run_comparison(D=256, N=8, num_layers=4, D_ff=512,
                   seq_len=32, batch_size=2, num_steps=10):
    """对比: 原始 vs spike输出(无output_neuron) vs V_post输出"""
    device = 'cuda'

    variants = ['original', 'voltage_output']
    results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"{'='*60}")

        model = SNNLanguageModel(
            vocab_size=6144, D=D, N=N, num_layers=num_layers, D_ff=D_ff,
        ).to(device)

        if variant == 'voltage_output':
            for layer_module in model.layers:
                block = layer_module.snn_block
                block.forward_parallel = lambda x, b=block: VoltageOutputPatch.patch_forward_parallel(b, x)

        grad_history = {i: [] for i in range(num_layers)}
        w_in_grads = {i: [] for i in range(num_layers)}
        w_beta_x_grads = {i: [] for i in range(num_layers)}

        for step in range(num_steps):
            model.train()
            model.zero_grad()

            token_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)
            target_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(token_ids, target_ids)
                loss = out.last_loss.mean()

            loss.backward()

            for i, layer_module in enumerate(model.layers):
                g = layer_module.snn_block.b_beta.grad
                if g is not None:
                    grad_history[i].append(g.clone())
                g_w = layer_module.snn_block.W_in.weight.grad
                if g_w is not None:
                    w_in_grads[i].append(g_w.abs().mean().item())
                g_wb = layer_module.snn_block.W_beta_x.weight.grad
                if g_wb is not None:
                    w_beta_x_grads[i].append(g_wb.abs().mean().item())

        # 分析
        for i in [0, num_layers - 1]:
            grads = grad_history[i]
            if len(grads) < 2:
                continue
            grad_stack = torch.stack(grads, dim=0)
            signs = (grad_stack > 0).float()
            sign_consistency = signs.mean(dim=0)

            consistent = ((sign_consistency > 0.7) | (sign_consistency < 0.3)).float().mean().item()
            avg_abs_grad = torch.stack([g.abs().mean() for g in grads]).mean().item()

            w_in_avg = sum(w_in_grads[i]) / len(w_in_grads[i]) if w_in_grads[i] else 0
            w_beta_avg = sum(w_beta_x_grads[i]) / len(w_beta_x_grads[i]) if w_beta_x_grads[i] else 0
            ratio = w_in_avg / avg_abs_grad if avg_abs_grad > 0 else float('inf')

            print(f"  L{i}: b_beta |grad|={avg_abs_grad:.8f}, sign_consistent={100*consistent:.1f}%, "
                  f"W_in/b_beta ratio={ratio:.1f}x, W_beta_x |grad|={w_beta_avg:.8f}")

            # 按 N 组分析 sign consistency
            sc_grouped = sign_consistency.reshape(D, N)
            for n in [0, 3, 7]:  # low, mid, high β
                sc_n = sc_grouped[:, n]
                consist_n = ((sc_n > 0.7) | (sc_n < 0.3)).float().mean().item()
                print(f"    N={n} (β≈{[0.80,0.83,0.85,0.88,0.91,0.94,0.96,0.99][n]:.2f}): "
                      f"sign_consistent={100*consist_n:.1f}%")

        results[variant] = {
            i: {
                'grads': grad_history[i],
                'avg_abs': torch.stack([g.abs().mean() for g in grad_history[i]]).mean().item() if grad_history[i] else 0,
            }
            for i in range(num_layers)
        }

    # 对比总结
    print(f"\n{'='*60}")
    print("对比总结")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Original |grad|':>16} {'Voltage |grad|':>16} {'改善':>8}")
    print("-" * 50)
    for i in range(num_layers):
        orig = results['original'][i]['avg_abs']
        volt = results['voltage_output'][i]['avg_abs']
        ratio = volt / orig if orig > 0 else float('inf')
        print(f"L{i:<6} {orig:>16.8f} {volt:>16.8f} {ratio:>8.1f}x")

    # 一致性对比
    print(f"\n{'Layer':<8} {'Orig consistent':>16} {'Voltage consistent':>18}")
    print("-" * 50)
    for i in [0, num_layers - 1]:
        for variant in variants:
            gs = results[variant][i]['grads']
            if gs:
                grad_stack = torch.stack(gs)
                signs = (grad_stack > 0).float()
                sc = signs.mean(dim=0)
                consistent = ((sc > 0.7) | (sc < 0.3)).float().mean().item()
            else:
                consistent = 0
            if variant == 'original':
                orig_c = consistent
            else:
                volt_c = consistent
        print(f"L{i:<6} {100*orig_c:>15.1f}% {100*volt_c:>17.1f}%")


if __name__ == '__main__':
    run_comparison()
