"""
对比实验：移除 SNNBlock 的 output_neuron 后，b_beta 梯度信号是否恢复。

双 surrogate 瓶颈假说：
  当前: input → hidden_neuron(surrogate#1) → W_out → output_neuron(surrogate#2) → out
  修改: input → hidden_neuron(surrogate#1) → W_out → gate → out (连续输出，无 surrogate#2)

如果假说正确，移除 surrogate#2 后 b_beta 梯度应该显著增大且方向一致。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


class PatchedSNNBlock:
    """
    猴子补丁：替换 SNNBlock.forward_parallel，跳过 output_neuron。
    只影响 forward_parallel 中的 Phase 5（输出神经元），改为直接输出连续值。
    """
    @staticmethod
    def patch_forward_parallel(block, spike_in_seq):
        """替换后的 forward_parallel：无 output_neuron"""
        from atomic_ops.parallel_scan import plif_parallel_forward
        from atomic_ops.snn_block import _fused_modulation

        TK, batch, D = spike_in_seq.shape
        DN = block.D * block.N

        # Phase 1: 批量投影
        flat = spike_in_seq.reshape(TK * batch, D)

        I_all = F.linear(flat, block.W_in.weight).reshape(TK, batch, DN)
        raw_beta = F.linear(flat, block.W_beta_x.weight).reshape(TK, batch, DN)
        raw_alpha = F.linear(flat, block.W_alpha_x.weight).reshape(TK, batch, DN)
        raw_th = F.linear(flat, block.W_th_x.weight).reshape(TK, batch, DN)
        gate_all = torch.sigmoid(
            F.linear(flat, block.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, block.W_skip.weight).reshape(TK, batch, D)

        # Phase 1b: 融合激活
        beta_all, u_hidden, v_th_all = _fused_modulation(
            raw_beta, block.b_beta, raw_alpha, block.b_alpha,
            raw_th, block.b_th, block.v_th_min, I_all,
        )

        # Phase 2: Hidden neuron parallel scan (保留 — 这是唯一的 surrogate)
        v_init_hidden = block.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch, DN, device=flat.device, dtype=flat.dtype)

        s_hidden, V_post_hidden, _ = plif_parallel_forward(
            beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
            surrogate_function=block.hidden_neuron.surrogate_function,
        )
        block.hidden_neuron.v = V_post_hidden[-1].detach()

        # Phase 4: 输出投影
        s_flat = s_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(s_flat, block.W_out.weight).reshape(TK, batch, D)
        I_total_all = I_out_all * gate_all + I_skip_all

        # Phase 5: 【移除 output_neuron】直接返回连续值
        # 原代码: spike_out = output_neuron(I_total_all)  ← 第二道 surrogate 瓶颈
        # 修改后: 直接返回连续值
        return I_total_all  # (TK, batch, D) — 连续值，非 spike


def run_comparison(D=256, N=8, num_layers=4, D_ff=512,
                   seq_len=32, batch_size=2, num_steps=10):
    """对比有/无 output_neuron 时 b_beta 的梯度"""
    device = 'cuda'

    results = {}

    for variant in ['original', 'no_output_neuron']:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")
        print(f"{'='*60}")

        model = SNNLanguageModel(
            vocab_size=6144, D=D, N=N, num_layers=num_layers, D_ff=D_ff,
        ).to(device)

        # 猴子补丁
        if variant == 'no_output_neuron':
            for layer_module in model.layers:
                block = layer_module.snn_block
                original_fwd = block.forward_parallel
                block.forward_parallel = lambda x, b=block: PatchedSNNBlock.patch_forward_parallel(b, x)

        # 多步梯度收集
        grad_history = {i: [] for i in range(num_layers)}

        for step in range(num_steps):
            model.train()
            model.zero_grad()

            token_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)
            target_ids = torch.randint(1, model.vocab_size, (batch_size, seq_len), device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(token_ids, target_ids)
                loss = out.last_loss.mean()

            loss.backward()

            # 收集原始梯度（补偿前）
            for i, layer_module in enumerate(model.layers):
                g = layer_module.snn_block.b_beta.grad
                if g is not None:
                    grad_history[i].append(g.clone())

            if step == 0:
                # 报告梯度幅值
                for i in [0, num_layers - 1]:
                    g = model.layers[i].snn_block.b_beta.grad
                    w_g = model.layers[i].snn_block.W_in.weight.grad
                    if g is not None and w_g is not None:
                        print(f"  L{i} b_beta |grad|={g.abs().mean().item():.8f}  "
                              f"W_in |grad|={w_g.abs().mean().item():.8f}  "
                              f"ratio={w_g.abs().mean().item()/max(g.abs().mean().item(), 1e-12):.1f}x")

        # 分析梯度一致性
        for i in [0, num_layers - 1]:
            grads = grad_history[i]
            if len(grads) < 2:
                continue
            grad_stack = torch.stack(grads, dim=0)
            signs = (grad_stack > 0).float()
            sign_consistency = signs.mean(dim=0)

            mostly_positive = (sign_consistency > 0.7).sum().item()
            mostly_negative = (sign_consistency < 0.3).sum().item()
            random_sign = ((sign_consistency >= 0.3) & (sign_consistency <= 0.7)).sum().item()
            total = sign_consistency.numel()

            print(f"  L{i} sign_consistency: {sign_consistency.mean().item():.4f}")
            print(f"    一致 (>70% or <30%): {mostly_positive + mostly_negative} ({100*(mostly_positive+mostly_negative)/total:.1f}%)")
            print(f"    随机 (30-70%): {random_sign} ({100*random_sign/total:.1f}%)")

        results[variant] = {
            i: {
                'grad_abs_mean': grad_history[i][-1].abs().mean().item() if grad_history[i] else 0,
                'grad_stack': torch.stack(grad_history[i]) if grad_history[i] else None,
            }
            for i in range(num_layers)
        }

    # ====== 对比总结 ======
    print(f"\n{'='*60}")
    print("对比总结: Original vs No-Output-Neuron")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Orig |grad|':>14} {'NoOut |grad|':>14} {'改善倍数':>10}")
    print("-" * 50)
    for i in range(num_layers):
        orig = results['original'][i]['grad_abs_mean']
        noout = results['no_output_neuron'][i]['grad_abs_mean']
        ratio = noout / orig if orig > 0 else float('inf')
        print(f"L{i:<6} {orig:>14.8f} {noout:>14.8f} {ratio:>10.1f}x")

    # 一致性对比
    print(f"\n{'Layer':<8} {'Orig consistency':>18} {'NoOut consistency':>18}")
    print("-" * 50)
    for i in [0, num_layers - 1]:
        for variant in ['original', 'no_output_neuron']:
            gs = results[variant][i]['grad_stack']
            if gs is not None:
                signs = (gs > 0).float()
                sc = signs.mean(dim=0)
                consistent = ((sc > 0.7) | (sc < 0.3)).float().mean().item()
            else:
                consistent = 0
            if variant == 'original':
                orig_c = consistent
            else:
                noout_c = consistent
        print(f"L{i:<6} {100*orig_c:>17.1f}% {100*noout_c:>17.1f}%")


if __name__ == '__main__':
    run_comparison()
