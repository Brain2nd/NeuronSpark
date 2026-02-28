#!/usr/bin/env python
"""
梯度诊断脚本 v1.0 — 详细分析 checkpoint 每层每个参数的真实梯度情况（无补偿）

分析内容:
  Section 1: 模型概况 + checkpoint 元信息
  Section 2: β/ω/r 分布 + 理论时序梯度衰减（静态，不需 forward）
  Section 3: Forward+Backward 原始梯度统计（每参数详情）
  Section 4: 逐层梯度 norm 汇总
  Section 5: 参数组对比（b_beta vs b_alpha vs W_in vs ... 平均梯度规模）
  Section 6: β vs |grad(b_beta)| 相关性（高 β 是否梯度更小）
  Section 7: 补偿无效性定量证明
  Section 8: 时序梯度流实测（每层独立实验，实测首尾 token 梯度比）
  Section 9: FFN / MoE 神经元梯度分析
  Section 10: 诊断总结

用法:
    # 基本用法（合成随机 token 输入，适合快速分析）
    python analyze_gradients.py --checkpoint checkpoints/ckpt_step10000.pth

    # 指定序列长度和 batch（影响 TK 和显存占用）
    python analyze_gradients.py --checkpoint checkpoints/ckpt_step10000.pth \
        --seq_len 128 --batch_size 2

    # 仅静态分析（不跑 forward+backward，零显存/零数据）
    python analyze_gradients.py --checkpoint checkpoints/ckpt_step10000.pth --static_only

    # CPU 模式（无 GPU 时）
    python analyze_gradients.py --checkpoint checkpoints/ckpt_step10000.pth --device cpu

    # 保存 JSON 结果供后续绘图
    python analyze_gradients.py --checkpoint checkpoints/ckpt_step10000.pth \
        --output_json gradient_report.json
"""

import os
import sys
import argparse
import json
import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ======================================================================
# Utilities
# ======================================================================

def P(msg=''):
    print(msg, flush=True)

def section(title):
    P(f"\n{'='*90}")
    P(f"  {title}")
    P(f"{'='*90}")

def subsec(title):
    P(f"\n  --- {title} ---")

def fmt(x, w=12):
    if x == 0:
        return f"{'0':>{w}}"
    if abs(x) < 1e-30:
        return f"{'~0':>{w}}"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return f"{x:>{w}.3e}"
    return f"{x:>{w}.5f}"

def quantiles_str(t):
    """返回 min/q25/q50/q75/max 字符串。"""
    t = t.float().flatten()
    if t.numel() == 0:
        return "empty"
    qs = [t.min().item(), t.quantile(0.25).item(), t.quantile(0.5).item(),
          t.quantile(0.75).item(), t.max().item()]
    return (f"min={qs[0]:.4f}  q25={qs[1]:.4f}  q50={qs[2]:.4f}  "
            f"q75={qs[3]:.4f}  max={qs[4]:.4f}")

def dead_pct(t, threshold=0.01):
    return f"{(t < threshold).float().mean().item()*100:.1f}%"

def param_count_str(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Checkpoint 梯度诊断')
    p.add_argument('--checkpoint', type=str, required=True, help='Checkpoint .pth 路径')
    p.add_argument('--seq_len', type=int, default=128,
                   help='分析用序列长度（默认 128，TK=2048；生产 512，TK=8192）')
    p.add_argument('--batch_size', type=int, default=2, help='分析用 batch size')
    p.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
    p.add_argument('--static_only', action='store_true',
                   help='仅静态分析（β/ω/r 分布），不跑 forward+backward')
    p.add_argument('--temporal_seq_len', type=int, default=32,
                   help='时序梯度实测的序列长度（默认 32, TK=512，单层不用 checkpoint）')
    p.add_argument('--output_json', type=str, default='gradient_analysis.json',
                   help='输出 JSON 路径')
    return p.parse_args()


# ======================================================================
# Section 1: Load model
# ======================================================================

def load_model(args):
    section("Section 1: 模型概况")

    assert os.path.isfile(args.checkpoint), f"Checkpoint 不存在: {args.checkpoint}"
    P(f"  加载 checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt['model_config']

    from model import SNNLanguageModel
    model = SNNLanguageModel(**config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    model.train()

    P(f"  Step       : {ckpt.get('step', '?')}")
    P(f"  Epoch      : {ckpt.get('epoch', '?')}")
    P(f"  Best loss  : {ckpt.get('best_loss', '?')}")
    P(f"  Tokens seen: {ckpt.get('tokens_seen', '?')}")
    P()
    P(f"  D={config['D']}, N={config['N']}, K={config['K']}, "
      f"layers={config['num_layers']}, D_ff={config['D_ff']}")
    P(f"  use_moe={config.get('use_moe', False)}, "
      f"num_experts={config.get('num_experts', 'N/A')}, "
      f"top_k={config.get('top_k', 'N/A')}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    P(f"  Total params   : {total:>14,} ({param_count_str(total)})")
    P(f"  Trainable      : {trainable:>14,} ({param_count_str(trainable)})")

    return model, config, ckpt


# ======================================================================
# Section 2: Static β/ω/r analysis
# ======================================================================

def static_analysis(model, config):
    section("Section 2: β/ω/r 分布 + 理论时序梯度衰减")

    K = config['K']
    D = config['D']
    N = config['N']
    DN = D * N
    num_layers = config['num_layers']
    TK_prod = 512 * K  # 生产序列长度

    P(f"  K={K}, 生产 TK={TK_prod} (seq_len=512)")
    P(f"  注意: β/ω 为静态分量 sigmoid(b_beta)/softplus(b_omega)，实际值受输入调制")

    report = {'layers': []}
    k_vals = [K, K * 16, K * 64, TK_prod]
    k_labels = ['1 token', '16 tokens', '64 tokens', '512 tokens (生产)']

    for i, dec_layer in enumerate(model.layers):
        block = dec_layer.snn_block
        layer_data = {'layer': i}

        subsec(f"Layer {i:2d} — SNNBlock hidden neurons (DN={DN})")

        with torch.no_grad():
            beta_s = torch.sigmoid(block.b_beta)
            omega_s = F.softplus(block.b_omega)
            r_s = torch.sqrt(beta_s ** 2 + omega_s ** 2).clamp(max=1.0)
            vth_s = block.v_th_min + torch.abs(block.b_th)

        for tag, t in [('β', beta_s), ('ω', omega_s), ('r', r_s), ('v_th', vth_s)]:
            P(f"    {tag:4s}: mean={t.mean().item():.4f}  std={t.std().item():.4f}  "
              f"{quantiles_str(t)}")

        layer_data['hidden'] = {
            'beta_mean': beta_s.mean().item(), 'beta_std': beta_s.std().item(),
            'omega_mean': omega_s.mean().item(), 'r_mean': r_s.mean().item(),
        }

        P()
        P(f"    {'时序衰减 r^steps':40s}  {'mean':>10s}  {'median':>10s}  "
          f"{'max':>10s}  {'<0.01 比例':>10s}  {'<1e-4 比例':>10s}")
        P(f"    {'-'*100}")

        for kv, kl in zip(k_vals, k_labels):
            decay = r_s ** kv
            P(f"    r^{kv:<5d} ({kl:22s}): {decay.mean().item():10.3e}  "
              f"{decay.median().item():10.3e}  {decay.max().item():10.3e}  "
              f"{dead_pct(decay, 0.01):>10s}  {dead_pct(decay, 1e-4):>10s}")
            layer_data[f'r^{kv}_mean'] = decay.mean().item()

        # ---- 输入神经元 ----
        for nname in ['input_neuron1', 'input_neuron2']:
            neuron = getattr(dec_layer, nname)
            with torch.no_grad():
                beta_n = torch.sigmoid(neuron.w)
            P(f"\n    {nname} (D={D}):  β mean={beta_n.mean().item():.4f}  "
              f"{quantiles_str(beta_n)}")
            for kv, kl in zip(k_vals, k_labels):
                decay = beta_n ** kv
                P(f"      β^{kv:<5d} ({kl:22s}): mean={decay.mean().item():.3e}  "
                  f"<0.01={dead_pct(decay, 0.01)}")

        # ---- Block 输出神经元 ----
        out_n = block.output_neuron
        with torch.no_grad():
            beta_o = torch.sigmoid(out_n.w)
        P(f"\n    block output_neuron (D={D}):  β mean={beta_o.mean().item():.4f}  "
          f"{quantiles_str(beta_o)}")

        # ---- FFN 神经元 ----
        ffn = dec_layer.snn_ffn
        if hasattr(ffn, 'shared_expert'):
            # MoE: 分析 shared expert
            shared = ffn.shared_expert
            _analyze_ffn_neurons_static(shared, 'MoE shared', D, K, TK_prod)
            # 分析 routed experts (堆叠参数)
            with torch.no_grad():
                egu_beta = torch.sigmoid(ffn.expert_gu_w)  # (E, 2*D_ff)
                eout_beta = torch.sigmoid(ffn.expert_out_w)  # (E, D)
            P(f"\n    MoE routed experts: gu_beta mean={egu_beta.mean().item():.4f}  "
              f"out_beta mean={eout_beta.mean().item():.4f}")
        else:
            _analyze_ffn_neurons_static(ffn, 'FFN', D, K, TK_prod)

        report['layers'].append(layer_data)

    # ---- 模型输出神经元 ----
    subsec(f"Model output_neuron (D={D})")
    with torch.no_grad():
        beta_mo = torch.sigmoid(model.output_neuron.w)
    P(f"    β mean={beta_mo.mean().item():.4f}  {quantiles_str(beta_mo)}")

    return report


def _analyze_ffn_neurons_static(ffn, label, D, K, TK_prod):
    """分析 SNNFFN 内神经元的 β 分布。"""
    D_ff = ffn.D_ff
    for nname, neuron, dim in [
        ('gate_neuron', ffn.gate_neuron, D_ff),
        ('up_neuron', ffn.up_neuron, D_ff),
        ('output_neuron', ffn.output_neuron, D),
    ]:
        with torch.no_grad():
            beta_n = torch.sigmoid(neuron.w)
        P(f"\n    {label} {nname} (dim={dim}):  β mean={beta_n.mean().item():.4f}  "
          f"{quantiles_str(beta_n)}")


# ======================================================================
# Section 3: Forward + Backward
# ======================================================================

def run_forward_backward(model, args, config):
    section("Section 3: Forward+Backward 梯度采集（原始梯度，无补偿）")

    K = config['K']
    vocab_size = config['vocab_size']
    seq_len = args.seq_len
    batch = args.batch_size
    TK = seq_len * K

    P(f"  seq_len={seq_len}, batch={batch}, TK={TK}, vocab={vocab_size}")
    P(f"  使用合成随机 token 输入")

    # 合成数据
    torch.manual_seed(42)
    token_ids = torch.randint(1, vocab_size, (batch, seq_len + 1), device=args.device)
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]

    # Forward + backward
    model.zero_grad()
    use_cuda_amp = args.device == 'cuda'

    if use_cuda_amp:
        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids, target_ids)
            loss = output.last_loss.mean()
        # backward in float32 (AMP standard)
        loss.backward()
    else:
        output = model(input_ids, target_ids)
        loss = output.last_loss.mean()
        loss.backward()

    # 不调用 model.compensate_modulation_gradients()!
    P(f"  Loss = {loss.item():.4f}")
    P(f"  注意: 以下梯度为原始 backward 梯度，未经 compensate_modulation_gradients 补偿")

    return loss.item()


# ======================================================================
# Section 4: Per-parameter gradient statistics
# ======================================================================

def gradient_analysis(model, config):
    section("Section 3 续: 逐参数梯度统计")

    header = (f"  {'Parameter':<55s} {'numel':>8s}  {'|g|_mean':>10s}  {'|g|_max':>10s}  "
              f"{'g_norm':>10s}  {'|w|_norm':>10s}  {'g/w ratio':>10s}")
    P(header)
    P(f"  {'-'*len(header)}")

    all_stats = OrderedDict()

    for name, param in model.named_parameters():
        if param.grad is None:
            all_stats[name] = {'numel': param.numel(), 'grad': None}
            P(f"  {name:<55s} {param.numel():>8d}  {'NO GRAD':>10s}")
            continue

        g = param.grad.detach().float()
        w = param.data.detach().float()

        g_mean = g.abs().mean().item()
        g_max = g.abs().max().item()
        g_norm = g.norm().item()
        w_norm = w.norm().item()
        ratio = g_norm / w_norm if w_norm > 1e-12 else float('inf')

        all_stats[name] = {
            'numel': param.numel(),
            'grad_abs_mean': g_mean,
            'grad_abs_max': g_max,
            'grad_norm': g_norm,
            'weight_norm': w_norm,
            'grad_weight_ratio': ratio,
        }

        P(f"  {name:<55s} {param.numel():>8d}  {fmt(g_mean)}  {fmt(g_max)}  "
          f"{fmt(g_norm)}  {fmt(w_norm)}  {fmt(ratio)}")

    return all_stats


# ======================================================================
# Section 4: Per-layer gradient norm summary
# ======================================================================

def layer_gradient_profile(model, config, all_stats):
    section("Section 4: 逐层梯度 norm 汇总")

    num_layers = config['num_layers']

    # 按层分组
    P(f"  {'Layer':>5s}  {'block_total':>12s}  {'ffn_total':>12s}  "
      f"{'b_beta':>12s}  {'b_alpha':>12s}  {'b_omega':>12s}  {'b_th':>12s}  "
      f"{'W_in':>12s}  {'W_out':>12s}")
    P(f"  {'-'*115}")

    layer_norms = []

    for i in range(num_layers):
        prefix = f'layers.{i}.'
        block_prefix = f'layers.{i}.snn_block.'
        ffn_prefix = f'layers.{i}.snn_ffn.'

        block_total = 0.0
        ffn_total = 0.0
        named_norms = {}

        for name, stats in all_stats.items():
            if not name.startswith(prefix):
                continue
            if stats['grad'] is None:
                continue
            gn = stats['grad_norm']

            if name.startswith(block_prefix):
                block_total += gn ** 2
            elif name.startswith(ffn_prefix):
                ffn_total += gn ** 2

            # 特定参数
            short = name[len(prefix):]
            for key in ['snn_block.b_beta', 'snn_block.b_alpha', 'snn_block.b_omega',
                        'snn_block.b_th', 'snn_block.W_in.weight', 'snn_block.W_out.weight']:
                if short == key:
                    named_norms[key.split('.')[-1] if '.' in key[10:] else key[10:]] = gn

        block_total = math.sqrt(block_total)
        ffn_total = math.sqrt(ffn_total)

        P(f"  {i:5d}  {fmt(block_total)}  {fmt(ffn_total)}  "
          f"{fmt(named_norms.get('b_beta', 0))}  {fmt(named_norms.get('b_alpha', 0))}  "
          f"{fmt(named_norms.get('b_omega', 0))}  {fmt(named_norms.get('b_th', 0))}  "
          f"{fmt(named_norms.get('weight', 0))}  "  # W_in.weight
          f"{fmt(named_norms.get('weight', 0))}")  # placeholder

        layer_norms.append({
            'layer': i, 'block_total': block_total, 'ffn_total': ffn_total,
            **{k: v for k, v in named_norms.items()},
        })

    return layer_norms


# ======================================================================
# Section 5: Parameter group comparison
# ======================================================================

def param_group_comparison(model, all_stats):
    section("Section 5: 参数组对比（平均梯度规模）")

    groups = defaultdict(list)

    for name, stats in all_stats.items():
        if stats['grad'] is None:
            continue
        gn = stats['grad_norm']
        n = stats['numel']

        # 分类
        if 'b_beta' in name:
            groups['b_beta'].append((gn, n))
        elif 'b_alpha' in name:
            groups['b_alpha'].append((gn, n))
        elif 'b_omega' in name:
            groups['b_omega'].append((gn, n))
        elif 'b_th' in name:
            groups['b_th'].append((gn, n))
        elif 'W_in.weight' in name:
            groups['W_in'].append((gn, n))
        elif 'W_beta_x.weight' in name:
            groups['W_beta_x'].append((gn, n))
        elif 'W_alpha_x.weight' in name:
            groups['W_alpha_x'].append((gn, n))
        elif 'W_th_x.weight' in name:
            groups['W_th_x'].append((gn, n))
        elif 'W_omega_x.weight' in name:
            groups['W_omega_x'].append((gn, n))
        elif 'W_gate.weight' in name:
            groups['W_gate'].append((gn, n))
        elif 'W_skip.weight' in name:
            groups['W_skip'].append((gn, n))
        elif 'W_out.weight' in name:
            groups['W_out'].append((gn, n))
        elif 'input_neuron' in name:
            groups['input_neurons'].append((gn, n))
        elif 'output_neuron' in name and 'snn_block' in name:
            groups['block_out_neuron'].append((gn, n))
        elif 'gate_proj' in name or 'up_proj' in name:
            groups['ffn_gate_up'].append((gn, n))
        elif 'down_proj' in name:
            groups['ffn_down'].append((gn, n))
        elif 'skip_proj' in name:
            groups['ffn_skip'].append((gn, n))
        elif 'ffn' in name and 'neuron' in name:
            groups['ffn_neurons'].append((gn, n))
        elif 'expert' in name and ('W_gus' in name or 'W_down' in name):
            groups['moe_expert_proj'].append((gn, n))
        elif 'expert' in name:
            groups['moe_expert_neuron'].append((gn, n))
        elif 'router' in name:
            groups['moe_router'].append((gn, n))
        elif 'embed_tokens' in name:
            groups['embedding'].append((gn, n))
        elif 'out_proj' in name or 'block_out_proj' in name or 'ffn_out_proj' in name:
            groups['residual_proj'].append((gn, n))
        elif 'norm' in name:
            groups['norms'].append((gn, n))
        elif 'decode_proj' in name:
            groups['decode_proj'].append((gn, n))
        else:
            groups['other'].append((gn, n))

    P(f"  {'Group':<25s}  {'count':>5s}  {'total_params':>12s}  "
      f"{'mean_gnorm':>12s}  {'max_gnorm':>12s}  {'min_gnorm':>12s}")
    P(f"  {'-'*85}")

    for gname in ['b_beta', 'b_alpha', 'b_omega', 'b_th',
                   'W_in', 'W_beta_x', 'W_alpha_x', 'W_th_x', 'W_omega_x',
                   'W_gate', 'W_skip', 'W_out',
                   'input_neurons', 'block_out_neuron',
                   'ffn_gate_up', 'ffn_down', 'ffn_skip', 'ffn_neurons',
                   'moe_expert_proj', 'moe_expert_neuron', 'moe_router',
                   'residual_proj', 'norms', 'embedding', 'decode_proj', 'other']:
        if gname not in groups or not groups[gname]:
            continue
        items = groups[gname]
        gnorms = [x[0] for x in items]
        total_n = sum(x[1] for x in items)
        mean_g = sum(gnorms) / len(gnorms)
        max_g = max(gnorms)
        min_g = min(gnorms)

        P(f"  {gname:<25s}  {len(items):5d}  {param_count_str(total_n):>12s}  "
          f"{fmt(mean_g)}  {fmt(max_g)}  {fmt(min_g)}")


# ======================================================================
# Section 6: β vs gradient correlation
# ======================================================================

def beta_gradient_correlation(model, config):
    section("Section 6: β vs |grad(b_beta)| 相关性分析")

    P("  对每层的 b_beta 参数，按 β 值分桶，统计各桶的平均梯度幅度。")
    P("  若高 β 区梯度显著更小，证实 sigmoid 饱和导致梯度消失。")
    P()

    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85),
            (0.85, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.99), (0.99, 1.0)]

    P(f"  {'β range':<12s}", end='')
    for lo, hi in bins:
        P(f"  [{lo:.2f},{hi:.2f})", end='')
    P()
    P(f"  {'-'*120}")

    all_betas = []
    all_grads = []

    for i, dec_layer in enumerate(model.layers):
        block = dec_layer.snn_block
        if block.b_beta.grad is None:
            continue

        with torch.no_grad():
            beta = torch.sigmoid(block.b_beta).cpu()
            grad_abs = block.b_beta.grad.abs().float().cpu()

        all_betas.append(beta)
        all_grads.append(grad_abs)

        row = f"  Layer {i:2d}   "
        for lo, hi in bins:
            mask = (beta >= lo) & (beta < hi)
            if mask.sum() == 0:
                row += f"  {'---':>10s}"
            else:
                val = grad_abs[mask].mean().item()
                row += f"  {fmt(val, 10)}"
        P(row)

    # 全局聚合
    if all_betas:
        all_beta = torch.cat(all_betas)
        all_grad = torch.cat(all_grads)

        P()
        row = f"  {'ALL layers':<11s}"
        for lo, hi in bins:
            mask = (all_beta >= lo) & (all_beta < hi)
            if mask.sum() == 0:
                row += f"  {'---':>10s}"
            else:
                val = all_grad[mask].mean().item()
                cnt = mask.sum().item()
                row += f"  {fmt(val, 10)}"
        P(row)

        row = f"  {'count':<11s}"
        for lo, hi in bins:
            mask = (all_beta >= lo) & (all_beta < hi)
            row += f"  {mask.sum().item():>10d}"
        P(row)

        # Pearson correlation
        corr = torch.corrcoef(torch.stack([all_beta, all_grad]))[0, 1].item()
        P(f"\n  Pearson 相关系数 (β, |grad|): {corr:.4f}")
        P(f"  解读: 负相关 → β 越高梯度越小（sigmoid 饱和效应）")
        P(f"         正相关 → β 越高梯度越大（长程依赖信号主导）")
        P(f"         ~0     → 无明显相关")


# ======================================================================
# Section 7: Compensation futility analysis
# ======================================================================

def compensation_futility(model, config):
    section("Section 7: 补偿无效性定量证明")

    K = config['K']
    TK_prod = 512 * K

    P("  对比: 补偿倍率 1/[β(1-β)] vs 时序衰减 r^TK")
    P("  补偿只消除 sigmoid 参数化的链式法则因子 β(1-β)。")
    P("  但时序梯度衰减 r^TK 是数十个数量级更大的问题。")
    P()

    P(f"  {'Layer':>5s}  {'β_mean':>8s}  {'β_max':>8s}  "
      f"{'1/β(1-β)_mean':>14s}  {'1/β(1-β)_max':>14s}  "
      f"{'r_mean':>8s}  {'r^{K}_mean':>12s}  {'r^{TK}_mean':>12s}  "
      f"{'补偿后仍':>12s}")
    P(f"  {'-'*115}")

    for i, dec_layer in enumerate(model.layers):
        block = dec_layer.snn_block
        with torch.no_grad():
            beta = torch.sigmoid(block.b_beta)
            omega = F.softplus(block.b_omega)
            r = torch.sqrt(beta**2 + omega**2).clamp(max=1.0)

            # 补偿倍率
            deriv = (beta * (1.0 - beta)).clamp(min=1e-6)
            comp = 1.0 / deriv

            # 时序衰减
            r_K = r ** K
            r_TK = r ** TK_prod

            # "补偿后仍" = comp_mean × r_TK_mean（补偿放大后乘以时序衰减）
            effective = comp.mean().item() * r_TK.mean().item()

        P(f"  {i:5d}  {beta.mean().item():8.4f}  {beta.max().item():8.4f}  "
          f"{comp.mean().item():14.1f}  {comp.max().item():14.1f}  "
          f"{r.mean().item():8.4f}  {r_K.mean().item():12.3e}  "
          f"{r_TK.mean().item():12.3e}  {effective:12.3e}")

    P()
    P("  解读:")
    P("  · 补偿倍率 ~100x（β=0.99 时 1/[0.99×0.01]=101）")
    P("  · 时序衰减 r^8192 ~ 10^{-4} 到 10^{-36}")
    P("  · 补偿后: 100 × 10^{-36} = 10^{-34}，仍然是零")
    P("  · 结论: 补偿解决的是 O(100) 量级的问题，真正的问题是 O(10^{-36}) 的时序衰减")


# ======================================================================
# Section 8: Temporal gradient flow measurement
# ======================================================================

def temporal_gradient_test(model, config, args):
    section("Section 8: 时序梯度流实测（每层独立实验）")

    K = config['K']
    D = config['D']
    seq_len = args.temporal_seq_len
    TK = seq_len * K
    batch = 1  # 节省显存

    P(f"  方法: 对每层 — 构造 h(TK={TK}, B={batch}, D={D}), requires_grad=True")
    P(f"         运行 layer.forward_parallel(h)")
    P(f"         loss = h_out[-{K}:].sum()（仅最后 1 个 token 的 loss）")
    P(f"         backward → 测量 ∂loss/∂h 在每个 token 位置的 norm")
    P(f"         梯度比 = grad_norm[第1个token] / grad_norm[最后1个token]")
    P()
    P(f"  注意: 残差连接使 ∂h_out[t]/∂h[t] 包含恒等分量，")
    P(f"         因此 loss 所在 token 的梯度 ≈ 1。跨 token 梯度衰减来自 SNN 时序动力学。")
    P()

    from spikingjelly.activation_based import functional

    results = []

    P(f"  {'Layer':>5s}  {'grad[-1]':>10s}  {'grad[-16]':>10s}  {'grad[-64]':>10s}  "
      f"{'grad[0]':>10s}  {'ratio 0/-1':>12s}  {'ratio -64/-1':>12s}")
    P(f"  {'-'*85}")

    for i, dec_layer in enumerate(model.layers):
        try:
            # 重置神经元
            functional.reset_net(dec_layer)

            # 构造输入
            torch.manual_seed(42 + i)
            h = torch.randn(TK, batch, D, device=args.device, dtype=torch.float32,
                            requires_grad=True)

            # forward (不用 gradient checkpoint, 因为只跑单层)
            with torch.no_grad():
                pass  # 确保上下文干净
            h_out = dec_layer.forward_parallel(h)

            # loss 仅来自最后 1 个 token
            loss = h_out[-K:].sum()
            loss.backward()

            # 收集每个 token 位置的梯度 norm
            grad = h.grad.detach()  # (TK, B, D)
            token_norms = []
            for t in range(seq_len):
                gn = grad[t * K:(t + 1) * K].norm().item()
                token_norms.append(gn)

            last = token_norms[-1] if token_norms[-1] > 0 else 1e-30
            first = token_norms[0]
            t16 = token_norms[-min(16, seq_len)] if seq_len >= 16 else token_norms[0]
            t64 = token_norms[-min(64, seq_len)] if seq_len >= 64 else token_norms[0]

            ratio_first = first / last
            ratio_64 = t64 / last

            P(f"  {i:5d}  {fmt(last)}  {fmt(t16)}  {fmt(t64)}  "
              f"{fmt(first)}  {ratio_first:12.3e}  {ratio_64:12.3e}")

            results.append({
                'layer': i,
                'token_norms': token_norms,
                'ratio_first_last': ratio_first,
            })

            # 清理
            del h, h_out, loss, grad
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            P(f"  {i:5d}  ERROR: {e}")
            results.append({'layer': i, 'error': str(e)})

    # 详细打印每层每 token 的梯度衰减曲线（选 layer 0 和 layer -1）
    for idx in [0, len(results) - 1]:
        if idx >= len(results) or 'error' in results[idx]:
            continue
        data = results[idx]
        layer_i = data['layer']
        norms = data['token_norms']
        subsec(f"Layer {layer_i} 逐 token 梯度 norm 详情 (共 {len(norms)} tokens)")
        ref = norms[-1] if norms[-1] > 0 else 1e-30
        P(f"    {'token':>6s}  {'steps':>12s}  {'grad_norm':>12s}  {'relative':>12s}")
        P(f"    {'-'*50}")
        # 打印首、尾各 8 个 + 中间采样
        indices = list(range(min(8, len(norms))))
        if len(norms) > 16:
            step = max(1, (len(norms) - 16) // 4)
            indices += list(range(8, len(norms) - 8, step))
        indices += list(range(max(0, len(norms) - 8), len(norms)))
        indices = sorted(set(indices))

        for t in indices:
            P(f"    {t:6d}  [{t*K:5d}-{(t+1)*K-1:5d}]  {fmt(norms[t])}  "
              f"{norms[t]/ref:12.3e}")

    return results


# ======================================================================
# Section 9: FFN/MoE neuron gradient details
# ======================================================================

def ffn_neuron_gradient_analysis(model, config):
    section("Section 9: FFN / MoE 神经元梯度详情")

    P("  分析每层 FFN（或 MoE shared expert）中 gate/up/output 神经元的梯度。")
    P()

    P(f"  {'Layer':>5s}  {'gate.w':>10s}  {'gate.vth':>10s}  "
      f"{'up.w':>10s}  {'up.vth':>10s}  "
      f"{'out.w':>10s}  {'out.vth':>10s}")
    P(f"  {'-'*75}")

    for i, dec_layer in enumerate(model.layers):
        ffn = dec_layer.snn_ffn
        if hasattr(ffn, 'shared_expert'):
            ffn_actual = ffn.shared_expert
        else:
            ffn_actual = ffn

        vals = []
        for nname in ['gate_neuron', 'up_neuron', 'output_neuron']:
            neuron = getattr(ffn_actual, nname)
            for pname in ['w', 'v_th']:
                p = getattr(neuron, pname)
                if p.grad is not None:
                    vals.append(p.grad.abs().mean().item())
                else:
                    vals.append(0.0)

        P(f"  {i:5d}  " + "  ".join(fmt(v) for v in vals))

    # MoE expert 参数梯度
    if hasattr(model.layers[0].snn_ffn, 'shared_expert'):
        subsec("MoE 路由 expert 堆叠参数梯度")
        P(f"  {'Layer':>5s}  {'W_gus':>12s}  {'W_down':>12s}  "
          f"{'gu_w':>12s}  {'gu_vth':>12s}  {'out_w':>12s}  {'out_vth':>12s}")
        P(f"  {'-'*85}")

        for i, dec_layer in enumerate(model.layers):
            moe = dec_layer.snn_ffn
            vals = []
            for pname in ['expert_W_gus', 'expert_W_down',
                          'expert_gu_w', 'expert_gu_v_th',
                          'expert_out_w', 'expert_out_v_th']:
                p = getattr(moe, pname)
                if p.grad is not None:
                    vals.append(p.grad.norm().item())
                else:
                    vals.append(0.0)
            P(f"  {i:5d}  " + "  ".join(fmt(v) for v in vals))


# ======================================================================
# Section 10: Diagnosis summary
# ======================================================================

def diagnosis_summary(model, config, static_report, all_stats):
    section("Section 10: 诊断总结")

    K = config['K']
    TK_prod = 512 * K
    num_layers = config['num_layers']

    P("  ┌─────────────────────────────────────────────────────────────────┐")
    P("  │                       梯度诊断报告                              │")
    P("  └─────────────────────────────────────────────────────────────────┘")
    P()

    # 1. β 饱和统计
    total_high_beta = 0
    total_neurons = 0
    for dec_layer in model.layers:
        with torch.no_grad():
            beta = torch.sigmoid(dec_layer.snn_block.b_beta)
        total_high_beta += (beta > 0.95).sum().item()
        total_neurons += beta.numel()

    P(f"  [1] β 饱和: {total_high_beta}/{total_neurons} 隐神经元 β>0.95 "
      f"({total_high_beta/total_neurons*100:.1f}%)")
    P(f"      这些神经元的 sigmoid 导数 β(1-β) < 0.05×0.95 = 0.0475")
    P(f"      参数化梯度衰减 ≤ 21× (可通过补偿解决)")

    # 2. 时序衰减统计
    total_dead = 0
    total_hidden = 0
    for dec_layer in model.layers:
        with torch.no_grad():
            beta = torch.sigmoid(dec_layer.snn_block.b_beta)
            omega = F.softplus(dec_layer.snn_block.b_omega)
            r = torch.sqrt(beta**2 + omega**2).clamp(max=1.0)
            dead = (r ** TK_prod < 0.01).sum().item()
        total_dead += dead
        total_hidden += r.numel()

    P()
    P(f"  [2] 时序梯度衰减 (r^{TK_prod}): {total_dead}/{total_hidden} 隐神经元 r^TK<0.01 "
      f"({total_dead/total_hidden*100:.1f}%)")
    P(f"      这些神经元从第 1 个 token 到第 512 个 token 的梯度传递 < 1%")
    P(f"      此衰减 O(10^{{-4}} ~ 10^{{-36}}) 无法通过补偿修复")

    # 3. 梯度最小的参数
    P()
    P(f"  [3] 梯度最小的 10 个参数组:")
    sorted_stats = sorted(
        [(n, s) for n, s in all_stats.items() if s.get('grad_abs_mean') is not None],
        key=lambda x: x[1]['grad_abs_mean'],
    )
    for name, stats in sorted_stats[:10]:
        P(f"      {name:<55s}  |g|_mean={stats['grad_abs_mean']:.3e}  "
          f"g/w={stats['grad_weight_ratio']:.3e}")

    # 4. 梯度最大的参数
    P()
    P(f"  [4] 梯度最大的 10 个参数组:")
    for name, stats in sorted_stats[-10:]:
        P(f"      {name:<55s}  |g|_mean={stats['grad_abs_mean']:.3e}  "
          f"g/w={stats['grad_weight_ratio']:.3e}")

    # 5. 梯度动态范围
    all_means = [s['grad_abs_mean'] for _, s in sorted_stats if s['grad_abs_mean'] > 0]
    if all_means:
        ratio = max(all_means) / min(all_means)
        P()
        P(f"  [5] 梯度动态范围: max/min |g|_mean = {ratio:.1e}")
        P(f"      最大: {max(all_means):.3e}")
        P(f"      最小: {min(all_means):.3e}")
        if ratio > 1e4:
            P(f"      ⚠ 动态范围 > 10^4，不同参数组训练速度差异巨大")

    # 6. 根本矛盾
    P()
    P("  ┌─────────────────────────────────────────────────────────────────┐")
    P("  │  根本矛盾:                                                      │")
    P("  │    稳定性要求 r = √(β²+ω²) < 1 → 前向状态有界（不爆炸）         │")
    P("  │    梯度流要求 r ≈ 1             → 反向梯度不消失                 │")
    P("  │    乘法递推 V[t]=β·V[t-1]+u 下，两者矛盾                        │")
    P("  │                                                                 │")
    P("  │  补偿 1/β(1-β) 修的是 O(100) 量级的参数化问题                    │")
    P("  │  时序衰减 r^TK 是 O(10^{-4}~10^{-36}) 量级的根本问题            │")
    P("  └─────────────────────────────────────────────────────────────────┘")


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        P("  CUDA 不可用，自动切换到 CPU")
        args.device = 'cpu'

    # Section 1: Load
    model, config, ckpt = load_model(args)

    # Section 2: Static analysis
    static_report = static_analysis(model, config)

    if args.static_only:
        P("\n  --static_only 模式，跳过 forward+backward 分析。")
        return

    # Section 3: Forward + Backward
    loss_val = run_forward_backward(model, args, config)

    # Section 3 续: Per-parameter gradient stats
    all_stats = gradient_analysis(model, config)

    # Section 4: Layer gradient profile
    layer_norms = layer_gradient_profile(model, config, all_stats)

    # Section 5: Param group comparison
    param_group_comparison(model, all_stats)

    # Section 6: β vs gradient correlation
    beta_gradient_correlation(model, config)

    # Section 7: Compensation futility
    compensation_futility(model, config)

    # Section 8: Temporal gradient test (clear gradients first)
    model.zero_grad()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    temporal_results = temporal_gradient_test(model, config, args)

    # Rerun forward+backward for sections 9-10 (temporal test cleared grads)
    _ = run_forward_backward(model, args, config)
    all_stats = gradient_analysis(model, config)

    # Section 9: FFN neuron details
    ffn_neuron_gradient_analysis(model, config)

    # Section 10: Diagnosis
    diagnosis_summary(model, config, static_report, all_stats)

    # Save JSON
    json_data = {
        'checkpoint': args.checkpoint,
        'config': config,
        'loss': loss_val,
        'static_report': static_report,
        'temporal_results': [
            {k: v for k, v in r.items()}
            for r in temporal_results
        ],
    }
    with open(args.output_json, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    P(f"\n  JSON 结果已保存: {args.output_json}")


if __name__ == '__main__':
    main()
