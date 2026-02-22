"""
SNNAdamW 10 阶段诊断：100 步训练，逐步记录所有神经动力学参数的演变。

输出 JSON 文件包含每一步的完整快照：
  - 每层每通道的 β/ω/α/V_th 分布 (mean, std, min, max, per-N histogram)
  - RF 谱半径 r = √(β²+ω²)
  - 多样性指标 (N 维度 std)
  - Lyapunov 惩罚量 log(β)² + log(1-β)²
  - Loss / PPL

用法：
    conda activate SNN
    python exp/diagnose_snn_adamw.py
    # 输出: exp/snn_adamw_diagnosis.json
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SNNLanguageModel
from atomic_ops import SNNAdamW

torch.manual_seed(42)

D, N, K, num_layers, D_ff = 32, 4, 16, 2, 96
vocab_size = 64

model = SNNLanguageModel(
    vocab_size=vocab_size, D=D, N=N, K=K,
    num_layers=num_layers, D_ff=D_ff,
).cuda()

# ====== 构建参数组（完全对齐 train.py） ======
_pg = model.get_param_groups()
_wd_groups = {
    'embedding': 0.1, 'decode': 0.1,
    'W_in': 0.1, 'W_beta': 0.1, 'W_alpha': 0.1, 'W_th': 0.1, 'W_omega': 0.1,
    'W_gate': 0.1, 'W_skip': 0.1, 'W_out': 0.1,
    'residual_projs': 0.1,
    'ffn_gate_proj': 0.1, 'ffn_up_proj': 0.1, 'ffn_down_proj': 0.1, 'ffn_skip_proj': 0.1,
    'b_beta': 0.0, 'b_alpha': 0.0, 'b_th': 0.0, 'b_omega': 0.0,
    'input_neurons': 0.0, 'block_output_neuron': 0.0, 'ffn_neurons': 0.0, 'output_neuron': 0.0,
    'norm': 0.0, 'rms_norms': 0.0,
}
_dynamics_map = {'b_beta': 'b_beta', 'b_alpha': 'b_alpha', 'b_omega': 'b_omega', 'b_th': 'b_th'}
_neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th', 'b_omega',
                'block_output_neuron', 'ffn_neurons', 'output_neuron'}
neuron_lr_mult = 10.0
base_lr = 2e-4

param_groups = []
for key, params in _pg.items():
    if not params:
        continue
    lr_mult = neuron_lr_mult if key in _neuron_keys else 1.0
    param_groups.append({
        'params': params,
        'lr': base_lr * lr_mult,
        'lr_mult': lr_mult,
        'weight_decay': _wd_groups.get(key, 0.1),
        'dynamics': _dynamics_map.get(key),
        'N': N if key in ('b_beta', 'b_alpha', 'b_omega') else None,
    })

optimizer = SNNAdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), grad_clip=1.0)
scaler = torch.amp.GradScaler('cuda', enabled=False)

x = torch.randint(0, vocab_size, (2, 8)).cuda()
y = torch.randint(0, vocab_size, (2, 8)).cuda()


# ====== 诊断快照 ======
def to_list(t):
    return t.detach().cpu().tolist()


def detailed_snapshot(step_idx, loss_val):
    """采集完整的参数快照。"""
    record = {'step': step_idx, 'loss': loss_val, 'ppl': math.exp(min(loss_val, 20.0))}
    layers_data = []

    for li, layer_module in enumerate(model.layers):
        blk = layer_module.snn_block
        ld = {'layer': li}

        # ---- β ----
        beta_flat = torch.sigmoid(blk.b_beta.data)           # (D*N,)
        beta_2d = beta_flat.reshape(D, N)                     # (D, N)
        ld['beta'] = {
            'mean': beta_flat.mean().item(),
            'std': beta_flat.std().item(),
            'min': beta_flat.min().item(),
            'max': beta_flat.max().item(),
            'per_n_mean': to_list(beta_2d.mean(dim=0)),       # N 个均值
            'diversity_per_channel': beta_2d.std(dim=1).mean().item(),  # 通道内 N 维 std 均值
            'diversity_per_channel_min': beta_2d.std(dim=1).min().item(),
        }

        # ---- ω ----
        omega_flat = F.softplus(blk.b_omega.data)
        omega_2d = omega_flat.reshape(D, N)
        ld['omega'] = {
            'mean': omega_flat.mean().item(),
            'std': omega_flat.std().item(),
            'min': omega_flat.min().item(),
            'max': omega_flat.max().item(),
            'per_n_mean': to_list(omega_2d.mean(dim=0)),
            'diversity_per_channel': omega_2d.std(dim=1).mean().item(),
            'diversity_per_channel_min': omega_2d.std(dim=1).min().item(),
        }

        # ---- α ----
        alpha_flat = F.softplus(blk.b_alpha.data)
        alpha_2d = alpha_flat.reshape(D, N)
        ld['alpha'] = {
            'mean': alpha_flat.mean().item(),
            'std': alpha_flat.std().item(),
            'min': alpha_flat.min().item(),
            'max': alpha_flat.max().item(),
            'per_n_mean': to_list(alpha_2d.mean(dim=0)),
            'diversity_per_channel': alpha_2d.std(dim=1).mean().item(),
            'diversity_per_channel_min': alpha_2d.std(dim=1).min().item(),
        }

        # ---- V_th (from b_th only, not including input-dependent raw_th) ----
        b_th_flat = blk.b_th.data
        vth_flat = 0.1 + b_th_flat.abs()
        vth_2d = vth_flat.reshape(D, N)
        ld['vth'] = {
            'mean': vth_flat.mean().item(),
            'std': vth_flat.std().item(),
            'min': vth_flat.min().item(),
            'max': vth_flat.max().item(),
            'b_th_absmax': b_th_flat.abs().max().item(),
            'per_n_mean': to_list(vth_2d.mean(dim=0)),
        }

        # ---- RF 谱半径 r = √(β²+ω²) ----
        r_flat = (beta_flat ** 2 + omega_flat ** 2).sqrt()
        ld['rf_radius'] = {
            'mean': r_flat.mean().item(),
            'std': r_flat.std().item(),
            'min': r_flat.min().item(),
            'max': r_flat.max().item(),
            'n_violations': int((r_flat > 0.999).sum().item()),
        }

        # ---- Lyapunov 惩罚量 L = (log β)² + (log(1-β))² ----
        log_beta = torch.log(beta_flat.clamp(min=1e-7))
        log_1m_beta = torch.log((1.0 - beta_flat).clamp(min=1e-7))
        lyap = log_beta ** 2 + log_1m_beta ** 2
        ld['lyapunov_penalty'] = {
            'mean': lyap.mean().item(),
            'std': lyap.std().item(),
            'min': lyap.min().item(),
            'max': lyap.max().item(),
        }

        # ---- 振荡周期 T = 2π/arctan(ω/β) ----
        theta = torch.atan2(omega_flat, beta_flat)
        period = (2 * math.pi) / theta.clamp(min=1e-7)
        ld['oscillation_period'] = {
            'mean': period.mean().item(),
            'std': period.std().item(),
            'min': period.min().item(),
            'max': period.max().item(),
        }

        # ---- 原始参数空间 b_beta / b_omega 值 ----
        ld['b_beta_raw'] = {
            'mean': blk.b_beta.data.mean().item(),
            'std': blk.b_beta.data.std().item(),
            'min': blk.b_beta.data.min().item(),
            'max': blk.b_beta.data.max().item(),
        }
        ld['b_omega_raw'] = {
            'mean': blk.b_omega.data.mean().item(),
            'std': blk.b_omega.data.std().item(),
            'min': blk.b_omega.data.min().item(),
            'max': blk.b_omega.data.max().item(),
        }

        # ---- 梯度大小（如果有） ----
        if blk.b_beta.grad is not None:
            ld['b_beta_grad_norm'] = blk.b_beta.grad.norm().item()
        if blk.b_omega.grad is not None:
            ld['b_omega_grad_norm'] = blk.b_omega.grad.norm().item()
        if blk.b_alpha.grad is not None:
            ld['b_alpha_grad_norm'] = blk.b_alpha.grad.norm().item()
        if blk.b_th.grad is not None:
            ld['b_th_grad_norm'] = blk.b_th.grad.norm().item()

        layers_data.append(ld)

    record['layers'] = layers_data
    return record


# ====== 训练循环 ======
all_records = []

# Step -1: 初始状态（训练前）
all_records.append(detailed_snapshot(-1, float('nan')))

print("训练 100 步...")
for step in range(100):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(x, y)
        loss = out.last_loss.mean()
    loss.backward()

    # 记录 unscale 前的梯度范数
    scaler.unscale_(optimizer)

    # 采快照（optimizer.step 之前，梯度已 unscale）
    rec = detailed_snapshot(step, loss.item())

    # 记录补偿前的梯度
    for li, layer_module in enumerate(model.layers):
        blk = layer_module.snn_block
        if blk.b_beta.grad is not None:
            rec['layers'][li]['b_beta_grad_before_comp'] = blk.b_beta.grad.norm().item()
        if blk.b_omega.grad is not None:
            rec['layers'][li]['b_omega_grad_before_comp'] = blk.b_omega.grad.norm().item()

    # β/ω 快照（step 前）
    for li, layer_module in enumerate(model.layers):
        blk = layer_module.snn_block
        rec['layers'][li]['beta_before_step'] = torch.sigmoid(blk.b_beta.data).mean().item()
        rec['layers'][li]['omega_before_step'] = F.softplus(blk.b_omega.data).mean().item()

    scaler.step(optimizer)
    scaler.update()

    # β/ω 快照（step 后）
    for li, layer_module in enumerate(model.layers):
        blk = layer_module.snn_block
        rec['layers'][li]['beta_after_step'] = torch.sigmoid(blk.b_beta.data).mean().item()
        rec['layers'][li]['omega_after_step'] = F.softplus(blk.b_omega.data).mean().item()
        rec['layers'][li]['beta_delta'] = (
            rec['layers'][li]['beta_after_step'] - rec['layers'][li]['beta_before_step'])
        rec['layers'][li]['omega_delta'] = (
            rec['layers'][li]['omega_after_step'] - rec['layers'][li]['omega_before_step'])

    optimizer.zero_grad(set_to_none=True)
    all_records.append(rec)

    if step % 10 == 0 or step == 99:
        s = rec['layers'][0]
        print("  Step %3d | loss=%.4f | beta=[%.4f, %.4f] div=%.4f | omega=[%.4f, %.4f] div=%.4f | "
              "alpha div=%.4f | r_max=%.4f | Vth=[%.4f, %.4f] | lyap=%.4f" % (
                  step, rec['loss'],
                  s['beta']['min'], s['beta']['max'], s['beta']['diversity_per_channel'],
                  s['omega']['min'], s['omega']['max'], s['omega']['diversity_per_channel'],
                  s['alpha']['diversity_per_channel'],
                  s['rf_radius']['max'],
                  s['vth']['min'], s['vth']['max'],
                  s['lyapunov_penalty']['mean']))

# ====== 保存 ======
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snn_adamw_diagnosis.json')
with open(out_path, 'w') as f:
    json.dump(all_records, f, indent=2, ensure_ascii=False, default=str)
print("\n诊断数据已保存: %s" % out_path)

# ====== 打印关键演变摘要 ======
init = all_records[0]   # step -1
final = all_records[-1]  # step 99

print("\n" + "=" * 80)
print("参数演变摘要 (init → step 99)")
print("=" * 80)

for li in range(num_layers):
    i_layer = init['layers'][li]
    f_layer = final['layers'][li]
    print("\n--- Layer %d ---" % li)

    # β
    print("  β:  mean %.4f→%.4f | std %.4f→%.4f | range [%.4f,%.4f]→[%.4f,%.4f] | "
          "N-div %.4f→%.4f" % (
              i_layer['beta']['mean'], f_layer['beta']['mean'],
              i_layer['beta']['std'], f_layer['beta']['std'],
              i_layer['beta']['min'], i_layer['beta']['max'],
              f_layer['beta']['min'], f_layer['beta']['max'],
              i_layer['beta']['diversity_per_channel'],
              f_layer['beta']['diversity_per_channel']))
    print("      per-N mean: %s → %s" % (
        ['%.3f' % v for v in i_layer['beta']['per_n_mean']],
        ['%.3f' % v for v in f_layer['beta']['per_n_mean']]))

    # ω
    print("  ω:  mean %.4f→%.4f | std %.4f→%.4f | range [%.4f,%.4f]→[%.4f,%.4f] | "
          "N-div %.4f→%.4f" % (
              i_layer['omega']['mean'], f_layer['omega']['mean'],
              i_layer['omega']['std'], f_layer['omega']['std'],
              i_layer['omega']['min'], i_layer['omega']['max'],
              f_layer['omega']['min'], f_layer['omega']['max'],
              i_layer['omega']['diversity_per_channel'],
              f_layer['omega']['diversity_per_channel']))
    print("      per-N mean: %s → %s" % (
        ['%.3f' % v for v in i_layer['omega']['per_n_mean']],
        ['%.3f' % v for v in f_layer['omega']['per_n_mean']]))

    # α
    print("  α:  mean %.4f→%.4f | std %.4f→%.4f | N-div %.4f→%.4f" % (
        i_layer['alpha']['mean'], f_layer['alpha']['mean'],
        i_layer['alpha']['std'], f_layer['alpha']['std'],
        i_layer['alpha']['diversity_per_channel'],
        f_layer['alpha']['diversity_per_channel']))

    # V_th
    print("  Vth: mean %.4f→%.4f | range [%.4f,%.4f]→[%.4f,%.4f] | |b_th|_max %.4f→%.4f" % (
        i_layer['vth']['mean'], f_layer['vth']['mean'],
        i_layer['vth']['min'], i_layer['vth']['max'],
        f_layer['vth']['min'], f_layer['vth']['max'],
        i_layer['vth']['b_th_absmax'], f_layer['vth']['b_th_absmax']))

    # RF
    print("  RF:  r_mean %.4f→%.4f | r_max %.4f→%.4f | violations %d→%d" % (
        i_layer['rf_radius']['mean'], f_layer['rf_radius']['mean'],
        i_layer['rf_radius']['max'], f_layer['rf_radius']['max'],
        i_layer['rf_radius']['n_violations'], f_layer['rf_radius']['n_violations']))

    # Lyapunov
    print("  Lyap: mean %.4f→%.4f | max %.4f→%.4f" % (
        i_layer['lyapunov_penalty']['mean'], f_layer['lyapunov_penalty']['mean'],
        i_layer['lyapunov_penalty']['max'], f_layer['lyapunov_penalty']['max']))

    # 振荡周期
    print("  T_osc: mean %.1f→%.1f | range [%.1f,%.1f]→[%.1f,%.1f] steps" % (
        i_layer['oscillation_period']['mean'], f_layer['oscillation_period']['mean'],
        i_layer['oscillation_period']['min'], i_layer['oscillation_period']['max'],
        f_layer['oscillation_period']['min'], f_layer['oscillation_period']['max']))

# β/ω step-over-step 变化统计
print("\n" + "=" * 80)
print("逐步变化统计 (Layer 0)")
print("=" * 80)
deltas_beta = []
deltas_omega = []
for rec in all_records[1:]:  # skip init
    deltas_beta.append(rec['layers'][0].get('beta_delta', 0))
    deltas_omega.append(rec['layers'][0].get('omega_delta', 0))

if deltas_beta:
    db = torch.tensor(deltas_beta)
    dw = torch.tensor(deltas_omega)
    print("  Δβ per step:  mean=%.2e | std=%.2e | min=%.2e | max=%.2e" % (
        db.mean().item(), db.std().item(), db.min().item(), db.max().item()))
    print("  Δω per step:  mean=%.2e | std=%.2e | min=%.2e | max=%.2e" % (
        dw.mean().item(), dw.std().item(), dw.min().item(), dw.max().item()))

print("\nDone.")
