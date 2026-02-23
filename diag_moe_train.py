"""
MoE SNN 训练诊断: 死寂/癫痫/趋同检测 + 参数动态观测 + SNNAdamW 验证

检测项目:
  1. 死寂 (Dead):     spike rate < 0.01 — 神经元从不发放
  2. 癫痫 (Epileptic): spike rate > 0.99 — 神经元总是发放
  3. 趋同 (Convergence): expert 权重余弦相似度 > 0.95 — expert 退化为相同
  4. 参数漂移:        β, v_th 分布随训练变化
  5. 梯度健康:        各参数组梯度范数
  6. 负载均衡:        expert 选中次数分布
  7. SNNAdamW 约束:   β 范围, v_th 上界, RF 稳定性
"""

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from model import SNNLanguageModel
from atomic_ops.snn_adamw import SNNAdamW
from atomic_ops.snn_decoder_layer import _fused_residual_center
from atomic_ops.parallel_scan import plif_rowparam_forward_alpha


# ============================================================
# 构建模型和优化器（参数组完全匹配 train.py）
# ============================================================
def build(cfg, lr=2e-4, neuron_lr_mult=10.0, grad_clip=1.0):
    torch.manual_seed(42)
    model = SNNLanguageModel(**cfg).cuda()

    _pg = model.get_param_groups()

    _wd = {
        'embedding': 0.1, 'decode': 0.1,
        'W_in': 0.1, 'W_beta': 0.1, 'W_alpha': 0.1, 'W_th': 0.1, 'W_omega': 0.1,
        'W_gate': 0.1, 'W_skip': 0.1, 'W_out': 0.1,
        'residual_projs': 0.1,
        'ffn_gate_proj': 0.1, 'ffn_up_proj': 0.1, 'ffn_down_proj': 0.1, 'ffn_skip_proj': 0.1,
        'ffn_expert_projs': 0.1,
        'ffn_expert_neurons': 0.0, 'ffn_router': 0.0,
        'b_beta': 0.0, 'b_alpha': 0.0, 'b_th': 0.0, 'b_omega': 0.0,
        'input_neurons': 0.0, 'block_output_neuron': 0.0,
        'ffn_neurons': 0.0, 'output_neuron': 0.0,
        'norm': 0.0, 'rms_norms': 0.0,
    }
    _dyn = {'b_beta': 'b_beta', 'b_alpha': 'b_alpha', 'b_omega': 'b_omega', 'b_th': 'b_th'}
    _neuron = {'input_neurons', 'b_beta', 'b_alpha', 'b_th', 'b_omega',
               'block_output_neuron', 'ffn_neurons', 'ffn_expert_neurons', 'output_neuron'}

    param_groups = []
    for key, params in _pg.items():
        if not params:
            continue
        lm = neuron_lr_mult if key in _neuron else 1.0
        param_groups.append({
            'params': params, 'lr': lr * lm, 'lr_mult': lm,
            'weight_decay': _wd.get(key, 0.1),
            'dynamics': _dyn.get(key), 'N': cfg.get('N') if key in ('b_beta', 'b_alpha', 'b_omega') else None,
        })

    opt = SNNAdamW(param_groups, lr=lr, betas=(0.9, 0.95), grad_clip=grad_clip)
    return model, opt


# ============================================================
# 参数统计（无需前向传播）
# ============================================================
@torch.no_grad()
def param_stats(model):
    """检查所有神经元参数的健康状态。"""
    stats = {}
    for i, layer in enumerate(model.layers):
        # SNNBlock 动力学参数
        block = layer.snn_block
        beta_block = torch.sigmoid(block.b_beta)
        stats[f'L{i}_block_β'] = (beta_block.min().item(), beta_block.mean().item(), beta_block.max().item())

        # Input neurons
        for j, neuron in enumerate([layer.input_neuron1, layer.input_neuron2]):
            beta_in = neuron.beta
            stats[f'L{i}_in{j+1}_β'] = (beta_in.min().item(), beta_in.mean().item(), beta_in.max().item())
            stats[f'L{i}_in{j+1}_vth'] = (neuron.v_th.min().item(), neuron.v_th.mean().item(), neuron.v_th.max().item())

        if not layer.use_moe:
            continue
        moe = layer.snn_ffn
        E = moe.num_experts

        # Expert GU β and v_th
        beta_gu = torch.sigmoid(moe.expert_gu_w)  # (E, 2*D_ff)
        for e in range(E):
            stats[f'L{i}_E{e}_gu_β'] = (beta_gu[e].min().item(), beta_gu[e].mean().item(), beta_gu[e].max().item())
        stats[f'L{i}_gu_vth'] = (moe.expert_gu_v_th.min().item(), moe.expert_gu_v_th.mean().item(), moe.expert_gu_v_th.max().item())

        # Expert output β and v_th
        beta_out = torch.sigmoid(moe.expert_out_w)  # (E, D)
        for e in range(E):
            stats[f'L{i}_E{e}_out_β'] = (beta_out[e].min().item(), beta_out[e].mean().item(), beta_out[e].max().item())
        stats[f'L{i}_out_vth'] = (moe.expert_out_v_th.min().item(), moe.expert_out_v_th.mean().item(), moe.expert_out_v_th.max().item())

        # Shared expert neurons
        shared = moe.shared_expert
        for name, neuron in [('gate', shared.gate_neuron), ('up', shared.up_neuron), ('out', shared.output_neuron)]:
            beta_s = neuron.beta
            stats[f'L{i}_shared_{name}_β'] = (beta_s.min().item(), beta_s.mean().item(), beta_s.max().item())

    return stats


# ============================================================
# Expert 趋同检测
# ============================================================
@torch.no_grad()
def expert_similarity(model):
    """计算 expert 权重对之间的余弦相似度。"""
    sims = {}
    for i, layer in enumerate(model.layers):
        if not layer.use_moe:
            continue
        moe = layer.snn_ffn
        E = moe.num_experts

        # W_gus 相似度
        W = moe.expert_W_gus.reshape(E, -1)  # (E, flattened)
        for a in range(E):
            for b in range(a + 1, E):
                sim = F.cosine_similarity(W[a].unsqueeze(0), W[b].unsqueeze(0)).item()
                sims[f'L{i}_Wgus_E{a}E{b}'] = sim

        # W_down 相似度
        W_d = moe.expert_W_down.reshape(E, -1)
        for a in range(E):
            for b in range(a + 1, E):
                sim = F.cosine_similarity(W_d[a].unsqueeze(0), W_d[b].unsqueeze(0)).item()
                sims[f'L{i}_Wdown_E{a}E{b}'] = sim

        # Neuron param 相似度 (gu_w)
        for a in range(E):
            for b in range(a + 1, E):
                sim = F.cosine_similarity(moe.expert_gu_w[a].unsqueeze(0), moe.expert_gu_w[b].unsqueeze(0)).item()
                sims[f'L{i}_gu_w_E{a}E{b}'] = sim

    return sims


# ============================================================
# 梯度范数
# ============================================================
@torch.no_grad()
def grad_norms(model):
    """各参数组的梯度 L2 范数。"""
    _pg = model.get_param_groups()
    norms = {}
    for key, params in _pg.items():
        grads = [p.grad for p in params if p.grad is not None]
        if grads:
            total = torch.sqrt(sum(g.flatten().float().pow(2).sum() for g in grads))
            norms[key] = total.item()
    return norms


# ============================================================
# Spike Rate 诊断前向（手动步进，不用 checkpoint）
# ============================================================
@torch.no_grad()
def spike_rate_forward(model, x):
    """无梯度诊断前向，手动步进各组件，捕获 spike rate。"""
    model.eval()
    rates = {}

    spike_seq = model.encode(x)
    rates['encode'] = spike_seq.mean().item()

    h = spike_seq
    for i, layer in enumerate(model.layers):
        functional.reset_net(layer)

        # 子层 1: input neuron 1 → snn_block
        spike_in1 = layer._input_neuron_parallel(layer.input_neuron1, layer.block_norm(h))
        rates[f'L{i}_in1'] = spike_in1.mean().item()

        spike_block = layer.snn_block.forward_parallel(spike_in1)
        rates[f'L{i}_block'] = spike_block.mean().item()

        spike_bw = layer._apply_bit_weights(spike_block)
        h = _fused_residual_center(h, layer.block_out_proj(spike_bw))

        # 子层 2: input neuron 2 → FFN/MoE
        spike_in2 = layer._input_neuron_parallel(layer.input_neuron2, layer.ffn_norm(h))
        rates[f'L{i}_in2'] = spike_in2.mean().item()

        if layer.use_moe:
            moe = layer.snn_ffn
            functional.reset_net(moe)

            # Shared expert
            shared_out = moe.shared_expert.forward_parallel(spike_in2)
            rates[f'L{i}_shared'] = shared_out.mean().item()

            # Per-expert spike rates via _batched_expert_forward
            expert_outs = moe._batched_expert_forward(spike_in2)  # (E, TK, B, D)
            for e in range(moe.num_experts):
                rates[f'L{i}_E{e}_out'] = expert_outs[e].mean().item()
            rates[f'L{i}_expert_avg'] = expert_outs.mean().item()

            # Router distribution
            TK, B, D = spike_in2.shape
            T = TK // moe.K
            h_tokens = h.reshape(T, moe.K, B, D).mean(dim=1)
            router_logits = moe.router(h_tokens)
            biased_logits = router_logits + moe.expert_bias
            _, top_k_idx = biased_logits.topk(moe.top_k, dim=-1)
            counts = torch.bincount(top_k_idx.reshape(-1), minlength=moe.num_experts)
            rates[f'L{i}_routing'] = counts.float().cpu().tolist()

            # Combine for residual (use routing weights)
            top_k_original = router_logits.gather(-1, top_k_idx)
            routing_weights = F.softmax(top_k_original, dim=-1)
            full_weights = torch.zeros(T, B, moe.num_experts, device=h.device, dtype=h.dtype)
            full_weights.scatter_add_(-1, top_k_idx, routing_weights)
            weights_tk = full_weights.unsqueeze(1).expand(T, moe.K, B, moe.num_experts).reshape(TK, B, moe.num_experts)

            from atomic_ops.moe_kernels import moe_combine
            out_flat = moe_combine(
                shared_out.reshape(TK * B, D),
                expert_outs.reshape(moe.num_experts, TK * B, D),
                weights_tk.reshape(TK * B, moe.num_experts).contiguous(),
            )
            spike_ffn = out_flat.reshape(TK, B, D)
        else:
            spike_ffn = layer.snn_ffn.forward_parallel(spike_in2)

        rates[f'L{i}_ffn'] = spike_ffn.mean().item()
        spike_bw2 = layer._apply_bit_weights(spike_ffn)
        h = _fused_residual_center(h, layer.ffn_out_proj(spike_bw2))

    model.train()
    return rates


# ============================================================
# 打印工具
# ============================================================
DEAD_TH = 0.01
EPILEPTIC_TH = 0.99
CONVERGENCE_TH = 0.95

def print_header(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

def print_spike_report(rates, num_layers, num_experts):
    """打印 spike rate 表格 + 死寂/癫痫警告。"""
    dead, epileptic = [], []

    print(f"\n  {'Component':<25} {'Rate':>8}  Status")
    print(f"  {'-'*50}")
    print(f"  {'encode':<25} {rates['encode']:8.4f}")

    for i in range(num_layers):
        for key, label in [
            (f'L{i}_in1', f'L{i} InputNeuron1'),
            (f'L{i}_block', f'L{i} SNNBlock'),
            (f'L{i}_in2', f'L{i} InputNeuron2'),
            (f'L{i}_shared', f'L{i} SharedExpert'),
        ]:
            if key not in rates:
                continue
            r = rates[key]
            status = ''
            if r < DEAD_TH:
                status = '⚠ DEAD'
                dead.append(label)
            elif r > EPILEPTIC_TH:
                status = '⚠ EPILEPTIC'
                epileptic.append(label)
            print(f"  {label:<25} {r:8.4f}  {status}")

        for e in range(num_experts):
            key = f'L{i}_E{e}_out'
            if key not in rates:
                continue
            r = rates[key]
            label = f'L{i} Expert{e} Out'
            status = ''
            if r < DEAD_TH:
                status = '⚠ DEAD'
                dead.append(label)
            elif r > EPILEPTIC_TH:
                status = '⚠ EPILEPTIC'
                epileptic.append(label)
            print(f"  {label:<25} {r:8.4f}  {status}")

        key = f'L{i}_ffn'
        if key in rates:
            print(f"  {'L'+str(i)+' FFN Combined':<25} {rates[key]:8.4f}")

        # Routing
        rkey = f'L{i}_routing'
        if rkey in rates:
            counts = rates[rkey]
            total = sum(counts)
            pct = [f'{c/total*100:.1f}%' for c in counts]
            print(f"  {'L'+str(i)+' Routing':<25} {' '.join(pct)}")

    return dead, epileptic


def print_param_report(stats, num_layers, num_experts):
    """打印参数统计表格。"""
    print(f"\n  {'Parameter':<30} {'Min':>8} {'Mean':>8} {'Max':>8}")
    print(f"  {'-'*58}")
    for i in range(num_layers):
        for e in range(num_experts):
            key = f'L{i}_E{e}_gu_β'
            if key in stats:
                lo, mu, hi = stats[key]
                print(f"  {key:<30} {lo:8.4f} {mu:8.4f} {hi:8.4f}")
        for e in range(num_experts):
            key = f'L{i}_E{e}_out_β'
            if key in stats:
                lo, mu, hi = stats[key]
                print(f"  {key:<30} {lo:8.4f} {mu:8.4f} {hi:8.4f}")
        for key in [f'L{i}_gu_vth', f'L{i}_out_vth']:
            if key in stats:
                lo, mu, hi = stats[key]
                print(f"  {key:<30} {lo:8.4f} {mu:8.4f} {hi:8.4f}")
        # Input neurons
        for j in [1, 2]:
            key = f'L{i}_in{j}_β'
            if key in stats:
                lo, mu, hi = stats[key]
                print(f"  {key:<30} {lo:8.4f} {mu:8.4f} {hi:8.4f}")


def print_similarity_report(sims, num_layers, num_experts):
    """打印 expert 趋同检测。"""
    convergence_alerts = []
    print(f"\n  {'Expert Pair':<30} {'Wgus':>8} {'Wdown':>8} {'gu_w':>8}")
    print(f"  {'-'*58}")
    for i in range(num_layers):
        for a in range(num_experts):
            for b in range(a + 1, num_experts):
                s1 = sims.get(f'L{i}_Wgus_E{a}E{b}', 0)
                s2 = sims.get(f'L{i}_Wdown_E{a}E{b}', 0)
                s3 = sims.get(f'L{i}_gu_w_E{a}E{b}', 0)
                alert = ''
                if s1 > CONVERGENCE_TH or s2 > CONVERGENCE_TH:
                    alert = '⚠ CONVERGING'
                    convergence_alerts.append(f'L{i} E{a}-E{b}')
                print(f"  L{i} E{a}-E{b:<23} {s1:8.4f} {s2:8.4f} {s3:8.4f}  {alert}")
    return convergence_alerts


def print_grad_report(gnorms):
    """打印梯度范数。"""
    print(f"\n  {'Param Group':<30} {'Grad Norm':>10}")
    print(f"  {'-'*42}")
    for key in sorted(gnorms.keys()):
        print(f"  {key:<30} {gnorms[key]:10.4f}")


# ============================================================
# 主训练诊断循环
# ============================================================
def main():
    N_ITERS = 100
    DIAG_INTERVAL = 10
    BATCH_SIZE = 4
    SEQ_LEN = 128

    cfg = dict(
        vocab_size=6144, D=256, N=4, K=16, num_layers=4, D_ff=768,
        use_moe=True, num_experts=4, top_k=2,
        D_ff_shared=256, D_ff_expert=128,
    )

    print_header("MoE SNN 训练诊断")
    print(f"  Config: D={cfg['D']}, layers={cfg['num_layers']}, E={cfg['num_experts']}, "
          f"top_k={cfg['top_k']}, K={cfg['K']}")
    print(f"  Training: {N_ITERS} iters, batch={BATCH_SIZE}x{SEQ_LEN}, diag every {DIAG_INTERVAL} steps")
    print(f"  Optimizer: SNNAdamW (10-phase, lr=2e-4, neuron_lr_mult=10x)")

    model, optimizer = build(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    loss_history = []
    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

    for step in range(N_ITERS):
        # ---- 生成随机数据 ----
        x = torch.randint(0, cfg['vocab_size'], (BATCH_SIZE, SEQ_LEN), device='cuda')
        y = torch.randint(0, cfg['vocab_size'], (BATCH_SIZE, SEQ_LEN), device='cuda')

        # ---- 训练步 ----
        for layer in model.layers:
            functional.reset_net(layer)
        functional.reset_net(model.output_neuron)

        with ctx:
            out = model(x, y)
            loss = out.last_loss.mean()

        loss.backward()

        # 保存梯度范数（在 optimizer.step 之前，因为 step 会修改梯度）
        if step % DIAG_INTERVAL == 0 or step == N_ITERS - 1:
            gnorms = grad_norms(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ---- Expert bias 更新 ----
        for layer in model.layers:
            if layer.use_moe and hasattr(layer, '_last_router_info'):
                layer.snn_ffn.update_expert_bias(
                    layer._last_router_info['expert_counts'])

        loss_val = loss.item()
        loss_history.append(loss_val)

        # ---- 诊断 ----
        if step % DIAG_INTERVAL == 0 or step == N_ITERS - 1:
            print_header(f"Step {step}/{N_ITERS}  |  Loss = {loss_val:.4f}")

            # 参数统计
            print("\n  [Parameter Health]")
            ps = param_stats(model)
            print_param_report(ps, cfg['num_layers'], cfg['num_experts'])

            # Expert 趋同检测
            print("\n  [Expert Convergence Detection]")
            sims = expert_similarity(model)
            conv_alerts = print_similarity_report(sims, cfg['num_layers'], cfg['num_experts'])

            # 梯度范数
            print("\n  [Gradient Norms]")
            print_grad_report(gnorms)

            # Spike rate 诊断
            print("\n  [Spike Rates]")
            with ctx:
                rates = spike_rate_forward(model, x)
            dead, epileptic = print_spike_report(rates, cfg['num_layers'], cfg['num_experts'])

            # Expert bias
            print("\n  [Expert Bias]")
            for i, layer in enumerate(model.layers):
                if layer.use_moe:
                    bias = layer.snn_ffn.expert_bias
                    print(f"  L{i}: {bias.cpu().tolist()}")

            # SNNAdamW 约束检查
            print("\n  [SNNAdamW Constraints]")
            for i, layer in enumerate(model.layers):
                block = layer.snn_block
                beta = torch.sigmoid(block.b_beta)
                omega = F.softplus(block.b_omega)
                r_sq = beta * beta + omega * omega
                r_max = r_sq.sqrt().max().item()
                beta_min_val = beta.min().item()
                beta_max_val = beta.max().item()
                print(f"  L{i} block: β∈[{beta_min_val:.4f}, {beta_max_val:.4f}], "
                      f"RF r_max={r_max:.4f} (limit=0.999)")

            # 警告汇总
            if dead or epileptic or conv_alerts:
                print("\n  [WARNINGS]")
                if dead:
                    print(f"  Dead neurons ({len(dead)}): {', '.join(dead)}")
                if epileptic:
                    print(f"  Epileptic neurons ({len(epileptic)}): {', '.join(epileptic)}")
                if conv_alerts:
                    print(f"  Converging experts ({len(conv_alerts)}): {', '.join(conv_alerts)}")

        elif step % 10 == 0:
            print(f"  Step {step:3d}  loss={loss_val:.4f}")

    # ============================================================
    # 最终报告
    # ============================================================
    print_header("FINAL REPORT")

    print(f"\n  Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
    # 10-step 移动平均
    if len(loss_history) >= 20:
        early_avg = sum(loss_history[:10]) / 10
        late_avg = sum(loss_history[-10:]) / 10
        print(f"  Loss (10-step avg): {early_avg:.4f} → {late_avg:.4f}")
        if late_avg < early_avg * 0.95:
            print(f"  ✓ Loss 正在收敛")
        elif late_avg > early_avg * 1.05:
            print(f"  ✗ Loss 在发散！")
        else:
            print(f"  ~ Loss 变化不大（可能需要更多迭代）")

    # Final spike rates
    print("\n  [Final Spike Rates]")
    with torch.no_grad(), ctx:
        final_rates = spike_rate_forward(model, x)
    dead, epileptic = print_spike_report(final_rates, cfg['num_layers'], cfg['num_experts'])

    # Final similarity
    print("\n  [Final Expert Similarity]")
    sims = expert_similarity(model)
    conv_alerts = print_similarity_report(sims, cfg['num_layers'], cfg['num_experts'])

    # Summary
    print_header("SUMMARY")
    print(f"  Dead neurons:       {len(dead)}")
    print(f"  Epileptic neurons:  {len(epileptic)}")
    print(f"  Converging experts: {len(conv_alerts)}")

    if not dead and not epileptic and not conv_alerts:
        print(f"\n  ✓ 所有检测通过 — 训练正常")
    else:
        print(f"\n  ✗ 发现问题 — 请检查上方警告")

    print(f"\n{'='*90}\n")


if __name__ == '__main__':
    main()
