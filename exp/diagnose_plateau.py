"""
v7.5 loss 停滞深度诊断（加载 checkpoint）

不改代码，只观测。诊断项：
  1. 各层 spike 发放率分布 — 信息瓶颈 + 死神经元检测
  2. 残差流中 SNN 贡献占比 — SNN 层是否在做有效计算
  3. h 方向变化 — SNN 是否在改变信息，还是只在加噪声
  4. out_proj 权重变化 — 从初始化到训练后的演化
  5. 隐神经元多样性 — 不同神经元是否学到了不同模式
  6. 梯度流: SNN 路径 vs 残差 skip
  7. 信息瓶颈量化
  8. decode 路径信息损失
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
import torch.nn.functional as F
import math
import numpy as np
from model import SNNLanguageModel
from spikingjelly.activation_based import functional


def main():
    device = 'cuda'
    dtype = torch.bfloat16

    D, N, D_ff = 768, 8, 2304
    num_layers, vocab_size, K = 20, 6144, 16
    seq_len, batch = 512, 1  # batch=1 节省内存

    print("=" * 70)
    print("v7.5 Loss 停滞深度诊断（训练后 checkpoint）")
    print("=" * 70)

    # 加载训练后模型
    ckpt_path = 'checkpoints/pretrain_768_20_6144.pth'
    model = SNNLanguageModel(
        vocab_size=vocab_size, D=D, N=N, D_ff=D_ff,
        num_layers=num_layers, K=K,
    ).to(device).to(dtype)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt.get('step', '?')
        print(f"  Checkpoint loaded: {ckpt_path} (step {step})")
    else:
        print(f"  WARNING: No checkpoint found at {ckpt_path}, using random init")
        step = 0

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    # ============================================================
    # 诊断 1: spike 发放率分布 + 死神经元检测
    # ============================================================
    print("\n" + "=" * 70)
    print("[1] 逐层 spike 发放率分布 + SNN 贡献占比")
    print("=" * 70)
    print(f"  {'Layer':>5s}  {'block_fire%':>11s}  {'ffn_fire%':>10s}  "
          f"{'|snn_blk|':>10s}  {'|snn_ffn|':>10s}  {'|h|':>10s}  "
          f"{'blk/h%':>7s}  {'ffn/h%':>7s}  {'cos_blk':>8s}  {'cos_ffn':>8s}")
    print("  " + "-" * 105)

    model.eval()
    layer_data = []
    with torch.no_grad():
        for layer_module in model.layers:
            functional.reset_net(layer_module)

        h = model._encode_all_tokens(input_ids)  # (TK, batch, D)
        h_norm_init = h.float().norm(dim=-1).mean().item()

        for i, layer in enumerate(model.layers):
            functional.reset_net(layer)
            h_before = h.clone()

            # 子层 1: SNNBlock
            spike_in1 = layer._input_neuron_parallel(layer.input_neuron1, h)
            fire_rate1 = spike_in1.float().mean().item()

            spike_block = layer.snn_block.forward_parallel(spike_in1)
            snn_contrib_block = layer.block_out_proj(spike_block)
            block_norm = snn_contrib_block.float().norm(dim=-1).mean().item()

            h = h + snn_contrib_block

            # 子层 1 方向变化: cos(h_before, snn_contrib)
            h_flat = h_before.float().reshape(-1, D)
            c_flat = snn_contrib_block.float().reshape(-1, D)
            cos_block = F.cosine_similarity(h_flat, c_flat, dim=-1).mean().item()

            # 子层 2: SNNFFN
            h_before_ffn = h.clone()
            spike_in2 = layer._input_neuron_parallel(layer.input_neuron2, h)
            fire_rate2 = spike_in2.float().mean().item()

            spike_ffn = layer.snn_ffn.forward_parallel(spike_in2)
            snn_contrib_ffn = layer.ffn_out_proj(spike_ffn)
            ffn_norm = snn_contrib_ffn.float().norm(dim=-1).mean().item()

            h = h + snn_contrib_ffn

            # 子层 2 方向变化
            h_flat2 = h_before_ffn.float().reshape(-1, D)
            c_flat2 = snn_contrib_ffn.float().reshape(-1, D)
            cos_ffn = F.cosine_similarity(h_flat2, c_flat2, dim=-1).mean().item()

            h_norm = h.float().norm(dim=-1).mean().item()
            block_ratio = block_norm / (h_norm + 1e-10) * 100
            ffn_ratio = ffn_norm / (h_norm + 1e-10) * 100

            print(f"  L{i:>3d}  {fire_rate1*100:>10.1f}%  {fire_rate2*100:>9.1f}%  "
                  f"{block_norm:>10.4f}  {ffn_norm:>10.4f}  {h_norm:>10.4f}  "
                  f"{block_ratio:>6.1f}%  {ffn_ratio:>6.1f}%  "
                  f"{cos_block:>8.4f}  {cos_ffn:>8.4f}")

            layer_data.append({
                'fire1': fire_rate1, 'fire2': fire_rate2,
                'block_norm': block_norm, 'ffn_norm': ffn_norm,
                'h_norm': h_norm, 'cos_block': cos_block, 'cos_ffn': cos_ffn,
                'spike_in1': spike_in1, 'spike_block': spike_block,
            })

    # ============================================================
    # 诊断 2: 输入/隐神经元 per-dim 发放率分布
    # ============================================================
    print(f"\n{'='*70}")
    print("[2] 神经元维度级发放率分布（选取 L0, L9, L19）")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        for layer_module in model.layers:
            functional.reset_net(layer_module)
        h = model._encode_all_tokens(input_ids)

        for i, layer_m in enumerate(model.layers):
            functional.reset_net(layer_m)
            if i in [0, 9, 19]:
                spike_in = layer_m._input_neuron_parallel(layer_m.input_neuron1, h)
                # per-dim firing rate: (D,) — 在 (TK, batch) 上取均值
                per_dim_fire = spike_in.float().mean(dim=(0, 1))  # (D,)
                dead = (per_dim_fire < 0.01).sum().item()
                saturated = (per_dim_fire > 0.99).sum().item()
                p10 = per_dim_fire.quantile(0.1).item()
                p50 = per_dim_fire.quantile(0.5).item()
                p90 = per_dim_fire.quantile(0.9).item()
                print(f"  L{i} input_neuron1 per-dim fire rate:")
                print(f"    dead(<1%): {dead}/{D}  saturated(>99%): {saturated}/{D}")
                print(f"    percentiles: p10={p10:.3f}  p50={p50:.3f}  p90={p90:.3f}")
                print(f"    range: [{per_dim_fire.min().item():.4f}, {per_dim_fire.max().item():.4f}]")

                # 隐神经元: SNNBlock hidden
                spike_block_out = layer_m.snn_block.forward_parallel(spike_in)
                # spike_block_out 是输出神经元的 spike (D,)，不是 hidden
                # 要看 hidden，需要手动展开 SNNBlock.forward_parallel
                # 先跳过，只看 output spike
                per_dim_block_out = spike_block_out.float().mean(dim=(0, 1))
                dead_out = (per_dim_block_out < 0.01).sum().item()
                sat_out = (per_dim_block_out > 0.99).sum().item()
                print(f"  L{i} snn_block output per-dim fire rate:")
                print(f"    dead(<1%): {dead_out}/{D}  saturated(>99%): {sat_out}/{D}")
                print(f"    percentiles: p10={per_dim_block_out.quantile(0.1).item():.3f}  "
                      f"p50={per_dim_block_out.quantile(0.5).item():.3f}  "
                      f"p90={per_dim_block_out.quantile(0.9).item():.3f}")

                snn_contrib = layer_m.block_out_proj(spike_block_out)
                h = h + snn_contrib
                spike_in2 = layer_m._input_neuron_parallel(layer_m.input_neuron2, h)
                spike_ffn_out = layer_m.snn_ffn.forward_parallel(spike_in2)
                h = h + layer_m.ffn_out_proj(spike_ffn_out)
            else:
                h = layer_m.forward_parallel(h)

    # ============================================================
    # 诊断 3: out_proj 权重统计 + 变化量
    # ============================================================
    print(f"\n{'='*70}")
    print("[3] out_proj 权重统计")
    init_std = 0.02 / math.sqrt(2 * num_layers)
    print(f"  初始化 σ = 0.02/√(2×{num_layers}) = {init_std:.6f}")
    print("=" * 70)
    print(f"  {'Layer':>5s}  {'block_σ':>10s}  {'block_|max|':>12s}  "
          f"{'ffn_σ':>10s}  {'ffn_|max|':>12s}  "
          f"{'block_σ/init':>12s}  {'ffn_σ/init':>12s}")
    print("  " + "-" * 80)
    for i, layer_m in enumerate(model.layers):
        bw = layer_m.block_out_proj.weight.float()
        fw = layer_m.ffn_out_proj.weight.float()
        bstd = bw.std().item()
        fstd = fw.std().item()
        print(f"  L{i:>3d}  {bstd:>10.6f}  {bw.abs().max().item():>12.6f}  "
              f"{fstd:>10.6f}  {fw.abs().max().item():>12.6f}  "
              f"{bstd/init_std:>12.2f}x  {fstd/init_std:>12.2f}x")

    # ============================================================
    # 诊断 4: input_neuron β 和 V_th 学习情况
    # ============================================================
    print(f"\n{'='*70}")
    print("[4] 输入神经元参数（β=sigmoid(w), V_th）学习情况")
    print("=" * 70)
    print(f"  初始: β=sigmoid(logit(1/2.0))=0.5, V_th=0.5")
    print(f"  {'Layer':>5s}  {'β1_mean':>8s}  {'β1_std':>8s}  {'Vth1_mean':>10s}  {'Vth1_std':>10s}  "
          f"{'β2_mean':>8s}  {'β2_std':>8s}  {'Vth2_mean':>10s}  {'Vth2_std':>10s}")
    print("  " + "-" * 90)
    for i, layer_m in enumerate(model.layers):
        b1 = torch.sigmoid(layer_m.input_neuron1.w)
        vth1 = layer_m.input_neuron1.v_th
        b2 = torch.sigmoid(layer_m.input_neuron2.w)
        vth2 = layer_m.input_neuron2.v_th
        print(f"  L{i:>3d}  {b1.mean().item():>8.4f}  {b1.std().item():>8.4f}  "
              f"{vth1.mean().item():>10.4f}  {vth1.std().item():>10.4f}  "
              f"{b2.mean().item():>8.4f}  {b2.std().item():>8.4f}  "
              f"{vth2.mean().item():>10.4f}  {vth2.std().item():>10.4f}")

    # ============================================================
    # 诊断 5: 隐神经元参数多样性（SNNBlock b_beta, b_alpha, b_th）
    # ============================================================
    print(f"\n{'='*70}")
    print("[5] SNNBlock 隐神经元调制参数 — 多样性")
    print("=" * 70)
    print(f"  初始 b_beta: logit-spaced [0.80,0.99] → [{torch.log(torch.tensor(0.8)/0.2).item():.2f}, {torch.log(torch.tensor(0.99)/0.01).item():.2f}]")
    print(f"  {'Layer':>5s}  {'b_beta σ':>10s}  {'b_beta range':>14s}  "
          f"{'β range':>14s}  {'b_th σ':>10s}  {'b_th range':>14s}")
    print("  " + "-" * 75)
    for i, layer_m in enumerate(model.layers):
        bb = layer_m.snn_block.b_beta
        ba = layer_m.snn_block.b_alpha
        bt = layer_m.snn_block.b_th
        beta_actual = torch.sigmoid(bb)
        print(f"  L{i:>3d}  {bb.std().item():>10.4f}  "
              f"[{bb.min().item():>5.2f},{bb.max().item():>5.2f}]  "
              f"[{beta_actual.min().item():.3f},{beta_actual.max().item():.3f}]  "
              f"{bt.std().item():>10.4f}  "
              f"[{bt.min().item():>5.2f},{bt.max().item():>5.2f}]")

    # ============================================================
    # 诊断 6: 梯度流 — 关键路径对比
    # ============================================================
    print(f"\n{'='*70}")
    print("[6] 梯度分析: SNN 内部 vs 残差 skip vs 输入神经元")
    print("=" * 70)
    model.train()
    model.zero_grad()
    for layer_module in model.layers:
        functional.reset_net(layer_module)

    output = model(input_ids, labels)
    loss = output.last_loss.mean()
    loss.backward()
    print(f"  Loss = {loss.item():.4f}")

    print(f"  {'Layer':>5s}  {'|∇ blk_proj|':>13s}  {'|∇ ffn_proj|':>13s}  "
          f"{'|∇ W_in|':>10s}  {'|∇ W_out|':>10s}  "
          f"{'|∇ n1.w|':>10s}  {'|∇ n1.vth|':>10s}  "
          f"{'ratio proj/Win':>15s}")
    print("  " + "-" * 100)
    for i, layer_m in enumerate(model.layers):
        bp = layer_m.block_out_proj.weight.grad
        fp = layer_m.ffn_out_proj.weight.grad
        win = layer_m.snn_block.W_in.weight.grad
        wout = layer_m.snn_block.W_out.weight.grad
        n1w = layer_m.input_neuron1.w.grad
        n1vth = layer_m.input_neuron1.v_th.grad

        bp_n = bp.float().norm().item() if bp is not None else 0
        fp_n = fp.float().norm().item() if fp is not None else 0
        win_n = win.float().norm().item() if win is not None else 0
        wout_n = wout.float().norm().item() if wout is not None else 0
        n1w_n = n1w.float().norm().item() if n1w is not None else 0
        n1vth_n = n1vth.float().norm().item() if n1vth is not None else 0

        ratio = bp_n / (win_n + 1e-10)
        print(f"  L{i:>3d}  {bp_n:>13.6f}  {fp_n:>13.6f}  "
              f"{win_n:>10.6f}  {wout_n:>10.6f}  "
              f"{n1w_n:>10.6f}  {n1vth_n:>10.6f}  "
              f"{ratio:>15.2f}")

    # 特别关注: embedding 和 encode/decode proj 的梯度
    print(f"\n  关键组件梯度:")
    embed_grad = model.embed_tokens.weight.grad
    if embed_grad is not None:
        print(f"    embed_tokens: |∇| = {embed_grad.float().norm().item():.6f}")
    enc_grad = model.encode_proj.weight.grad
    if enc_grad is not None:
        print(f"    encode_proj:  |∇| = {enc_grad.float().norm().item():.6f}")
    dec_grad = model.decode_proj.weight.grad
    if dec_grad is not None:
        print(f"    decode_proj:  |∇| = {dec_grad.float().norm().item():.6f}")
    norm_grad = model.norm.gain.grad
    if norm_grad is not None:
        print(f"    norm.gain:    |∇| = {norm_grad.float().norm().item():.6f}")

    # ============================================================
    # 诊断 7: decode 路径信息损失分析
    # ============================================================
    print(f"\n{'='*70}")
    print("[7] Decode 路径分析")
    print("=" * 70)
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        for layer_module in model.layers:
            functional.reset_net(layer_module)

        h = model._encode_all_tokens(input_ids)

        # 逐层前向
        for layer_module in model.layers:
            functional.reset_net(layer_module)
            h = layer_module.forward_parallel(h)

        # h 是连续值 (TK, batch, D)
        print(f"  最终层输出 h: shape={h.shape}")
        print(f"    |h| mean = {h.float().norm(dim=-1).mean().item():.4f}")
        print(f"    h min/max = [{h.float().min().item():.4f}, {h.float().max().item():.4f}]")

        # decode: K-bit 加权求和
        decoded = model._decode_all_tokens(h, seq_len)
        print(f"  decode 后: shape={decoded.shape}")
        print(f"    |decoded| mean = {decoded.float().norm(dim=-1).mean().item():.4f}")
        print(f"    decoded min/max = [{decoded.float().min().item():.4f}, {decoded.float().max().item():.4f}]")

        # decode_proj → norm → logits
        h_dec = model.decode_proj(decoded)
        print(f"  decode_proj 后:")
        print(f"    |h_dec| mean = {h_dec.float().norm(dim=-1).mean().item():.4f}")

        h_norm = model.norm(h_dec)
        print(f"  norm 后:")
        print(f"    |h_norm| mean = {h_norm.float().norm(dim=-1).mean().item():.4f}")

        logits = F.linear(h_norm, model.embed_tokens.weight)
        print(f"  logits:")
        print(f"    |logits| mean = {logits.float().norm(dim=-1).mean().item():.4f}")
        print(f"    logits std per token = {logits.float().std(dim=-1).mean().item():.4f}")
        print(f"    logits max per token = {logits.float().max(dim=-1)[0].mean().item():.4f}")

        # softmax 信息: entropy of predicted distribution
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        print(f"    prediction entropy: mean={entropy.mean().item():.4f}, "
              f"min={entropy.min().item():.4f}, max={entropy.max().item():.4f}")
        print(f"    uniform entropy = ln({vocab_size}) = {math.log(vocab_size):.4f}")
        print(f"    entropy ratio (pred/uniform) = {entropy.mean().item()/math.log(vocab_size)*100:.1f}%")

        # top-1 prediction 覆盖率
        top1 = logits.argmax(dim=-1)  # (batch, seq_len)
        unique_preds = top1.unique().numel()
        print(f"    unique top-1 predictions: {unique_preds}/{vocab_size} "
              f"({unique_preds/vocab_size*100:.1f}% of vocab)")

    # ============================================================
    # 诊断 8: K-bit 编码信息保留
    # ============================================================
    print(f"\n{'='*70}")
    print("[8] K-bit 编码信息损失")
    print("=" * 70)
    with torch.no_grad():
        emb = model.embed_tokens(input_ids)  # (batch, seq_len, D)
        h_enc = torch.sigmoid(model.encode_proj(emb))  # (batch, seq_len, D) ∈ [0,1]

        # K-bit 量化
        scaled = h_enc.unsqueeze(2) * model.bit_scales.view(1, 1, K, 1)
        bit_hard = torch.floor(scaled) % 2
        # 从 K bits 重建: sum(bit * 2^{-(k+1)})
        reconstructed = torch.einsum('bskd,k->bsd', bit_hard, model.bit_weights)

        # 量化误差
        quant_error = (h_enc - reconstructed).float()
        max_error = quant_error.abs().max().item()
        mse = (quant_error ** 2).mean().item()
        print(f"  K={K} bit 量化:")
        print(f"    理论最大误差: {2**(-K):.6f}")
        print(f"    实际最大误差: {max_error:.6f}")
        print(f"    MSE: {mse:.8f}")
        print(f"    h_enc 值域: [{h_enc.min().item():.4f}, {h_enc.max().item():.4f}]")
        print(f"    h_enc 均值: {h_enc.mean().item():.4f}")

        # h_enc 值分布 — 如果集中在 0 或 1 附近, 低位 bits 全是 0/1
        h_enc_f = h_enc.float()
        print(f"    h_enc 分位数: p10={h_enc_f.quantile(0.1).item():.4f} "
              f"p50={h_enc_f.quantile(0.5).item():.4f} "
              f"p90={h_enc_f.quantile(0.9).item():.4f}")

        # 各 bit 位的利用率
        print(f"    各 bit 的发放率:")
        for k in range(K):
            bits_k = bit_hard[:, :, k, :]
            fire_k = bits_k.float().mean().item()
            print(f"      bit {k:>2d} (weight={model.bit_weights[k].item():.6f}): "
                  f"fire={fire_k*100:.1f}%")

    # ============================================================
    # 诊断 9: 信息瓶颈量化
    # ============================================================
    print(f"\n{'='*70}")
    print("[9] 信息瓶颈总结")
    print("=" * 70)
    print(f"  每 token 经过 SNN 子层:")
    print(f"  - 输入 spike: D={D} 维 × K={K} 步 = {D*K} binary values")
    print(f"  - 最大信息容量: {D*K} bits")
    print(f"  - bf16 对应: D={D} × 16 bits = {D*16} bits")
    print(f"  - 信息比 SNN/continuous: {D*K/(D*16):.2f}x")
    print()

    # 用诊断 1 的数据计算实际信息
    for idx, ld in enumerate(layer_data):
        if idx not in [0, 9, 19]:
            continue
        for key, name in [('fire1', 'input_n1'), ('fire2', 'input_n2')]:
            p = ld[key]
            if 0 < p < 1:
                ent = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            else:
                ent = 0
            eff = D * K * ent
            print(f"  L{idx} {name}: p={p:.3f}, entropy={ent:.3f} bits/element, "
                  f"effective={eff:.0f}/{D*K} bits ({ent*100:.1f}%)")

    # ============================================================
    # 关键问题汇总
    # ============================================================
    print(f"\n{'='*70}")
    print("自动诊断总结")
    print("=" * 70)

    issues = []

    # 检查 1: SNN 贡献是否太小
    avg_block_ratio = np.mean([ld['block_norm'] / (ld['h_norm'] + 1e-10) for ld in layer_data])
    avg_ffn_ratio = np.mean([ld['ffn_norm'] / (ld['h_norm'] + 1e-10) for ld in layer_data])
    if avg_block_ratio < 0.05:
        issues.append(f"  [CRITICAL] SNN Block 贡献极小: 平均 {avg_block_ratio*100:.1f}% of |h|, "
                      f"out_proj 初始化 σ={init_std:.6f} 可能压死了学习信号")
    if avg_ffn_ratio < 0.05:
        issues.append(f"  [CRITICAL] SNN FFN 贡献极小: 平均 {avg_ffn_ratio*100:.1f}% of |h|")

    # 检查 2: 发放率极端
    for idx, ld in enumerate(layer_data):
        if ld['fire1'] < 0.05 or ld['fire1'] > 0.95:
            issues.append(f"  [WARNING] L{idx} input_n1 发放率极端: {ld['fire1']*100:.1f}%")
        if ld['fire2'] < 0.05 or ld['fire2'] > 0.95:
            issues.append(f"  [WARNING] L{idx} input_n2 发放率极端: {ld['fire2']*100:.1f}%")

    # 检查 3: cos similarity 接近 0 说明 SNN 输出与 h 正交（加噪声而非有效计算）
    avg_cos = np.mean([abs(ld['cos_block']) for ld in layer_data])
    if avg_cos < 0.05:
        issues.append(f"  [WARNING] SNN 输出与 h 几乎正交 (avg |cos|={avg_cos:.3f}), "
                      f"SNN 可能在加随机噪声而非有效计算")

    if not issues:
        print("  未检测到明显异常。")
    else:
        for issue in issues:
            print(issue)

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
