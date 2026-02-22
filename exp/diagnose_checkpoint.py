"""
v7.5 Checkpoint 深度诊断

方法：Monkey-patch forward_parallel 采集中间统计，不手动复制任何前向逻辑。
  Phase 1: 参数统计（checkpoint vs 随机初始化的差异）
  Phase 2: no_grad 推理 + monkey-patch 采集逐层激活/发放率
  Phase 3: forward+backward 梯度分析
  Phase 4: logits 分布
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda-13.0/bin/ptxas'

import torch
import torch.nn.functional as F
from model import SNNLanguageModel
from spikingjelly.activation_based import functional
from collections import defaultdict
import math

device = 'cuda'
dtype = torch.bfloat16

D = 768
N = 8
D_ff = 2304
num_layers = 20
vocab_size = 6144
K = 16
seq_len = 512
batch = 2


def main():
    ckpt_path = 'checkpoints/pretrain_768_20_6144.pth'

    print("=" * 80)
    print(f"Checkpoint 深度诊断: {ckpt_path}")
    print("=" * 80)

    # ---- 构建模型并加载 checkpoint ----
    model = SNNLanguageModel(
        vocab_size=vocab_size, D=D, N=N, D_ff=D_ff,
        num_layers=num_layers, K=K,
    ).to(device).to(dtype)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    print(f"  step={ckpt.get('step')}, epoch={ckpt.get('epoch')}, "
          f"best_loss={ckpt.get('best_loss')}, tokens_seen={ckpt.get('tokens_seen')}")
    model.load_state_dict(state_dict)
    print("  模型加载完成\n")

    # ==================================================================
    # Phase 1: 参数变化分析 (checkpoint vs 初始化)
    # ==================================================================
    print("=" * 80)
    print("[1] 参数变化分析 (checkpoint vs 随机初始化)")
    print("=" * 80)

    # 构建同结构的初始化参考模型
    torch.manual_seed(0)  # 确保 init 可复现
    model_init = SNNLanguageModel(
        vocab_size=vocab_size, D=D, N=N, D_ff=D_ff,
        num_layers=num_layers, K=K,
    ).to(device).to(dtype)

    print(f"\n{'参数名':>55s} {'|Δ|_mean':>10s} {'|Δ|_max':>10s} {'|p|_mean':>10s} {'变化率':>8s}")
    print("-" * 100)

    group_deltas = defaultdict(list)  # group_name → [(ratio, delta_mean)]

    for (name, p_ckpt), (_, p_init) in zip(
        model.named_parameters(), model_init.named_parameters()
    ):
        delta = (p_ckpt.float() - p_init.float())
        d_abs_mean = delta.abs().mean().item()
        d_abs_max = delta.abs().max().item()
        p_abs_mean = p_ckpt.float().abs().mean().item()
        ratio = d_abs_mean / (p_abs_mean + 1e-10)

        # 提取层号和短名
        if 'layers.' in name:
            parts = name.split('.')
            li = int(parts[1])
            short = '.'.join(parts[2:])
        else:
            li = -1
            short = name

        # 按语义分组
        if 'input_neuron' in short:
            gk = 'input_neuron'
        elif 'snn_block' in short:
            if any(k in short for k in ['b_beta', 'b_alpha', 'b_th']):
                gk = 'snn_block.bias'
            elif 'W_in' in short or 'W_out' in short:
                gk = 'snn_block.W_io'
            elif 'W_beta' in short or 'W_alpha' in short or 'W_th' in short:
                gk = 'snn_block.W_mod'
            elif 'W_gate' in short or 'W_skip' in short:
                gk = 'snn_block.W_gate_skip'
            elif 'output_neuron' in short:
                gk = 'snn_block.output_neuron'
            else:
                gk = 'snn_block.other'
        elif 'snn_ffn' in short:
            if any(k in short for k in ['gate_neuron', 'up_neuron', 'output_neuron']):
                gk = 'snn_ffn.neurons'
            else:
                gk = 'snn_ffn.proj'
        elif 'block_out_proj' in short or 'ffn_out_proj' in short:
            gk = 'residual_proj'
        else:
            gk = short.split('.')[0]

        group_deltas[gk].append((ratio, d_abs_mean))

        # 只打印 L0, L10, L19 和顶层参数
        if li in [-1, 0, 10, 19]:
            tag = f"L{li:02d}." if li >= 0 else ""
            full = tag + short
            print(f"  {full:>53s} {d_abs_mean:>10.3e} {d_abs_max:>10.3e} "
                  f"{p_abs_mean:>10.3e} {ratio:>7.1%}")

    del model_init
    torch.cuda.empty_cache()

    # 按组汇总
    print(f"\n--- 按参数组汇总变化率 ---")
    print(f"{'参数组':>30s} {'avg 变化率':>12s} {'min':>10s} {'max':>10s}")
    print("-" * 65)
    for gk in sorted(group_deltas.keys()):
        ratios = [r for r, _ in group_deltas[gk]]
        avg_r = sum(ratios) / len(ratios)
        print(f"  {gk:>28s} {avg_r:>11.2%} {min(ratios):>9.2%} {max(ratios):>9.2%}")

    # 逐层神经元参数实际值
    print(f"\n--- 神经元参数实际值（5 个代表层）---")
    for i in [0, 5, 10, 15, 19]:
        layer = model.layers[i]
        blk = layer.snn_block

        b1 = torch.sigmoid(layer.input_neuron1.w)
        b2 = torch.sigmoid(layer.input_neuron2.w)
        bh = torch.sigmoid(blk.b_beta)
        vth_h = blk.v_th_min + torch.abs(blk.b_th)

        print(f"  L{i:2d}: in1_β={b1.mean():.6f}±{b1.std():.6f}  "
              f"in1_vth={layer.input_neuron1.v_th.mean():.6f}  "
              f"hidden_β=[{bh.min():.4f},{bh.max():.4f}]  "
              f"hidden_Vth=[{vth_h.min():.4f},{vth_h.max():.4f}]")

    # ==================================================================
    # Phase 2: 前向激活诊断 (monkey-patch + no_grad)
    # ==================================================================
    print("\n" + "=" * 80)
    print("[2] 前向激活诊断（monkey-patch 采集，不复制前向逻辑）")
    print("=" * 80)

    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    model.eval()

    # ---- 安装 monkey-patch ----
    layer_stats = {}
    saved_originals = []

    for i, layer_mod in enumerate(model.layers):
        st = {}
        layer_stats[i] = st

        orig_fwd = layer_mod.forward_parallel
        orig_inp = layer_mod._input_neuron_parallel
        orig_blk = layer_mod.snn_block.forward_parallel
        orig_ffn = layer_mod.snn_ffn.forward_parallel

        saved_originals.append((layer_mod, orig_fwd, orig_inp, orig_blk, orig_ffn))

        # 用工厂函数创建闭包，避免 late-binding 问题
        def _make(of, oi, ob, off, s, lm):
            def wrap_layer(h):
                s['h_in_mean'] = h.float().mean().item()
                s['h_in_std'] = h.float().std().item()
                h_out = of(h)
                s['h_out_mean'] = h_out.float().mean().item()
                s['h_out_std'] = h_out.float().std().item()
                s['delta_std'] = (h_out - h).float().std().item()
                return h_out

            def wrap_inp(neuron, x):
                spike = oi(neuron, x)
                pfx = 'in1' if neuron is lm.input_neuron1 else 'in2'
                s[f'{pfx}_fire'] = spike.float().mean().item()
                s[f'{pfx}_row0'] = (spike.sum(-1) == 0).float().mean().item()
                return spike

            def wrap_blk(sp_in):
                sp_out = ob(sp_in)
                s['blk_fire'] = sp_out.float().mean().item()
                return sp_out

            def wrap_ffn(sp_in):
                sp_out = off(sp_in)
                s['ffn_fire'] = sp_out.float().mean().item()
                return sp_out

            return wrap_layer, wrap_inp, wrap_blk, wrap_ffn

        wl, wi, wb, wf = _make(orig_fwd, orig_inp, orig_blk, orig_ffn, st, layer_mod)
        layer_mod.forward_parallel = wl
        layer_mod._input_neuron_parallel = wi
        layer_mod.snn_block.forward_parallel = wb
        layer_mod.snn_ffn.forward_parallel = wf

    # ---- 执行推理 ----
    with torch.no_grad():
        output = model(input_ids)

    # ---- 恢复原方法 ----
    for layer_mod, of, oi, ob, off in saved_originals:
        layer_mod.forward_parallel = of
        layer_mod._input_neuron_parallel = oi
        layer_mod.snn_block.forward_parallel = ob
        layer_mod.snn_ffn.forward_parallel = off

    # ---- 打印激活表 ----
    print(f"\n{'L':>3s} {'h_in_std':>9s} {'h_out_std':>9s} {'Δ_std':>9s} "
          f"{'in1_fire':>9s} {'in2_fire':>9s} {'blk_fire':>9s} {'ffn_fire':>9s} "
          f"{'in1_0row':>9s} {'in2_0row':>9s}")
    print("-" * 100)

    for i in range(num_layers):
        s = layer_stats[i]
        print(f" {i:2d} "
              f"{s.get('h_in_std',0):>9.4f} {s.get('h_out_std',0):>9.4f} "
              f"{s.get('delta_std',0):>9.4f} "
              f"{s.get('in1_fire',0):>9.4f} {s.get('in2_fire',0):>9.4f} "
              f"{s.get('blk_fire',0):>9.4f} {s.get('ffn_fire',0):>9.4f} "
              f"{s.get('in1_row0',0):>9.4f} {s.get('in2_row0',0):>9.4f}")

    # 趋势分析
    print(f"\n--- 趋势分析 ---")
    first = layer_stats[0]
    last = layer_stats[num_layers - 1]
    print(f"  残差流 std: L0={first['h_in_std']:.4f} → L{num_layers-1}={last['h_out_std']:.4f} "
          f"(增长 {last['h_out_std']/first['h_in_std']:.2f}x)")
    print(f"  in1 发放率: L0={first.get('in1_fire',0):.4f} → L{num_layers-1}={last.get('in1_fire',0):.4f}")
    print(f"  blk 发放率: L0={first.get('blk_fire',0):.4f} → L{num_layers-1}={last.get('blk_fire',0):.4f}")
    print(f"  ffn 发放率: L0={first.get('ffn_fire',0):.4f} → L{num_layers-1}={last.get('ffn_fire',0):.4f}")

    # ==================================================================
    # Phase 3: 反向传播梯度分析
    # ==================================================================
    print("\n" + "=" * 80)
    print("[3] 反向传播梯度分析")
    print("=" * 80)

    model.train()
    labels = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    model.zero_grad()
    for lm in model.layers:
        functional.reset_net(lm)

    out3 = model(input_ids, labels)
    loss = out3.last_loss.mean()
    loss.backward()

    print(f"\n  Loss = {loss.item():.4f}")

    # 按层统计梯度
    per_layer_norms = defaultdict(list)
    print(f"\n{'参数':>50s} {'grad_norm':>12s} {'grad_std':>12s}")
    print("-" * 78)

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.float()
        gnorm = g.norm().item()

        if 'layers.' in name:
            parts = name.split('.')
            li = int(parts[1])
            short = '.'.join(parts[2:])
            per_layer_norms[li].append(gnorm)
        else:
            li = -1
            short = name

        # 只打印关键参数
        if li in [-1, 0, 10, 19]:
            if li == -1 or any(k in short for k in [
                'W_in.weight', 'b_beta', 'b_alpha', 'b_th',
                'input_neuron1.w', 'input_neuron1.v_th',
                'block_out_proj.weight', 'output_neuron.w',
                'gate_proj.weight', 'down_proj.weight',
                'ffn_out_proj.weight',
            ]):
                tag = f"L{li:02d}." if li >= 0 else ""
                print(f"  {tag + short:>48s} {gnorm:>12.4e} {g.std().item():>12.4e}")

    # 逐层汇总
    print(f"\n{'Layer':>5s} {'avg_norm':>12s} {'min_norm':>12s} {'max_norm':>12s} {'#params':>8s}")
    print("-" * 50)
    for li in sorted(per_layer_norms.keys()):
        ns = per_layer_norms[li]
        print(f"  L{li:2d} {sum(ns)/len(ns):>12.4e} {min(ns):>12.4e} {max(ns):>12.4e} {len(ns):>8d}")

    # 梯度流比率
    if 0 in per_layer_norms and (num_layers - 1) in per_layer_norms:
        avg0 = sum(per_layer_norms[0]) / len(per_layer_norms[0])
        avgL = sum(per_layer_norms[num_layers - 1]) / len(per_layer_norms[num_layers - 1])
        print(f"\n  L0/L{num_layers-1} avg grad norm ratio: {avg0 / (avgL + 1e-30):.4f}")

    # 无梯度参数检查
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    zero_grad = [n for n, p in model.named_parameters()
                 if p.grad is not None and p.grad.float().norm().item() == 0]
    print(f"\n  grad=None: {len(no_grad)}, grad=0: {len(zero_grad)}")
    if no_grad:
        for n in no_grad[:5]:
            print(f"    None: {n}")
    if zero_grad:
        for n in zero_grad[:5]:
            print(f"    Zero: {n}")

    # ==================================================================
    # Phase 4: Logits 分布
    # ==================================================================
    print("\n" + "=" * 80)
    print("[4] Logits 分布")
    print("=" * 80)

    logits = output.logits.float()  # Phase 2 的输出
    print(f"  shape: {list(logits.shape)}")
    print(f"  mean={logits.mean():.4e}  std={logits.std():.4e}  "
          f"min={logits.min():.4e}  max={logits.max():.4e}")

    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)
    max_ent = math.log(vocab_size)
    print(f"  entropy: mean={entropy.mean():.4f} / max={max_ent:.4f} "
          f"(ratio={entropy.mean() / max_ent:.4f})")
    print(f"  top-1 prob: {probs.max(-1).values.mean():.6f}")

    top5, _ = probs.topk(5, dim=-1)
    print(f"  top-5 avg probs: {[f'{top5[:,:,j].mean():.6f}' for j in range(5)]}")

    # 检查 logits 是否退化（所有 token 概率近乎均匀）
    uniform_prob = 1.0 / vocab_size
    top1_mean = probs.max(-1).values.mean().item()
    print(f"  uniform baseline: {uniform_prob:.6f}  |  top-1: {top1_mean:.6f}  "
          f"(比均匀高 {top1_mean / uniform_prob:.1f}x)")

    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
