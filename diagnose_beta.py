import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/workspace/NeuronSpark")

from model import SNNLanguageModel as SNNLM

def diagnose_beta_gradient():
    print("="*100)
    print("诊断 b_beta 为什么不更新")
    print("="*100)

    # 1. 加载模型和checkpoint
    print("\n【1. 加载模型】")
    model = SNNLM(vocab_size=6144, D=768, N=8, K=16, num_layers=20, D_ff=2304)
    ckpt = torch.load("checkpoints/ckpt_step500.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    # 2. 检查 b_beta 是否 requires_grad
    print("\n【2. 检查 b_beta requires_grad】")
    for name, param in model.named_parameters():
        if "b_beta" in name:
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
            break

    # 3. 做一次前向+反向传播，检查梯度
    print("\n【3. 前向+反向传播测试】")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 随机输入
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 6144, (batch_size, seq_len), device=device)

    # 前向
    model.zero_grad()
    output = model(input_ids)
    logits = output.logits if hasattr(output, 'logits') else output

    # 用真正的CE loss
    loss = nn.CrossEntropyLoss()(logits[:, :-1].reshape(-1, 6144), input_ids[:, 1:].reshape(-1))
    print(f"  Forward done, CE loss={loss.item():.4f}")

    # 反向
    loss.backward()
    print(f"  Backward done")

    # 4. 检查各参数的梯度
    print("\n【4. 各参数梯度统计】")
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            grad_stats.append({
                "name": name,
                "grad_mean": g.abs().mean().item(),
                "grad_max": g.abs().max().item(),
                "grad_nonzero": (g != 0).float().mean().item() * 100,
                "param_mean": param.data.abs().mean().item()
            })

    # 按梯度大小排序
    grad_stats.sort(key=lambda x: -x["grad_mean"])

    # 找 b_beta 相关的
    beta_stats = [s for s in grad_stats if "b_beta" in s["name"] or "b_alpha" in s["name"] or "b_th" in s["name"]]
    other_stats = [s for s in grad_stats if s not in beta_stats]

    print("\n--- b_beta/b_alpha/b_th 梯度 ---")
    print(f"{'参数名':<55} {'|grad|':<12} {'|param|':<12} {'grad/param':<12}")
    print("-"*95)
    for s in beta_stats[:15]:
        ratio = s["grad_mean"] / (s["param_mean"] + 1e-10)
        print(f"{s['name']:<55} {s['grad_mean']:<12.2e} {s['param_mean']:<12.4f} {ratio:<12.2e}")

    print("\n--- 其他参数梯度 (top 15) ---")
    for s in other_stats[:15]:
        ratio = s["grad_mean"] / (s["param_mean"] + 1e-10)
        print(f"{s['name']:<55} {s['grad_mean']:<12.2e} {s['param_mean']:<12.4f} {ratio:<12.2e}")

    # 5. 计算梯度比例
    print("\n【5. 梯度幅度对比】")
    if beta_stats and other_stats:
        beta_grad_avg = sum(s["grad_mean"] for s in beta_stats) / len(beta_stats)
        other_grad_avg = sum(s["grad_mean"] for s in other_stats) / len(other_stats)
        print(f"  b_beta/b_alpha/b_th 平均梯度: {beta_grad_avg:.2e}")
        print(f"  其他参数平均梯度: {other_grad_avg:.2e}")
        print(f"  比例: {beta_grad_avg/other_grad_avg:.4f}x")

    # 6. 检查 b_beta 梯度的来源 - 通过计算图分析
    print("\n【6. b_beta 梯度流分析】")

    # 找到第一层的 snn_block
    snn_block = model.layers[0].snn_block
    b_beta = snn_block.b_beta

    print(f"  Layer 0 b_beta: shape={b_beta.shape}, value_mean={b_beta.data.mean():.4f}")

    # 计算 sigmoid 导数在当前值的大小
    with torch.no_grad():
        beta = torch.sigmoid(b_beta)
        sigmoid_deriv = beta * (1 - beta)
    print(f"  beta = sigmoid(b_beta): mean={beta.mean():.4f}, range=[{beta.min():.4f}, {beta.max():.4f}]")
    print(f"  sigmoid导数 = beta*(1-beta): mean={sigmoid_deriv.mean():.4f}, min={sigmoid_deriv.min():.4f}, max={sigmoid_deriv.max():.4f}")
    print(f"  >>> sigmoid导数衰减倍数: {1.0/sigmoid_deriv.mean():.1f}x")

    # 7. 检查 W_beta_x 的输出对 beta 的影响
    print("\n【7. 动态调制分析】")
    print("  检查 W_beta_x 输出是否有效...")

    # hook W_beta_x 的输出
    w_beta_x_output = []
    def hook_fn(module, input, output):
        w_beta_x_output.append(output.detach())

    hook = snn_block.W_beta_x.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids)

    hook.remove()

    if w_beta_x_output:
        out = w_beta_x_output[0]
        print(f"  W_beta_x 输出: shape={out.shape}")
        print(f"  W_beta_x 输出: mean={out.mean():.4f}, std={out.std():.4f}, range=[{out.min():.4f}, {out.max():.4f}]")

        # 计算动态 beta
        b_beta_val = b_beta.data.unsqueeze(0).unsqueeze(0)
        beta_dynamic = torch.sigmoid(b_beta_val + out)
        beta_static = torch.sigmoid(b_beta_val)

        print(f"  静态 beta (只用b_beta): mean={beta_static.mean():.4f}")
        print(f"  动态 beta (b_beta + W_beta_x): mean={beta_dynamic.mean():.4f}, std={beta_dynamic.std():.4f}")
        print(f"  动态调制变化量: {(beta_dynamic - beta_static).abs().mean():.6f}")
        print(f"  >>> W_beta_x 输出相对于 b_beta 的比例: {out.abs().mean() / b_beta.data.abs().mean() * 100:.2f}%")

    # 8. 检查 loss 对 b_beta 的敏感度
    print("\n【8. Loss 对 b_beta 的敏感度分析】")

    model.zero_grad()
    original_b_beta = snn_block.b_beta.data.clone()

    # 原始 loss
    with torch.no_grad():
        out1 = model(input_ids)
        logits1 = out1.logits if hasattr(out1, 'logits') else out1
        loss1 = nn.CrossEntropyLoss()(logits1[:, :-1].reshape(-1, 6144), input_ids[:, 1:].reshape(-1))

    # 扰动 b_beta (所有层)
    with torch.no_grad():
        for layer in model.layers:
            layer.snn_block.b_beta.data -= 2.0  # 让 beta 从 ~0.9 降到 ~0.6

        out2 = model(input_ids)
        logits2 = out2.logits if hasattr(out2, 'logits') else out2
        loss2 = nn.CrossEntropyLoss()(logits2[:, :-1].reshape(-1, 6144), input_ids[:, 1:].reshape(-1))

    new_beta = torch.sigmoid(snn_block.b_beta.data)
    print(f"  原始 beta: mean={torch.sigmoid(original_b_beta).mean():.4f}")
    print(f"  扰动后 beta: mean={new_beta.mean():.4f}")
    print(f"  原始 loss: {loss1.item():.4f}")
    print(f"  扰动后 loss: {loss2.item():.4f}")
    print(f"  Loss 变化: {(loss2 - loss1).item():.4f} ({(loss2 - loss1).item() / loss1.item() * 100:.2f}%)")

    # 恢复
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            layer.snn_block.b_beta.data += 2.0

    # 9. 检查是否是 surrogate gradient 的问题
    print("\n【9. 脉冲发放率分析】")

    spike_rates = []
    def spike_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            # 假设输出是脉冲 (0/1)
            rate = (output > 0).float().mean().item()
            spike_rates.append(rate)

    hooks = []
    for name, module in model.named_modules():
        if "neuron" in name.lower() and hasattr(module, "forward"):
            h = module.register_forward_hook(spike_hook)
            hooks.append(h)

    with torch.no_grad():
        _ = model(input_ids)

    for h in hooks:
        h.remove()

    if spike_rates:
        print(f"  检测到 {len(spike_rates)} 个神经元层")
        print(f"  平均脉冲发放率: {sum(spike_rates)/len(spike_rates)*100:.2f}%")
        print(f"  发放率范围: [{min(spike_rates)*100:.2f}%, {max(spike_rates)*100:.2f}%]")

    print("\n" + "="*100)
    print("诊断总结")
    print("="*100)

if __name__ == "__main__":
    diagnose_beta_gradient()
