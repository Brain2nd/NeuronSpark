"""
训练脚本 v7.1：SNN 语言模型预训练（Surrogate Gradient + 反向传播）

严格对齐 happy-llm 教程（/home/dgxspark/Desktop/happy-llm/docs/chapter5/code/ddp_pretrain.py），
训练基础设施照搬教程，只有模型架构是 SNN 创新。

数据加载（对齐教程）：
- PretrainDataset: byte-offset 随机访问 JSONL，bos_token 前缀，固定 max_length
- loss_mask: 忽略 padding 位置的 loss
- Loss 计算: out.last_loss * loss_mask（与教程完全一致）

训练算法（v7.1: 反向传播，替代 v5 的 SPSA 零阶优化）：
- Adam 优化器（对齐教程）
- Warmup + Cosine LR 调度（对齐教程 get_lr()）
- GradScaler + autocast 混合精度（对齐教程）
- 梯度累积 accumulation_steps（对齐教程）
- 梯度裁剪 clip_grad_norm_（对齐教程）

用法：
    conda activate SNN

    python train.py \
        --data_path data/seq-monkey/seq_monkey_datawhale.jsonl \
        --batch_size 8 --accumulation_steps 8

    # 断续训练
    python train.py --resume checkpoints/latest.pt
"""

import os
import glob
import time
import math
import argparse
import warnings

import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import PretrainDataset

# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """简单的日志记录函数"""
    print(content)


# ============================================================
# 学习率调度（照搬教程 get_lr）
# ============================================================

def get_lr(it, all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略（对齐教程）。

    学习率调度策略：
    1. Warmup 阶段：学习率从 0 线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率

    Args:
        it: 当前迭代步数
        all: 总迭代步数
    """
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10  # 最小学习率 = 初始学习率的 1/10

    # Warmup 阶段：线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters

    # 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr

    # 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, best_loss, tokens_seen,
                    max_keep=5):
    """保存训练状态，每次不覆盖（带步数），仅保留最新 max_keep 个。"""
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')
    torch.save({
        'model_state_dict': raw.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'tokens_seen': tokens_seen,
        'model_config': {
            'vocab_size': raw.vocab_size,
            'D': raw.D,
            'N': raw.N,
            'K': raw.K,
            'num_layers': raw.num_layers,
            'D_ff': raw.D_ff,
        },
    }, path)
    Logger(f"  → Checkpoint saved: {path}")

    # 清理旧 checkpoint，仅保留最新 max_keep 个
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        Logger(f"  → Removed old checkpoint: {old}")


def load_checkpoint(path, model, optimizer, scaler, device):
    """加载 checkpoint，恢复训练状态。"""
    Logger(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    if 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except (ValueError, KeyError):
            Logger("  Warning: Optimizer state incompatible (SPSA checkpoint?), starting fresh.")

    if 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    tokens_seen = ckpt.get('tokens_seen', 0)
    Logger(f"  Resumed: step={step}, epoch={epoch}, tokens={tokens_seen:,}")
    return step, epoch, best_loss, tokens_seen


# ============================================================
# 初始化
# ============================================================

def init_model(args):
    """
    初始化模型和分词器（对齐教程 init_model）。

    Returns:
        tuple: (model, tokenizer)
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载自训练 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 创建 SNN 语言模型
    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    )

    # 将模型移动到指定设备
    model = model.to(args.device)

    Logger(f'SNN LM 总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


# ============================================================
# 训练循环（对齐教程 train_epoch）
# ============================================================

def train_epoch(epoch, model, train_loader, optimizer, scaler, ctx, args, iter_per_epoch, tokens_seen):
    """
    训练一个 epoch（对齐教程训练循环，使用标准反向传播）。

    对齐教程 ddp_pretrain.py L86-168 的完整训练循环：
    1. 动态学习率（get_lr warmup + cosine）
    2. 前向传播（autocast 混合精度）
    3. 反向传播（scaler.scale(loss).backward()）
    4. 梯度累积（每 accumulation_steps 步更新一次）
    5. 梯度裁剪（clip_grad_norm_）
    6. 日志记录（额外加 PPL、TPS、显存）
    """
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备（对齐教程 L88-90）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前步骤的学习率（对齐教程 L93-96，扩展：按 lr_mult 分组缩放）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # 前向传播 + 损失计算（对齐教程 L99-107）
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()
            # 动态 K: ponder cost 正则化（鼓励用更少步数处理简单 token）
            if out.ponder_cost is not None and args.ponder_weight > 0:
                loss = loss + args.ponder_weight * out.ponder_cost / args.accumulation_steps

        # 反向传播（对齐教程 L110）
        scaler.scale(loss).backward()

        # 梯度累积：每 accumulation_steps 步更新一次（对齐教程 L113-125）
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # Natural Gradient: 补偿 b_beta/b_alpha 的 sigmoid/softplus 梯度衰减
            model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 有效 token 数
        valid_tokens = int(loss_mask_flat.sum().item())
        tokens_seen += valid_tokens

        # 日志（对齐教程 L128-146，额外加 PPL、TPS、显存监控）
        if step % args.log_interval == 0:
            batch_loss = loss.item() * args.accumulation_steps  # 恢复真实 loss
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_str = ""
            if args.device != 'cpu':
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem {mem_cur:.1f}/{mem_peak:.1f}GB"
            # 动态 K: 报告期望步数（E[K] 均值 + per-token 范围 [min~max]）
            ponder_str = ""
            if out.ponder_cost is not None:
                ek_mean = out.ponder_cost.item()
                # 从层属性读取 per-token E[K] 范围
                raw_model = model.module if hasattr(model, 'module') else model
                ek_mins = [l._ek_min for l in raw_model.layers if hasattr(l, '_ek_min')]
                ek_maxs = [l._ek_max for l in raw_model.layers if hasattr(l, '_ek_max')]
                if ek_mins:
                    ponder_str = f" | E[K]:{ek_mean:.1f} [{min(ek_mins):.1f}~{max(ek_maxs):.1f}]"
                else:
                    ponder_str = f" | E[K]:{ek_mean:.1f}"
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} epoch_Time:{}min{}{}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    batch_loss,
                    batch_ppl,
                    lr,
                    tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_str,
                    ponder_str))

        # 定期保存（带步数，自动清理保留最新 5 个）
        if (step + 1) % args.save_interval == 0:
            model.eval()
            global_step = epoch * iter_per_epoch + step + 1
            save_checkpoint(args.save_dir, model, optimizer, scaler,
                            global_step, epoch, batch_loss, tokens_seen)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining v7.1 (Backprop)")

    # SNN 模型参数（创新部分）
    parser.add_argument('--vocab_size', type=int, default=6144, help='词表大小')
    parser.add_argument('--D', type=int, default=1024, help='隐层维度')
    parser.add_argument('--N', type=int, default=8, help='状态扩展因子')
    parser.add_argument('--K', type=int, default=32, help='每 token 最大 SNN 时间步数（K_max），PonderNet 动态决定有效步数')
    parser.add_argument('--num_layers', type=int, default=20, help='SNN 解码层数')
    parser.add_argument('--D_ff', type=int, default=3072, help='FFN 中间层维度')
    parser.add_argument('--v_th_min', type=float, default=0.1, help='阈值下限')

    # 基础训练参数（对齐教程）
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小（反向传播需更多显存，默认 8）")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型（对齐教程）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")

    # 训练优化参数（对齐教程）
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学习率（对齐教程）')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='梯度累积步数（对齐教程）')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值（对齐教程）')
    parser.add_argument('--warmup_iters', type=int, default=0, help='学习率预热迭代次数（对齐教程）')
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0, help='神经元参数学习率倍率（相对 base lr）')
    parser.add_argument('--ponder_weight', type=float, default=0.01, help='动态 K ponder cost 正则化权重')

    # 日志和保存参数（对齐教程）
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 数据（对齐教程）
    parser.add_argument("--data_path", type=str,
                        default="data/seq-monkey/seq_monkey_datawhale.jsonl",
                        help="预处理后的 JSONL 数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/",
                        help="自训练 tokenizer 路径")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复')

    args = parser.parse_args()

    # ==================== 训练环境设置 ====================
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(42)

    # 混合精度上下文（对齐教程 L286-290）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # ==================== 模型和数据初始化 ====================
    model, tokenizer = init_model(args)

    # 创建训练数据集（对齐教程 PretrainDataset）
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 创建数据加载器（对齐教程 L300-307）
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器和训练组件初始化 ====================
    # GradScaler（对齐教程 L312）
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # Adam 优化器（分组学习率：神经元参数 neuron_lr_mult × base_lr）
    # 神经元参数（PLIFNode w/v_th, 调制偏置 b_beta/b_alpha/b_th）梯度天然较弱
    # （surrogate sigmoid 窗口窄），需要更高 lr 才能跟上权重矩阵的漂移速度。
    _pg = model.get_param_groups()
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult)},
    ])

    # 恢复 checkpoint
    tokens_seen = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, best_loss, tokens_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, args.device,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)

    Logger(f"\n{'='*60}")
    Logger(f"SNN Language Model Pretraining v7.1 (Backprop + Surrogate Gradient)")
    Logger(f"  Vocab:       {args.vocab_size}")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}")
    Logger(f"  Data:        {args.data_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Max length:  {args.max_length}")
    Logger(f"  Batch size:  {args.batch_size} × accum {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} effective")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})")
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Precision:   {args.dtype}")
    Logger(f"  Save every:  {args.save_interval} steps")
    if args.device != 'cpu':
        mem_baseline = torch.cuda.memory_allocated() / 1e9
        Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline")
    Logger(f"{'='*60}\n")

    # ==================== 开始训练 ====================
    for epoch in range(start_epoch, args.epochs):
        tokens_seen = train_epoch(epoch, model, train_loader, optimizer, scaler, ctx, args, iter_per_epoch, tokens_seen)

    # 训练结束，保存最终 checkpoint
    Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}")
    if args.device != 'cpu':
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    save_checkpoint(args.save_dir, model, optimizer, scaler,
                    args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen)
