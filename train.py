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

def save_checkpoint(path, model, optimizer, scaler, step, epoch, best_loss, tokens_seen):
    """保存训练状态（扩展教程，额外保存 optimizer/scaler 以支持断续训练）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'tokens_seen': tokens_seen,
        'model_config': {
            'vocab_size': model.vocab_size if not isinstance(model, torch.nn.DataParallel) else model.module.vocab_size,
            'D': model.D if not isinstance(model, torch.nn.DataParallel) else model.module.D,
            'N': model.N if not isinstance(model, torch.nn.DataParallel) else model.module.N,
            'K': model.K if not isinstance(model, torch.nn.DataParallel) else model.module.K,
            'num_layers': model.num_layers if not isinstance(model, torch.nn.DataParallel) else model.module.num_layers,
            'D_ff': model.D_ff if not isinstance(model, torch.nn.DataParallel) else model.module.D_ff,
        },
    }, path)
    Logger(f"  → Checkpoint saved: {path} (step {step})")


def load_checkpoint(path, model, optimizer, scaler, device):
    """加载 checkpoint，恢复训练状态。"""
    Logger(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
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

        # 计算当前步骤的学习率（对齐教程 L93-96）
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播 + 损失计算（对齐教程 L99-107）
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播（对齐教程 L110）
        scaler.scale(loss).backward()

        # 梯度累积：每 accumulation_steps 步更新一次（对齐教程 L113-125）
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 清理
        if step % 10 == 0 and args.device != 'cpu':
            torch.cuda.empty_cache()

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
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} epoch_Time:{}min{}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    batch_loss,
                    batch_ppl,
                    lr,
                    tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_str))

        # 定期保存（对齐教程 L149-157）
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/pretrain_{args.D}_{args.num_layers}_{args.vocab_size}.pth'
            save_checkpoint(ckp, model, optimizer, scaler,
                            epoch * iter_per_epoch + step + 1, epoch, batch_loss, tokens_seen)
            model.train()

        # 每 20000 步保存带步数标记的检查点（对齐教程 L160-168）
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/pretrain_{args.D}_{args.num_layers}_{args.vocab_size}_step{step+1}.pth'
            save_checkpoint(ckp, model, optimizer, scaler,
                            epoch * iter_per_epoch + step + 1, epoch, batch_loss, tokens_seen)
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
    parser.add_argument('--K', type=int, default=16, help='每 token SNN 时间步数')
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
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype in ['float16', 'bfloat16']))

    # Adam 优化器（对齐教程 L315）
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

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
    save_checkpoint(
        os.path.join(args.save_dir, f'pretrain_{args.D}_{args.num_layers}_{args.vocab_size}_final.pth'),
        model, optimizer, scaler, args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen,
    )
