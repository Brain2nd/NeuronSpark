"""
训练脚本 v5：从零预训练 SNN 语言模型

数据加载（对齐 happy-llm 教程）：
- PretrainDataset: byte-offset 随机访问 JSONL，bos_token 前缀，固定 max_length
- loss_mask: 忽略 padding 位置的 loss
- Loss 计算: out.last_loss * loss_mask（与教程 ddp_pretrain.py 完全一致）

训练算法（SNN 适配）：
- SPSA 零阶优化（SNN 不走 backward，用 forward 差分估计梯度）
- Cosine 学习率衰减

用法：
    conda activate SNN

    python train.py \
        --data_path data/seq-monkey/seq_monkey_datawhale.jsonl \
        --max_steps 100000

    # 断续训练
    python train.py --resume checkpoints/latest.pt
"""

import os
import time
import math
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import PretrainDataset

# 忽略警告信息
warnings.filterwarnings('ignore')


def Logger(content):
    """简单的日志记录函数"""
    print(content)


# ============================================================
# SPSA 零阶优化器
# ============================================================

class SPSAOptimizer:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) 零阶优化器。

    稳定性机制：
    - 梯度裁剪：限制单步更新幅度
    - Cosine 学习率衰减
    - 扰动幅度衰减
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 5e-4,
        perturbation_scale: float = 1e-3,
        total_steps: int = 100000,
        lr_min_ratio: float = 0.1,
        grad_clip: float = 0.5,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr_init = lr
        self.lr = lr
        self.c_init = perturbation_scale
        self.c = perturbation_scale
        self.total_steps = total_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip = grad_clip
        self.current_step = 0

    def _update_schedule(self):
        """Cosine 衰减学习率和扰动幅度。"""
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.lr = self.lr_init * (self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine_decay)
        self.c = self.c_init * (0.5 + 0.5 * cosine_decay)

    def generate_perturbation(self) -> list[torch.Tensor]:
        """生成 Rademacher 随机扰动（±1）。"""
        deltas = []
        for p in self.params:
            delta = torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2 - 1
            deltas.append(delta)
        return deltas

    def perturb(self, deltas: list[torch.Tensor], sign: float = 1.0):
        """对参数施加扰动：θ ± c·δ。"""
        with torch.no_grad():
            for p, d in zip(self.params, deltas):
                p.add_(d, alpha=sign * self.c)

    def step(self, loss_plus: float, loss_minus: float, deltas: list[torch.Tensor]):
        """SPSA 梯度估计 + 参数更新。"""
        self._update_schedule()
        grad_scale = (loss_plus - loss_minus) / (2.0 * self.c)
        grad_scale = max(min(grad_scale, self.grad_clip), -self.grad_clip)
        with torch.no_grad():
            for p, d in zip(self.params, deltas):
                p.add_(d, alpha=-self.lr * grad_scale)
        self.current_step += 1

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'lr_init': self.lr_init,
            'c_init': self.c_init,
            'total_steps': self.total_steps,
            'lr_min_ratio': self.lr_min_ratio,
            'grad_clip': self.grad_clip,
        }

    def load_state_dict(self, state):
        self.current_step = state['current_step']
        self.lr_init = state['lr_init']
        self.c_init = state['c_init']
        self.total_steps = state['total_steps']
        self.lr_min_ratio = state['lr_min_ratio']
        self.grad_clip = state['grad_clip']
        self._update_schedule()


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(path, model, optimizer, step, best_loss, tokens_seen):
    """保存训练状态。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'best_loss': best_loss,
        'tokens_seen': tokens_seen,
        'model_config': {
            'vocab_size': model.vocab_size if not isinstance(model, torch.nn.DataParallel) else model.module.vocab_size,
            'D': model.D if not isinstance(model, torch.nn.DataParallel) else model.module.D,
            'N': model.N if not isinstance(model, torch.nn.DataParallel) else model.module.N,
            'K': model.K if not isinstance(model, torch.nn.DataParallel) else model.module.K,
            'num_blocks': model.num_blocks if not isinstance(model, torch.nn.DataParallel) else model.module.num_blocks,
        },
    }, path)
    Logger(f"  → Checkpoint saved: {path} (step {step})")


def load_checkpoint(path, model, optimizer, device):
    """加载 checkpoint，恢复训练状态。"""
    Logger(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'trainable_state_dict' in ckpt:
        model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    optimizer.load_state_dict(ckpt['optimizer_state'])
    step = ckpt['step']
    best_loss = ckpt.get('best_loss', float('inf'))
    tokens_seen = ckpt.get('tokens_seen', 0)
    Logger(f"  Resumed: step={step}, tokens={tokens_seen:,}")
    return step, best_loss, tokens_seen


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
        num_blocks=args.num_blocks,
        v_th_min=args.v_th_min,
    )

    # 将模型移动到指定设备
    model = model.to(args.device)

    Logger(f'SNN LM 总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, optimizer, args, iter_per_epoch, tokens_seen):
    """
    训练一个 epoch（SPSA 零阶优化，对齐教程训练循环结构）。

    SPSA 与教程 Adam 的区别：
    - 不走 backward，用两次 forward 差分估计梯度
    - 不用混合精度 / gradient scaler
    - 不用梯度累积（每步独立）
    """
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 有效 token 数
        loss_mask_flat = loss_mask.view(-1)
        valid_tokens = loss_mask_flat.sum()
        if valid_tokens == 0:
            continue

        # 1. 生成扰动
        deltas = optimizer.generate_perturbation()

        # 2. θ + c·δ → L+
        optimizer.perturb(deltas, sign=+1.0)
        with torch.no_grad():
            out = model(X, Y)
            loss_plus = (torch.sum(out.last_loss * loss_mask_flat) / valid_tokens).item()

        # 3. θ - 2c·δ → L-（从 +c 到 -c 需要 -2c）
        optimizer.perturb(deltas, sign=-2.0)
        with torch.no_grad():
            out = model(X, Y)
            loss_minus = (torch.sum(out.last_loss * loss_mask_flat) / valid_tokens).item()

        # 4. 恢复到 θ（从 -c 回到 0 需要 +c）
        optimizer.perturb(deltas, sign=+1.0)

        # 5. SPSA 更新
        optimizer.step(loss_plus, loss_minus, deltas)

        # 6. 清理
        del deltas, out
        if step % 10 == 0 and args.device != 'cpu':
            torch.cuda.empty_cache()

        # 记录
        batch_loss = (loss_plus + loss_minus) / 2.0
        batch_valid_tokens = int(valid_tokens.item())
        tokens_seen += batch_valid_tokens

        # 日志（对齐教程 log_interval）
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_str = ""
            if args.device != 'cpu':
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem {mem_cur:.1f}/{mem_peak:.1f}GB"
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} TPS:{:.0f} epoch_Time:{}min{}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    batch_loss,
                    optimizer.lr,
                    tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_str))

        # 定期保存
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/pretrain_{args.D}_{args.num_blocks}_{args.vocab_size}.pth'
            save_checkpoint(ckp, model, optimizer, epoch * iter_per_epoch + step + 1, batch_loss, tokens_seen)
            model.train()

        # 每 20000 步保存带步数标记的检查点
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/pretrain_{args.D}_{args.num_blocks}_{args.vocab_size}_step{step+1}.pth'
            save_checkpoint(ckp, model, optimizer, epoch * iter_per_epoch + step + 1, batch_loss, tokens_seen)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining v5")

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144, help='词表大小')
    parser.add_argument('--D', type=int, default=1024, help='隐层维度')
    parser.add_argument('--N', type=int, default=8, help='状态扩展因子')
    parser.add_argument('--K', type=int, default=8, help='每 token SNN 时间步数')
    parser.add_argument('--num_blocks', type=int, default=10, help='SNN Block 层数')
    parser.add_argument('--v_th_min', type=float, default=0.1, help='阈值下限')

    # 基础训练参数（对齐教程）
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")

    # SPSA 训练参数
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='SPSA 学习率')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3, help='SPSA 扰动幅度')
    parser.add_argument('--grad_clip', type=float, default=0.5, help='梯度裁剪阈值')
    parser.add_argument('--max_steps', type=int, default=100000, help='最大训练步数')

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

    # ==================== 模型和数据初始化 ====================
    model, tokenizer = init_model(args)

    # 创建训练数据集（对齐教程 PretrainDataset）
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 创建数据加载器（对齐教程）
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器初始化 ====================
    optimizer = SPSAOptimizer(
        params=list(model.parameters()),
        lr=args.learning_rate,
        perturbation_scale=args.perturbation_scale,
        total_steps=args.max_steps,
        lr_min_ratio=0.1,
        grad_clip=args.grad_clip,
    )

    # 恢复 checkpoint
    tokens_seen = 0
    if args.resume:
        start_step, best_loss, tokens_seen = load_checkpoint(
            args.resume, model, optimizer, args.device,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)

    Logger(f"\n{'='*60}")
    Logger(f"SNN Language Model Pretraining (SPSA)")
    Logger(f"  Vocab:       {args.vocab_size}")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Blocks={args.num_blocks}")
    Logger(f"  Data:        {args.data_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Max length:  {args.max_length}")
    Logger(f"  Batch size:  {args.batch_size}")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  LR:          {args.learning_rate} (cosine → {args.learning_rate * 0.1})")
    Logger(f"  Perturb:     {args.perturbation_scale}")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Save every:  {args.save_interval} steps")
    if args.device != 'cpu':
        mem_baseline = torch.cuda.memory_allocated() / 1e9
        Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline")
    Logger(f"{'='*60}\n")

    # ==================== 开始训练 ====================
    for epoch in range(args.epochs):
        tokens_seen = train_epoch(epoch, model, train_loader, optimizer, args, iter_per_epoch, tokens_seen)

    # 训练结束，保存最终 checkpoint
    Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}")
    if args.device != 'cpu':
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    save_checkpoint(
        os.path.join(args.save_dir, f'pretrain_{args.D}_{args.num_blocks}_{args.vocab_size}_final.pth'),
        model, optimizer, args.epochs * iter_per_epoch, 0.0, tokens_seen,
    )
