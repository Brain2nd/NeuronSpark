"""
IG-ZO 训练脚本：信息论引导的零阶优化

数据集：WikiText-2
训练方法：SPSA 零阶梯度估计（IG-ZO 的基础版本）
评估指标：perplexity

用法：
    conda activate SNN
    python train.py
"""

import os
import time
import math
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from model import SNNLanguageModel


# ============================================================
# 数据集
# ============================================================

class WikiTextDataset(Dataset):
    """WikiText-2 token 序列数据集，用于自回归语言建模。"""

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        """
        Args:
            token_ids: 1D tensor，整个语料的 token ID 序列
            seq_len: 每个样本的序列长度
        """
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.token_ids[start:end]
        return chunk[:-1], chunk[1:]  # (input, target)


def load_wikitext2(tokenizer_name: str, seq_len: int):
    """加载 WikiText-2 数据集并 tokenize。"""
    print("Loading WikiText-2...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_split(split_name):
        texts = dataset[split_name]['text']
        # 合并所有文本为一个长字符串
        full_text = '\n'.join([t for t in texts if t.strip()])
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        return torch.tensor(token_ids, dtype=torch.long)

    train_ids = tokenize_split('train')
    val_ids = tokenize_split('validation')
    test_ids = tokenize_split('test')

    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")
    print(f"  Test tokens:  {len(test_ids):,}")
    print(f"  Vocab size:   {tokenizer.vocab_size:,}")

    train_ds = WikiTextDataset(train_ids, seq_len)
    val_ds = WikiTextDataset(val_ids, seq_len)
    test_ds = WikiTextDataset(test_ids, seq_len)

    return train_ds, val_ds, test_ds, tokenizer.vocab_size


# ============================================================
# SPSA 零阶优化器
# ============================================================

class SPSAOptimizer:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) 零阶优化器。

    这是 IG-ZO 的基础版本。后续可扩展为：
    - Fisher 加权扰动
    - 诊断引导的定向扰动
    - 多时间尺度学习率

    稳定性机制：
    - 梯度裁剪：限制单步更新幅度，防止发散
    - Cosine 学习率衰减：训练后期减小步长
    - 扰动幅度衰减：后期更精细的梯度估计
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 1e-3,
        perturbation_scale: float = 1e-3,
        total_steps: int = 10000,
        lr_min_ratio: float = 0.1,
        grad_clip: float = 1.0,
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
        # lr: 从 lr_init 衰减到 lr_init * lr_min_ratio
        self.lr = self.lr_init * (self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine_decay)
        # c: 从 c_init 衰减到 c_init * 0.5（后期更精细估计）
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
        """
        SPSA 梯度估计 + 参数更新（带裁剪和衰减）。

        ĝ_i = (L+ - L-) / (2c · δ_i)
        θ_i -= lr · clip(ĝ_i)
        """
        self._update_schedule()
        grad_scale = (loss_plus - loss_minus) / (2.0 * self.c)
        # 梯度裁剪：限制 |grad_scale| 防止单步过大更新
        grad_scale = max(min(grad_scale, self.grad_clip), -self.grad_clip)
        with torch.no_grad():
            for p, d in zip(self.params, deltas):
                # ĝ_i = grad_scale · δ_i（因为 δ_i ∈ {±1}）
                p.add_(d, alpha=-self.lr * grad_scale)
        self.current_step += 1


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(model: SNNLanguageModel, dataloader: DataLoader, device: torch.device):
    """计算验证集 perplexity。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # (batch, seq_len, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()
        total_tokens += target_ids.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100))  # 限制防溢出
    return avg_loss, ppl


# ============================================================
# 训练循环
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    train_ds, val_ds, test_ds, vocab_size = load_wikitext2(
        args.tokenizer, args.seq_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=0)

    # 构建模型
    model = SNNLanguageModel(
        vocab_size=vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_blocks=args.num_blocks,
        v_th_min=args.v_th_min,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: D={args.D}, N={args.N}, K={args.K}, Blocks={args.num_blocks}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # 加载 checkpoint（如果指定）
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))
        val_loss, val_ppl = evaluate(model, val_loader, device)
        print(f"  Resumed model Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f}")

    # 优化器
    total_steps = args.epochs * len(train_loader)
    optimizer = SPSAOptimizer(
        params=list(model.parameters()),
        lr=args.lr,
        perturbation_scale=args.perturbation_scale,
        total_steps=total_steps,
        lr_min_ratio=0.1,
        grad_clip=args.grad_clip,
    )

    criterion = nn.CrossEntropyLoss()

    # 训练
    print(f"\n{'='*60}")
    print(f"Training with SPSA (IG-ZO base)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  LR:         {args.lr} (cosine → {args.lr * 0.1})")
    print(f"  Perturb:    {args.perturbation_scale}")
    print(f"  Grad clip:  {args.grad_clip}")
    print(f"  Total steps:{total_steps}")
    print(f"{'='*60}\n")

    best_val_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t_start = time.time()

        for step, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 1. 生成扰动
            deltas = optimizer.generate_perturbation()

            # 2. θ + c·δ → L+
            optimizer.perturb(deltas, sign=+1.0)
            logits_plus = model(input_ids)
            loss_plus = criterion(
                logits_plus.view(-1, logits_plus.size(-1)),
                target_ids.view(-1),
            ).item()

            # 3. θ - 2c·δ（从 θ+cδ 回退到 θ-cδ）→ L-
            optimizer.perturb(deltas, sign=-2.0)
            logits_minus = model(input_ids)
            loss_minus = criterion(
                logits_minus.view(-1, logits_minus.size(-1)),
                target_ids.view(-1),
            ).item()

            # 4. 恢复到 θ（从 θ-cδ 前进 +cδ）
            optimizer.perturb(deltas, sign=+1.0)

            # 5. SPSA 更新
            optimizer.step(loss_plus, loss_minus, deltas)

            # 记录（用 L+ 和 L- 的均值近似当前 loss）
            batch_loss = (loss_plus + loss_minus) / 2.0
            batch_tokens = target_ids.numel()
            epoch_loss += batch_loss * batch_tokens
            epoch_tokens += batch_tokens

            if (step + 1) % args.log_interval == 0:
                avg = epoch_loss / epoch_tokens
                ppl = math.exp(min(avg, 100))
                elapsed = time.time() - t_start
                print(
                    f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                    f"LR {optimizer.lr:.2e} | Time {elapsed:.1f}s"
                )

        # Epoch 结束：验证
        val_loss, val_ppl = evaluate(model, val_loader, device)
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch} done | "
            f"Train Loss {epoch_loss/epoch_tokens:.4f} | "
            f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f} | "
            f"Time {elapsed:.1f}s"
        )

        # 保存最优
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (PPL {val_ppl:.1f})")

    print(f"\nBest Val PPL: {best_val_ppl:.1f}")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SNN Language Model Training (IG-ZO)')

    # 模型参数
    parser.add_argument('--D', type=int, default=128, help='可见维度')
    parser.add_argument('--N', type=int, default=8, help='状态扩展因子')
    parser.add_argument('--K', type=int, default=8, help='每 token SNN 时间步数')
    parser.add_argument('--num_blocks', type=int, default=2, help='SNN Block 层数')
    parser.add_argument('--v_th_min', type=float, default=0.1, help='阈值下限')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=64, help='序列长度（token 数）')
    parser.add_argument('--lr', type=float, default=1e-3, help='SPSA 学习率')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3, help='SPSA 扰动幅度')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='SPSA 梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=10, help='打印间隔（步）')

    # 数据
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='HuggingFace tokenizer')

    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复训练')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
