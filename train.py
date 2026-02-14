"""
训练脚本 v3：SPSA 零阶优化 + 内存优化

数据加载模式（自动选择）：
- 有 binary token 文件 → mmap 随机访问（零内存开销）
- 无 binary 文件 → 直接流式读 JSONL + 实时 tokenize

其他特性：
- SPSA 零阶优化（forward 直接返回 loss，无 logits 累积）
- 完整 checkpoint 断续训练（仅存可训练参数 + 优化器状态）
- Cosine 学习率衰减 + 梯度裁剪
- 内存监控：每 N 步打印 CUDA 内存使用

用法：
    conda activate SNN

    # 直接从 JSONL 训练（无需预处理）
    python train.py --data_dir data/SkyPile-150B --max_steps 100000

    # 或预处理后用 binary 文件训练
    python prepare_data.py --data_dir data/SkyPile-150B --output_dir data/processed
    python train.py --data_dir data/processed --max_steps 100000

    # 断续训练
    python train.py --resume checkpoints/latest.pt
"""

import os
import json
import glob
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from model import SNNLanguageModel


# ============================================================
# 数据集
# ============================================================

class TokenMmapDataset(Dataset):
    """
    基于 numpy memmap 的 token 数据集。

    从 flat binary 文件（uint32）中随机采样连续片段。
    内存开销 ≈ 0（OS page cache 按需加载）。
    """

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.n_tokens = len(self.data)
        # 每个样本需要 seq_len+1 个连续 token（input + target）
        self.n_samples = self.n_tokens - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        # np.uint32 → torch.long 需要 int64 中转，避免负数
        chunk = torch.from_numpy(chunk.astype(np.int64))
        return chunk[:-1], chunk[1:]


class TokenSequenceDataset(Dataset):
    """固定长度的 token 序列数据集（用于验证集，小数据量）。"""

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.n_tokens = len(self.data)

    def __len__(self):
        return (self.n_tokens - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        return chunk[:-1], chunk[1:]


class StreamingJsonlDataset(IterableDataset):
    """
    直接从本地 JSONL 文件流式 tokenize 的数据集。
    无需预处理，逐文件逐行读取，内存开销仅为 token_buffer。
    """

    def __init__(self, jsonl_dir: str, tokenizer, seq_len: int):
        self.jsonl_dir = jsonl_dir
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # 收集所有 JSONL 文件
        self.files = sorted(glob.glob(os.path.join(jsonl_dir, '**/*.jsonl'), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .jsonl files found in {jsonl_dir}")

    def __iter__(self):
        token_buffer = []
        for fpath in self.files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get('text', '')
                    except json.JSONDecodeError:
                        continue
                    if not text.strip():
                        continue
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(tokens)

                    while len(token_buffer) >= self.seq_len + 1:
                        chunk = token_buffer[:self.seq_len + 1]
                        token_buffer = token_buffer[self.seq_len:]
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield x, y


def prepare_data(args):
    """自动选择数据加载模式：binary mmap 或 JSONL streaming。"""
    train_bin = os.path.join(args.data_dir, 'train_tokens.bin')
    val_bin = os.path.join(args.data_dir, 'val_tokens.bin')

    # 模式 1：预处理过的 binary 文件 → mmap 随机访问
    if os.path.exists(train_bin):
        print(f"  [mmap mode] Found {train_bin}")
        train_ds = TokenMmapDataset(train_bin, args.seq_len)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = None
        if os.path.exists(val_bin):
            val_ds = TokenSequenceDataset(val_bin, args.seq_len)
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=2,
                pin_memory=True,
            )
            print(f"  Train: {train_ds.n_tokens:,} tokens ({len(train_ds):,} samples)")
            print(f"  Val:   {val_ds.n_tokens:,} tokens ({len(val_ds):,} samples)")
        else:
            print(f"  Train: {train_ds.n_tokens:,} tokens ({len(train_ds):,} samples)")
            print(f"  Val:   [not found, skipping]")
        return train_loader, val_loader

    # 模式 2：JSONL 目录 → 流式 tokenize
    # 查找 JSONL 文件（可能在 data/ 子目录下）
    jsonl_dir = args.data_dir
    jsonl_files = glob.glob(os.path.join(jsonl_dir, '**/*.jsonl'), recursive=True)
    if not jsonl_files:
        raise FileNotFoundError(
            f"找不到 binary 文件或 JSONL 文件。\n"
            f"  检查的目录: {args.data_dir}\n"
            f"  期望: train_tokens.bin 或 *.jsonl 文件"
        )

    print(f"  [streaming mode] Found {len(jsonl_files)} JSONL files in {jsonl_dir}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    train_ds = StreamingJsonlDataset(jsonl_dir, tokenizer, args.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=0,  # streaming 不支持多 worker
    )
    print(f"  Tokenizer: {args.pretrained_model} (vocab={tokenizer.vocab_size})")
    print(f"  Val: [streaming mode 暂不支持 val split]")
    return train_loader, None


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
# Checkpoint（仅存可训练参数）
# ============================================================

def _trainable_state_dict(model):
    """只提取 requires_grad=True 的参数，节省 checkpoint 空间。"""
    return {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_checkpoint(path, model, optimizer, step, best_val_ppl, tokens_seen):
    """保存训练状态（仅可训练参数）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'trainable_state_dict': _trainable_state_dict(model),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'best_val_ppl': best_val_ppl,
        'tokens_seen': tokens_seen,
    }, path)
    print(f"  → Checkpoint saved: {path} (step {step}, PPL {best_val_ppl:.1f})")


def load_checkpoint(path, model, optimizer, device):
    """加载 checkpoint，恢复训练状态。"""
    print(f"Loading checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # 支持新格式（trainable_state_dict）和旧格式（model_state_dict）
    if 'trainable_state_dict' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['trainable_state_dict'], strict=False)
        print(f"  Loaded trainable params ({len(ckpt['trainable_state_dict'])} keys)")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded full model state (legacy format)")

    optimizer.load_state_dict(ckpt['optimizer_state'])
    step = ckpt['step']
    best_val_ppl = ckpt['best_val_ppl']
    tokens_seen = ckpt.get('tokens_seen', 0)
    print(f"  Resumed: step={step}, best_val_ppl={best_val_ppl:.1f}, tokens={tokens_seen:,}")
    return step, best_val_ppl, tokens_seen


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device):
    """计算验证集 perplexity。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        total_loss += loss.item()
        total_tokens += target_ids.numel()
        del logits, loss

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))
    return avg_loss, ppl


# ============================================================
# 训练循环
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 准备数据
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = prepare_data(args)

    # 固定 vocab_size（Qwen3-0.6B）
    vocab_size = 151936

    # 构建模型
    if args.resume:
        model = SNNLanguageModel(
            vocab_size=vocab_size,
            D=1024,
            N=args.N,
            K=args.K,
            num_blocks=args.num_blocks,
            v_th_min=args.v_th_min,
        ).to(device)
    else:
        model = SNNLanguageModel.from_pretrained(
            args.pretrained_model,
            N=args.N,
            K=args.K,
            num_blocks=args.num_blocks,
            v_th_min=args.v_th_min,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"\nModel: D=1024, N={args.N}, K={args.K}, Blocks={args.num_blocks}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Frozen params:    {frozen_params:,}")

    # 优化器
    optimizer = SPSAOptimizer(
        params=list(model.parameters()),
        lr=args.lr,
        perturbation_scale=args.perturbation_scale,
        total_steps=args.max_steps,
        lr_min_ratio=0.1,
        grad_clip=args.grad_clip,
    )

    # 恢复 checkpoint
    start_step = 0
    best_val_ppl = float('inf')
    tokens_seen = 0

    if args.resume:
        start_step, best_val_ppl, tokens_seen = load_checkpoint(
            args.resume, model, optimizer, device,
        )

    # 冻结预训练层
    model.embed_tokens.weight.requires_grad = False
    model.norm.weight.requires_grad = False

    # 内存基线
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        mem_baseline = torch.cuda.memory_allocated() / 1e9
        print(f"CUDA memory baseline: {mem_baseline:.2f} GB")

    # 训练信息
    print(f"\n{'='*60}")
    print(f"Training with SPSA (IG-ZO base)")
    print(f"  Max steps:   {args.max_steps:,}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq len:     {args.seq_len}")
    print(f"  LR:          {args.lr} (cosine → {args.lr * 0.1})")
    print(f"  Perturb:     {args.perturbation_scale}")
    print(f"  Grad clip:   {args.grad_clip}")
    print(f"  Eval every:  {args.eval_interval} steps")
    print(f"  Save every:  {args.save_interval} steps")
    if start_step > 0:
        print(f"  Resuming from step {start_step}")
    print(f"{'='*60}\n")

    # 训练循环
    model.train()
    running_loss = 0.0
    running_tokens = 0
    t_start = time.time()
    step = start_step

    for input_ids, target_ids in train_loader:
        if step >= args.max_steps:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 1. 生成扰动
        deltas = optimizer.generate_perturbation()

        # 2. θ + c·δ → L+
        optimizer.perturb(deltas, sign=+1.0)
        with torch.no_grad():
            loss_plus = model(input_ids, target_ids).item()

        # 3. θ - 2c·δ → L-
        optimizer.perturb(deltas, sign=-2.0)
        with torch.no_grad():
            loss_minus = model(input_ids, target_ids).item()

        # 4. 恢复到 θ
        optimizer.perturb(deltas, sign=+1.0)

        # 5. SPSA 更新
        optimizer.step(loss_plus, loss_minus, deltas)

        # 6. 清理：释放扰动张量 + CUDA 缓存
        del deltas
        if step % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

        # 记录
        batch_loss = (loss_plus + loss_minus) / 2.0
        batch_tokens = target_ids.numel()
        running_loss += batch_loss * batch_tokens
        running_tokens += batch_tokens
        tokens_seen += batch_tokens
        step += 1

        # 日志
        if step % args.log_interval == 0:
            avg = running_loss / running_tokens
            ppl = math.exp(min(avg, 100))
            elapsed = time.time() - t_start
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            mem_str = ""
            if device.type == 'cuda':
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem {mem_cur:.1f}/{mem_peak:.1f}GB"
            print(
                f"  Step {step}/{args.max_steps} | "
                f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                f"LR {optimizer.lr:.2e} | "
                f"Tokens {tokens_seen:,} | "
                f"TPS {tps:.0f}{mem_str}"
            )

        # 评估
        if val_loader is not None and step % args.eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            elapsed = time.time() - t_start
            print(
                f"  [Eval] Step {step} | "
                f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f} | "
                f"Time {elapsed:.0f}s"
            )

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                save_checkpoint(
                    os.path.join(args.save_dir, 'best_model.pt'),
                    model, optimizer, step, best_val_ppl, tokens_seen,
                )

            model.train()
            running_loss = 0.0
            running_tokens = 0

        # 定期保存
        if step % args.save_interval == 0:
            save_checkpoint(
                os.path.join(args.save_dir, 'latest.pt'),
                model, optimizer, step, best_val_ppl, tokens_seen,
            )

    # 训练结束
    print(f"\nTraining finished at step {step}.")
    print(f"Best Val PPL: {best_val_ppl:.1f}")
    print(f"Total tokens seen: {tokens_seen:,}")
    if device.type == 'cuda':
        print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    save_checkpoint(
        os.path.join(args.save_dir, 'final.pt'),
        model, optimizer, step, best_val_ppl, tokens_seen,
    )


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SNN Language Model Training v3')

    # 模型参数
    parser.add_argument('--pretrained_model', type=str, default='Qwen/Qwen3-0.6B',
                        help='预训练模型名称')
    parser.add_argument('--N', type=int, default=8, help='状态扩展因子')
    parser.add_argument('--K', type=int, default=8, help='每 token SNN 时间步数')
    parser.add_argument('--num_blocks', type=int, default=10, help='SNN Block 层数')
    parser.add_argument('--v_th_min', type=float, default=0.1, help='阈值下限')

    # 训练参数
    parser.add_argument('--max_steps', type=int, default=100000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128, help='序列长度')
    parser.add_argument('--lr', type=float, default=5e-4, help='SPSA 学习率')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--log_interval', type=int, default=100, help='日志间隔（步）')
    parser.add_argument('--eval_interval', type=int, default=1000, help='评估间隔（步）')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存间隔（步）')

    # 数据
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='预处理后的 binary token 文件目录')

    # Checkpoint
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
