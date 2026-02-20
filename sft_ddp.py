"""
分布式 SFT 训练脚本：SNN 语言模型监督微调（DDP 多卡并行）

基于 train_ddp.py 和 sft.py，核心差异：
  1. 使用 SFTDataset（对话格式 + loss mask 仅计算 assistant 回复）
  2. 加载预训练 checkpoint 权重（--pretrained_ckpt）
  3. Optimizer 状态不继承

用法：
    # 单机多卡
    torchrun --nproc_per_node=4 sft_ddp.py \
        --pretrained_ckpt checkpoints/ckpt_step10000.pth \
        --sft_data_path data/sft/sft_data.jsonl \
        --D 768 --D_ff 2304 --batch_size 2 --accumulation_steps 8

    # 断续训练
    torchrun --nproc_per_node=4 sft_ddp.py --resume checkpoints_sft/ckpt_step500.pth
"""

import os
import glob
import time
import math
import argparse
import warnings

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import SFTDataset

warnings.filterwarnings('ignore')


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def Logger(content, rank=0):
    if is_main_process(rank):
        print(content)


# ============================================================
# 学习率调度
# ============================================================

def get_lr(it, total_iters, learning_rate, warmup_iters):
    min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > total_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, best_loss, tokens_seen,
                    max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')
    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'tokens_seen': tokens_seen,
        'model_config': {
            'vocab_size': raw_model.vocab_size,
            'D': raw_model.D,
            'N': raw_model.N,
            'K': raw_model.K,
            'num_layers': raw_model.num_layers,
            'D_ff': raw_model.D_ff,
        },
    }, path)
    print(f"  → Checkpoint saved: {path}")

    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"  → Removed old checkpoint: {old}")


def load_pretrained(path, model, device, rank):
    Logger(f"Loading pretrained weights from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_model = model.module if isinstance(model, DDP) else model
    if 'model_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    pretrain_step = ckpt.get('step', '?')
    Logger(f"  Loaded pretrained model (step={pretrain_step})", rank)


def load_checkpoint(path, model, optimizer, scaler, device, rank):
    Logger(f"Loading SFT checkpoint from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_model = model.module if isinstance(model, DDP) else model
    if 'model_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)

    if 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except (ValueError, KeyError):
            Logger("  Warning: Optimizer state incompatible, starting fresh.", rank)

    if 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    tokens_seen = ckpt.get('tokens_seen', 0)
    Logger(f"  Resumed: step={step}, epoch={epoch}, tokens={tokens_seen:,}", rank)
    return step, epoch, best_loss, tokens_seen


# ============================================================
# 初始化
# ============================================================

def init_model(args, local_rank, rank):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN LM 总参数量：{total_params / 1e6:.3f} 百万', rank)

    return model, tokenizer, device


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
                iter_per_epoch, tokens_seen, rank, world_size):
    sampler.set_epoch(epoch)
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(f"cuda:{local_rank}")
        Y = Y.to(f"cuda:{local_rank}")
        loss_mask = loss_mask.to(f"cuda:{local_rank}")

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            torch.cuda.empty_cache()

        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                'SFT Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{}'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        if (step + 1) % args.save_interval == 0:
            if is_main_process(rank):
                model.eval()
                global_step = epoch * iter_per_epoch + step + 1
                save_checkpoint(args.save_dir, model, optimizer, scaler,
                                global_step, epoch, batch_loss, tokens_seen)
                model.train()
            if world_size > 1:
                dist.barrier()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model SFT (DDP)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # SFT 特有
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--sft_data_path', type=str, default='data/sft/sft_data.jsonl')

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 数据
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42 + rank)

    ctx = torch.amp.autocast('cuda',
                             dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args, local_rank, rank)

    # 加载预训练权重（DDP 包装前加载）
    if args.pretrained_ckpt and not args.resume:
        load_pretrained(args.pretrained_ckpt, model, device, rank)

    # DDP 包装
    model = DDP(model, device_ids=[local_rank])

    # ==================== 数据 ====================
    train_ds = SFTDataset(args.sft_data_path, tokenizer, max_length=args.max_length)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器 ====================
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype in ['float16', 'bfloat16']))

    _pg = model.module.get_param_groups()
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}
    neuron_params = [p for k in _neuron_keys for p in _pg[k]]
    other_params = [p for k, ps in _pg.items() if k not in _neuron_keys for p in ps]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'lr_mult': 1.0},
        {'params': neuron_params, 'lr': args.learning_rate * args.neuron_lr_mult,
         'lr_mult': float(args.neuron_lr_mult)},
    ])

    # 恢复 SFT checkpoint
    tokens_seen = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, best_loss, tokens_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, device, rank,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Logger(f"\n{'='*60}", rank)
    Logger(f"SNN Language Model SFT (DDP, {world_size} GPUs)", rank)
    Logger(f"  Vocab:       {args.vocab_size}", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}", rank)
    Logger(f"  SFT Data:    {args.sft_data_path}", rank)
    Logger(f"  Samples:     {len(train_ds):,}", rank)
    Logger(f"  Max length:  {args.max_length}", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective", rank)
    Logger(f"  Epochs:      {args.epochs}", rank)
    Logger(f"  Steps/epoch: {iter_per_epoch:,}", rank)
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})", rank)
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)", rank)
    Logger(f"  Grad clip:   {args.grad_clip}", rank)
    Logger(f"  Precision:   {args.dtype}", rank)
    Logger(f"  Save every:  {args.save_interval} steps", rank)
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen = train_epoch(
            epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size,
        )

    if is_main_process(rank):
        Logger(f"\nSFT finished. Total tokens seen: {tokens_seen:,}", rank)
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
        save_checkpoint(args.save_dir, model, optimizer, scaler,
                        args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen)

    cleanup_distributed()
