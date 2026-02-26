"""
分布式训练脚本：SNN 语言模型预训练（DDP 多卡并行）

基于 train.py 单卡脚本，使用 PyTorch DistributedDataParallel 实现数据并行。
训练逻辑、模型架构、超参数与单卡版完全一致。

用法：
    # 单机多卡（例如 4 张 GPU）
    torchrun --nproc_per_node=4 train_ddp.py \
        --D 768 --D_ff 2304 --batch_size 2 --accumulation_steps 8

    # 多机多卡（例如 2 机 × 4 卡）
    torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
        --nproc_per_node=4 train_ddp.py --D 768 --D_ff 2304

    # 单卡也能跑（等价于 train.py）
    torchrun --nproc_per_node=1 train_ddp.py --D 768 --D_ff 2304

    # 断续训练
    torchrun --nproc_per_node=4 train_ddp.py --resume checkpoints/latest.pt
"""

import os
import glob
import time
import math
import argparse
import warnings

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from dataset import PretrainDataset
from atomic_ops import SNNAdam

warnings.filterwarnings('ignore')


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境（由 torchrun 自动设置环境变量）。"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed():
    """清理分布式环境。"""
    dist.destroy_process_group()


def is_main_process(rank):
    """是否为主进程（rank 0）。"""
    return rank == 0


def Logger(content, rank=0):
    """仅主进程打印日志。"""
    if is_main_process(rank):
        print(content)


# ============================================================
# 学习率调度（与 train.py 一致）
# ============================================================

def get_lr(it, total_iters, learning_rate, warmup_iters):
    """余弦退火学习率调度。"""
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
    """保存训练状态，每次不覆盖（带步数），仅保留最新 max_keep 个。"""
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

    # 清理旧 checkpoint，仅保留最新 max_keep 个
    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"  → Removed old checkpoint: {old}")


def load_checkpoint(path, model, optimizer, scaler, device, rank):
    """加载 checkpoint，恢复训练状态。"""
    Logger(f"Loading checkpoint from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    raw_model = model.module if isinstance(model, DDP) else model

    if 'model_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

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
    """初始化模型和分词器。"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    model_kwargs = dict(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    )
    if args.use_moe:
        model_kwargs.update(
            use_moe=True,
            num_experts=args.num_experts,
            top_k=args.top_k,
            D_ff_shared=args.D_ff_shared,
            D_ff_expert=args.D_ff_expert,
        )
    model = SNNLanguageModel(**model_kwargs)

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
    """训练一个 epoch（DDP 版本）。"""
    # 设置 sampler 的 epoch 以保证每个 epoch 的 shuffle 不同
    sampler.set_epoch(epoch)

    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(f"cuda:{local_rank}")
        Y = Y.to(f"cuda:{local_rank}")
        loss_mask = loss_mask.to(f"cuda:{local_rank}")

        # 学习率调度
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)

            # 每 epoch 首个 accumulated batch 重校准 layer-wise LR (EMA α=0.3)
            if (step + 1) == args.accumulation_steps:
                _raw = model.module if isinstance(model, DDP) else model
                _rms = [l.snn_block.W_in.weight.grad.float().pow(2).mean().sqrt().item()
                        for l in _raw.layers]
                _new = [_rms[0] / max(r, 1e-10) for r in _rms]
                _prev = getattr(optimizer, '_layer_scales', None)
                if _prev is not None:
                    _scales = [0.3 * n + 0.7 * p for n, p in zip(_new, _prev)]
                else:
                    _scales = _new
                optimizer._layer_scales = _scales
                for pg in optimizer.param_groups:
                    _lidx = pg.get('_layer_idx')
                    if _lidx is not None:
                        pg['lr_mult'] = pg['_func_lr_mult'] * _scales[_lidx]
                        pg['lr'] = lr * pg['lr_mult']
                if is_main_process(rank):
                    print(f"  Layer LR recalib: L0={_scales[0]:.2f}x → L{args.num_layers-1}={_scales[-1]:.2f}x")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 有效 token 数（汇总所有卡）
        valid_tokens = loss_mask_flat.sum()
        if world_size > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += int(valid_tokens.item())

        # 日志（仅主进程）
        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            batch_ppl = math.exp(min(batch_loss, 20.0))
            spend_time = time.time() - start_time
            tps = tokens_seen / spend_time if spend_time > 0 else 0
            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} ppl:{:.1f} lr:{:.7f} TPS:{:.0f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{}'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, batch_ppl, lr, tps,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        # 定期保存（仅主进程，带步数，保留最新 5 个）
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
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining (DDP)")

    # SNN 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)

    # MoE 参数
    parser.add_argument('--use_moe', action='store_true', help='启用 MoE-SNNFFN')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--D_ff_shared', type=int, default=None)
    parser.add_argument('--D_ff_expert', type=int, default=None)

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8, help="每卡 batch size")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 数据
    parser.add_argument("--data_path", type=str,
                        default="data/seq-monkey/seq_monkey_datawhale.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_snn/")

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # 种子：每卡不同（保证数据不同），但可复现
    torch.manual_seed(42 + rank)

    # 混合精度
    ctx = torch.amp.autocast('cuda',
                             dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args, local_rank, rank)

    # DDP 包装
    model = DDP(model, device_ids=[local_rank])

    # ==================== 数据加载 ====================
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # DistributedSampler 自动按 rank 划分数据
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
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # SNNAdam: Adam + 10 阶段神经动力学增强（对齐 HappyLLM 预训练无 weight decay）
    _pg = model.module.get_param_groups()

    # dynamics 标记（SNNAdam Phase 1/4-10 使用）
    _dynamics_map = {'b_beta': 'b_beta', 'b_alpha': 'b_alpha', 'b_omega': 'b_omega',
                     'b_th': 'b_th'}

    # 神经元参数 lr_mult（surrogate gradient 天然较弱，需要更高 lr）
    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th', 'b_omega',
                    'block_output_neuron', 'ffn_neurons', 'ffn_expert_neurons',
                    'output_neuron'}

    # ---- Layer-wise LR: 按层拆分参数组，第一次 accumulated batch 后就地校准 ----
    _raw_model = model.module if isinstance(model, DDP) else model
    _layer_map = _raw_model.get_layer_indices()
    _no_layer_scale_keys = {'b_beta', 'b_alpha', 'b_omega', 'b_th'}

    param_groups = []
    for key, params in _pg.items():
        if not params:
            continue
        func_lr_mult = float(args.neuron_lr_mult) if key in _neuron_keys else 1.0
        dynamics = _dynamics_map.get(key)
        N_val = args.N if key in ('b_beta', 'b_alpha', 'b_omega', 'b_th') else None

        if key not in _no_layer_scale_keys:
            by_layer = {}
            for p in params:
                lidx = _layer_map.get(id(p))
                by_layer.setdefault(lidx, []).append(p)
            for lidx, lparams in sorted(by_layer.items(), key=lambda x: (x[0] is None, x[0])):
                param_groups.append({
                    'params': lparams,
                    'lr': args.learning_rate * func_lr_mult,
                    'lr_mult': func_lr_mult,
                    '_func_lr_mult': func_lr_mult,
                    '_layer_idx': lidx,
                    'dynamics': dynamics,
                    'N': N_val,
                })
        else:
            param_groups.append({
                'params': params,
                'lr': args.learning_rate * func_lr_mult,
                'lr_mult': func_lr_mult,
                '_func_lr_mult': func_lr_mult,
                '_layer_idx': None,
                'dynamics': dynamics,
                'N': N_val,
            })

    optimizer = SNNAdam(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        grad_clip=args.grad_clip,
    )

    # 恢复 checkpoint
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
    Logger(f"SNN Language Model Pretraining (DDP, {world_size} GPUs)", rank)
    Logger(f"  Vocab:       {args.vocab_size}", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}", rank)
    Logger(f"  Data:        {args.data_path}", rank)
    Logger(f"  Samples:     {len(train_ds):,}", rank)
    Logger(f"  Max length:  {args.max_length}", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective", rank)
    Logger(f"  Epochs:      {args.epochs}", rank)
    Logger(f"  Steps/epoch: {iter_per_epoch:,}", rank)
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})", rank)
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)", rank)
    Logger(f"  Layer LR:    auto-calibrate on first accumulated batch ({len(param_groups)} groups)", rank)
    Logger(f"  Grad clip:   {args.grad_clip}", rank)
    Logger(f"  Precision:   {args.dtype}", rank)
    Logger(f"  Save every:  {args.save_interval} steps", rank)
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

    # 最终保存
    if is_main_process(rank):
        Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}", rank)
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
        save_checkpoint(args.save_dir, model, optimizer, scaler,
                        args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen)

    cleanup_distributed()
