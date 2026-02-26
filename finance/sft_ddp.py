"""
分布式金融模型 SFT 脚本：阶段 2（DDP 多卡并行）

基于 finance/sft.py 单卡脚本，核心差异:
  1. 加载 BC 预训练 checkpoint（仅模型权重）
  2. 更低默认 LR (5e-5)
  3. compensate_modulation_gradients() + clip_grad_norm_()

用法：
    torchrun --nproc_per_node=4 finance/sft_ddp.py \
        --pretrained_ckpt checkpoints_finance/ckpt_step65.pth \
        --features_path data/finance/features.npy \
        --targets_path data/finance/targets_sft.npy

    # 断续训练
    torchrun --nproc_per_node=4 finance/sft_ddp.py \
        --resume checkpoints_finance_sft/ckpt_step500.pth \
        --features_path data/finance/features.npy \
        --targets_path data/finance/targets_sft.npy
"""

import os
import sys
import glob
import time
import math
import argparse
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from finance.model import SNNFinanceModel
from finance.dataset import FinanceDataset
from atomic_ops import SNNAdamW

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

def save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, best_loss, samples_seen,
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
        'samples_seen': samples_seen,
        'stage': 'sft',
        'model_config': {
            'n_features': raw_model.n_features,
            'n_assets': raw_model.n_assets,
            'D': raw_model.D,
            'N': raw_model.N,
            'K': raw_model.K,
            'num_layers': raw_model.num_layers,
            'D_ff': raw_model.D_ff,
            'max_leverage': raw_model.max_leverage,
            'sl_range': (raw_model.sl_min, raw_model.sl_max),
            'tp_range': (raw_model.tp_min, raw_model.tp_max),
        },
    }, path)
    print(f"  → Checkpoint saved: {path}")

    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        print(f"  → Removed old checkpoint: {old}")


def load_pretrained(path, model, device, rank):
    Logger(f"Loading pretrained BC weights from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    if 'model_state_dict' in ckpt:
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    Logger(f"  Loaded BC model (step={ckpt.get('step', '?')})", rank)


def load_checkpoint(path, model, optimizer, scaler, device, rank):
    Logger(f"Loading SFT checkpoint from {path}...", rank)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
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
    samples_seen = ckpt.get('samples_seen', 0)
    Logger(f"  Resumed: step={step}, epoch={epoch}, samples={samples_seen:,}", rank)
    return step, epoch, best_loss, samples_seen


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
                iter_per_epoch, samples_seen, rank, world_size):
    sampler.set_epoch(epoch)
    start_time = time.time()
    local_rank = int(os.environ["LOCAL_RANK"])
    raw_model = model.module if isinstance(model, DDP) else model

    for step, (features, targets) in enumerate(train_loader):
        features = features.to(f"cuda:{local_rank}")
        targets = targets.to(f"cuda:{local_rank}")

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch,
                     args.learning_rate, args.warmup_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

        with ctx:
            out = model(features, targets)
            loss = out.last_loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)

            # 每 epoch 首个 accumulated batch 重校准 layer LR
            if (step + 1) == args.accumulation_steps:
                _rms = [l.snn_block.W_in.weight.grad.float().pow(2).mean().sqrt().item()
                        for l in raw_model.layers]
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

            raw_model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_samples = torch.tensor(features.shape[0], device=f"cuda:{local_rank}")
        if world_size > 1:
            dist.all_reduce(batch_samples, op=dist.ReduceOp.SUM)
        samples_seen += int(batch_samples.item())

        if step % args.log_interval == 0 and is_main_process(rank):
            batch_loss = loss.item() * args.accumulation_steps
            spend_time = time.time() - start_time
            sps = samples_seen / spend_time if spend_time > 0 else 0

            with torch.no_grad():
                pos_mean = out.decisions[..., 0].mean().item()
                lev_mean = out.decisions[..., 1].mean().item()
                sl_mean  = out.decisions[..., 2].mean().item()
                tp_mean  = out.decisions[..., 3].mean().item()

            mem_cur = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                'SFT Epoch:[{}/{}]({}/{}) loss:{:.4f} lr:{:.7f} SPS:{:.0f} '
                'pos:{:+.3f} lev:{:.1f} sl:{:.3f} tp:{:.3f} '
                'epoch_Time:{}min | Mem {:.1f}/{:.1f}GB | GPUs:{}'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, lr, sps,
                    pos_mean, lev_mean, sl_mean, tp_mean,
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    mem_cur, mem_peak, world_size))

        if (step + 1) % args.save_interval == 0:
            if is_main_process(rank):
                model.eval()
                global_step = epoch * iter_per_epoch + step + 1
                save_checkpoint(args.save_dir, model, optimizer, scaler,
                                global_step, epoch, batch_loss, samples_seen)
                model.train()
            if world_size > 1:
                dist.barrier()

    return samples_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Finance Model SFT (DDP)")

    # 模型参数
    parser.add_argument('--n_features', type=int, default=987)
    parser.add_argument('--n_assets', type=int, default=1)
    parser.add_argument('--D', type=int, default=256)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--D_ff', type=int, default=768)
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--max_leverage', type=float, default=10.0)
    parser.add_argument('--sl_min', type=float, default=0.005)
    parser.add_argument('--sl_max', type=float, default=0.10)
    parser.add_argument('--tp_min', type=float, default=0.01)
    parser.add_argument('--tp_max', type=float, default=0.20)
    parser.add_argument('--recon_weight', type=float, default=0.1)

    # SFT 特有
    parser.add_argument('--pretrained_ckpt', type=str, default=None)

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_finance_sft")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4, help="每卡 batch size")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=50)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)

    # 数据
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--targets_path", type=str, required=True)

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

    # ==================== 模型 ====================
    device = torch.device(f"cuda:{local_rank}")

    model = SNNFinanceModel(
        n_features=args.n_features,
        n_assets=args.n_assets,
        D=args.D, N=args.N, K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
        max_leverage=args.max_leverage,
        sl_range=(args.sl_min, args.sl_max),
        tp_range=(args.tp_min, args.tp_max),
        recon_weight=args.recon_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN Finance Model 总参数量：{total_params / 1e6:.3f} 百万', rank)

    # 加载 BC 权重（DDP 包装前）
    if args.pretrained_ckpt and not args.resume:
        load_pretrained(args.pretrained_ckpt, model, device, rank)

    model = DDP(model, device_ids=[local_rank])
    raw_model = model.module

    # ==================== 数据 ====================
    train_ds = FinanceDataset(
        args.features_path, args.targets_path,
        seq_len=args.seq_len, stride=args.stride,
        n_assets=args.n_assets,
    )

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

    _pg = raw_model.get_param_groups()

    _wd_groups = {
        'input_proj': 0.01, 'decision_head': 0.01, 'decode': 0.01,
        'W_in': 0.01, 'W_beta': 0.01, 'W_alpha': 0.01, 'W_th': 0.01, 'W_omega': 0.01,
        'W_gate': 0.01, 'W_skip': 0.01, 'W_out': 0.01,
        'residual_projs': 0.01,
        'ffn_gate_proj': 0.01, 'ffn_up_proj': 0.01, 'ffn_down_proj': 0.01, 'ffn_skip_proj': 0.01,
        'b_beta': 0.0, 'b_alpha': 0.0, 'b_th': 0.0, 'b_omega': 0.0,
        'input_neurons': 0.0, 'block_output_neuron': 0.0, 'ffn_neurons': 0.0, 'output_neuron': 0.0,
        'norm': 0.0, 'rms_norms': 0.0,
    }

    _dynamics_map = {'b_beta': 'b_beta', 'b_alpha': 'b_alpha', 'b_omega': 'b_omega',
                     'b_th': 'b_th'}

    _neuron_keys = {'input_neurons', 'b_beta', 'b_alpha', 'b_th', 'b_omega',
                    'block_output_neuron', 'ffn_neurons', 'output_neuron'}

    _layer_map = raw_model.get_layer_indices()
    _no_layer_scale_keys = {'b_beta', 'b_alpha', 'b_omega', 'b_th'}

    param_groups = []
    for key, params in _pg.items():
        if not params:
            continue
        func_lr_mult = float(args.neuron_lr_mult) if key in _neuron_keys else 1.0
        wd = _wd_groups.get(key, 0.01)
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
                    'weight_decay': wd,
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
                'weight_decay': wd,
                'dynamics': dynamics,
                'N': N_val,
            })

    optimizer = SNNAdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        grad_clip=args.grad_clip,
    )

    samples_seen = 0
    start_epoch = 0
    if args.resume:
        start_step, start_epoch, best_loss, samples_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, device, rank,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps * world_size

    Logger(f"\n{'='*60}", rank)
    Logger(f"SNN Finance Model SFT (DDP, {world_size} GPUs)", rank)
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}", rank)
    Logger(f"  Features:    {args.n_features} → Decisions: {args.n_assets} assets × 4 dims", rank)
    Logger(f"  Params:      {total_params / 1e6:.3f}M", rank)
    Logger(f"  Data:        {args.features_path}", rank)
    Logger(f"  Samples:     {len(train_ds):,}", rank)
    Logger(f"  Seq length:  {args.seq_len} (stride {train_ds.stride})", rank)
    Logger(f"  Batch size:  {args.batch_size}/gpu × {world_size} gpus × accum {args.accumulation_steps} = {effective_batch} effective", rank)
    Logger(f"  Epochs:      {args.epochs}", rank)
    Logger(f"  Steps/epoch: {iter_per_epoch:,}", rank)
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})", rank)
    Logger(f"  Grad clip:   {args.grad_clip}", rank)
    Logger(f"  Precision:   {args.dtype}", rank)
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}", rank)
    mem_baseline = torch.cuda.memory_allocated() / 1e9
    Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline (GPU {local_rank})", rank)
    Logger(f"{'='*60}\n", rank)

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        samples_seen = train_epoch(
            epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
            iter_per_epoch, samples_seen, rank, world_size,
        )

    if is_main_process(rank):
        Logger(f"\nSFT finished. Total samples seen: {samples_seen:,}", rank)
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
        save_checkpoint(args.save_dir, model, optimizer, scaler,
                        args.epochs * iter_per_epoch, args.epochs - 1, 0.0, samples_seen)

    cleanup_distributed()
