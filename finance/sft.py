"""
金融模型 SFT 脚本：阶段 2 — 回测最优决策微调

与 train.py (BC) 的核心区别:
  1. 加载 BC 预训练 checkpoint（仅模型权重，optimizer 状态从头开始）
  2. 更低的默认学习率 (5e-5 vs 2e-4)
  3. 添加 compensate_modulation_gradients() + grad_norm clip
  4. 数据源：回测筛选的最优决策标签（features.npy + targets_sft.npy）

I/O 与 BC 完全一致: (batch, seq_len, n_features) → (batch, seq_len, n_assets, 4)

用法:
    conda activate SNN

    # 首次 SFT（从 BC checkpoint 开始）
    python finance/sft.py \
        --pretrained_ckpt checkpoints_finance/ckpt_step65.pth \
        --features_path data/finance/features.npy \
        --targets_path data/finance/targets_sft.npy

    # 断续训练（从 SFT checkpoint 恢复）
    python finance/sft.py --resume checkpoints_finance_sft/ckpt_step500.pth \
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
from torch.utils.data import DataLoader
from contextlib import nullcontext

from finance.model import SNNFinanceModel
from finance.dataset import FinanceDataset
from atomic_ops import SNNAdamW

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)


# ============================================================
# 学习率调度
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

def save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, best_loss, samples_seen,
                    max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'ckpt_step{step}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'samples_seen': samples_seen,
        'stage': 'sft',
        'model_config': {
            'n_features': model.n_features,
            'n_assets': model.n_assets,
            'D': model.D,
            'N': model.N,
            'K': model.K,
            'num_layers': model.num_layers,
            'D_ff': model.D_ff,
            'max_leverage': model.max_leverage,
            'sl_range': (model.sl_min, model.sl_max),
            'tp_range': (model.tp_min, model.tp_max),
        },
    }, path)
    Logger(f"  → Checkpoint saved: {path}")

    ckpts = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')))
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(old)
        Logger(f"  → Removed old checkpoint: {old}")


def load_pretrained(path, model, device):
    """加载 BC 预训练权重（仅模型参数，不加载 optimizer/scaler）。"""
    Logger(f"Loading pretrained BC weights from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    pretrain_step = ckpt.get('step', '?')
    pretrain_loss = ckpt.get('best_loss', '?')
    Logger(f"  Loaded BC model (step={pretrain_step}, loss={pretrain_loss})")


def load_checkpoint(path, model, optimizer, scaler, device):
    """加载 SFT checkpoint，恢复完整训练状态。"""
    Logger(f"Loading SFT checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except (ValueError, KeyError):
            Logger("  Warning: Optimizer state incompatible, starting fresh.")
    if 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    samples_seen = ckpt.get('samples_seen', 0)
    Logger(f"  Resumed: step={step}, epoch={epoch}, samples={samples_seen:,}")
    return step, epoch, best_loss, samples_seen


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, optimizer, scaler, ctx, args, iter_per_epoch, samples_seen):
    start_time = time.time()
    # Epoch 级别累积统计
    epoch_loss_sum, epoch_loss_cnt = 0.0, 0
    epoch_dir_correct, epoch_dir_total = 0, 0
    epoch_mae_sum = torch.zeros(4)

    for step, (features, targets) in enumerate(train_loader):
        features = features.to(args.device)   # (batch, seq_len, n_features)
        targets = targets.to(args.device)     # (batch, seq_len, n_assets, 4)

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
                        for l in model.layers]
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
                Logger(f"  Layer LR recalib: L0={_scales[0]:.2f}x → L{args.num_layers-1}={_scales[-1]:.2f}x")

            # Natural gradient 补偿 + 梯度裁剪
            model.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Epoch 级别统计累积
        batch_loss = loss.item() * args.accumulation_steps
        epoch_loss_sum += batch_loss
        epoch_loss_cnt += 1
        with torch.no_grad():
            pred = out.decisions
            tgt = targets
            tgt_pos = tgt[..., 0]
            pred_pos = pred[..., 0]
            nonzero = tgt_pos.abs() > 0.01
            epoch_dir_correct += ((pred_pos.sign() == tgt_pos.sign()) & nonzero).sum().item()
            epoch_dir_total += nonzero.sum().item()
            epoch_mae_sum += (pred - tgt).abs().mean(dim=(0, 1, 2)).cpu()

        samples_seen += features.shape[0]

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            sps = samples_seen / spend_time if spend_time > 0 else 0
            dir_acc = epoch_dir_correct / max(epoch_dir_total, 1) * 100
            avg_mae = epoch_mae_sum / max(epoch_loss_cnt, 1)

            mem_str = ""
            if args.device != 'cpu':
                mem_cur = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_str = f" | Mem {mem_cur:.1f}/{mem_peak:.1f}GB"
            Logger(
                'SFT [{}/{}]({}/{}) loss:{:.4f}(avg:{:.4f}) lr:{:.7f} '
                'dir:{:.1f}% MAE[p:{:.3f} l:{:.2f} s:{:.4f} t:{:.4f}] '
                'pos:{:+.3f} lev:{:.1f} sl:{:.3f} tp:{:.3f}{}'.format(
                    epoch + 1, args.epochs, step, iter_per_epoch,
                    batch_loss, epoch_loss_sum / max(epoch_loss_cnt, 1), lr,
                    dir_acc, avg_mae[0], avg_mae[1], avg_mae[2], avg_mae[3],
                    pred[..., 0].mean().item(), pred[..., 1].mean().item(),
                    pred[..., 2].mean().item(), pred[..., 3].mean().item(),
                    mem_str))

        if (step + 1) % args.save_interval == 0:
            model.eval()
            global_step = epoch * iter_per_epoch + step + 1
            save_checkpoint(args.save_dir, model, optimizer, scaler,
                            global_step, epoch, batch_loss, samples_seen)
            model.train()

    # Epoch 汇总
    avg_loss = epoch_loss_sum / max(epoch_loss_cnt, 1)
    avg_dir = epoch_dir_correct / max(epoch_dir_total, 1) * 100
    avg_mae = epoch_mae_sum / max(epoch_loss_cnt, 1)
    Logger(f"  ▸ Epoch {epoch+1} summary: avg_loss={avg_loss:.4f} dir_acc={avg_dir:.1f}% "
           f"MAE[pos={avg_mae[0]:.3f} lev={avg_mae[1]:.2f} sl={avg_mae[2]:.4f} tp={avg_mae[3]:.4f}]")

    return samples_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Finance Model SFT (Stage 2)")

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

    # SFT 特有参数
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='BC 预训练 checkpoint 路径（首次 SFT 必须指定）')

    # 训练参数
    parser.add_argument("--out_dir", type=str, default="checkpoints_finance_sft")
    parser.add_argument("--epochs", type=int, default=5, help="SFT 轮数（通常 3-5）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)

    # 优化参数（SFT 默认更低 LR）
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率（SFT 通常比 BC 低 2-5x）')
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=50)
    parser.add_argument('--neuron_lr_mult', type=float, default=10.0)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)

    # 数据
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--targets_path", type=str, required=True,
                        help='回测最优决策标签 (.npy)')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None, help='从 SFT checkpoint 恢复')

    args = parser.parse_args()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        'cuda', dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16)

    # ==================== 模型 ====================
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
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'SNN Finance Model 总参数量：{total_params / 1e6:.3f} 百万')

    # 加载 BC 预训练权重
    if args.pretrained_ckpt and not args.resume:
        load_pretrained(args.pretrained_ckpt, model, args.device)

    # ==================== 数据 ====================
    train_ds = FinanceDataset(
        args.features_path, args.targets_path,
        seq_len=args.seq_len, stride=args.stride,
        n_assets=args.n_assets,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ==================== 优化器（与 train.py 相同结构，但不继承 BC optimizer 状态）====================
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    _pg = model.get_param_groups()

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

    _layer_map = model.get_layer_indices()
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
            args.resume, model, optimizer, scaler, args.device,
        )

    # ==================== 训练信息 ====================
    iter_per_epoch = len(train_loader)

    Logger(f"\n{'='*60}")
    Logger(f"SNN Finance Model SFT (Stage 2: Backtested Optimal)")
    Logger(f"  Model:       D={args.D}, N={args.N}, K={args.K}, Layers={args.num_layers}, D_ff={args.D_ff}")
    Logger(f"  Features:    {args.n_features} → Decisions: {args.n_assets} assets × 4 dims")
    Logger(f"  Params:      {total_params / 1e6:.3f}M")
    Logger(f"  Data:        {args.features_path}")
    Logger(f"  Targets:     {args.targets_path}")
    Logger(f"  Samples:     {len(train_ds):,}")
    Logger(f"  Seq length:  {args.seq_len} (stride {train_ds.stride})")
    Logger(f"  Batch size:  {args.batch_size} × accum {args.accumulation_steps} = {args.batch_size * args.accumulation_steps} effective")
    Logger(f"  Epochs:      {args.epochs}")
    Logger(f"  Steps/epoch: {iter_per_epoch:,}")
    Logger(f"  LR:          {args.learning_rate} (warmup {args.warmup_iters} → cosine → {args.learning_rate/10})")
    Logger(f"  Neuron LR:   {args.learning_rate * args.neuron_lr_mult} ({args.neuron_lr_mult}× base)")
    Logger(f"  Grad clip:   {args.grad_clip}")
    Logger(f"  Precision:   {args.dtype}")
    if args.pretrained_ckpt:
        Logger(f"  Pretrained:  {args.pretrained_ckpt}")
    if args.device != 'cpu':
        mem_baseline = torch.cuda.memory_allocated() / 1e9
        Logger(f"  CUDA memory: {mem_baseline:.2f} GB baseline")
    Logger(f"{'='*60}\n")

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        samples_seen = train_epoch(
            epoch, model, train_loader, optimizer, scaler, ctx, args,
            iter_per_epoch, samples_seen,
        )

    Logger(f"\nSFT finished. Total samples seen: {samples_seen:,}")
    if args.device != 'cpu':
        Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    save_checkpoint(args.save_dir, model, optimizer, scaler,
                    args.epochs * iter_per_epoch, args.epochs - 1, 0.0, samples_seen)
