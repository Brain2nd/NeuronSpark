"""
分布式训练脚本：SNN 语言模型预训练（FSDP 多卡并行）

基于 train.py 单卡脚本，使用 PyTorch FullyShardedDataParallel (FSDP) 实现数据并行。
FSDP 将参数/梯度/优化器状态分片到多卡，大幅节省每卡显存。
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

import functools

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.api import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model import SNNLanguageModel
from torch.utils.tensorboard import SummaryWriter

from dataset import PretrainDataset
from atomic_ops import SNNAdam
from atomic_ops.snn_decoder_layer import SNNDecoderLayer

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
                    health_report=None, max_keep=5, rank=0):
    """保存训练状态（FSDP 版：所有 rank 参与 state_dict 聚合，仅 rank 0 写文件）。"""
    os.makedirs(save_dir, exist_ok=True)

    # 所有 rank 必须参与 state_dict 聚合（collective 操作）
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer)

    # 仅 rank 0 写文件
    if rank == 0:
        path = os.path.join(save_dir, f'ckpt_step{step}.pth')
        ckpt_data = {
            'model_state_dict': model_state,
            'optimizer_state': optim_state,
            'scaler_state': scaler.state_dict(),
            'step': step,
            'epoch': epoch,
            'best_loss': best_loss,
            'tokens_seen': tokens_seen,
            'model_config': {
                'vocab_size': model.vocab_size,
                'D': model.D,
                'N': model.N,
                'K': model.K,
                'num_layers': model.num_layers,
                'D_ff': model.D_ff,
            },
        }
        # 附加健康检查报告
        if health_report:
            ckpt_data['health_report'] = health_report
        torch.save(ckpt_data, path)
        print(f"  → Checkpoint saved: {path}")

        # 清理旧 checkpoint，仅保留最新 max_keep 个（按 step 数字排序，非字符串）
        ckpts = sorted(
            glob.glob(os.path.join(save_dir, 'ckpt_step*.pth')),
            key=lambda p: int(os.path.basename(p).split('step')[1].split('.')[0]),
        )
        while len(ckpts) > max_keep:
            old = ckpts.pop(0)
            os.remove(old)
            print(f"  → Removed old checkpoint: {old}")


def load_checkpoint(path, model, optimizer, scaler, device, rank):
    """加载 checkpoint，恢复训练状态（FSDP 版：所有 rank 参与）。"""
    Logger(f"Loading checkpoint from {path}...", rank)
    # 所有 rank 都需要加载文件（FSDP 需要完整 state_dict 做分片）
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'trainable_state_dict' in ckpt:
            model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    if 'optimizer_state' in ckpt:
        try:
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, ckpt['optimizer_state']
            )
            optimizer.load_state_dict(optim_state)
        except (ValueError, KeyError, RuntimeError):
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
# SNN 健康检查
# ============================================================

@torch.no_grad()
def snn_health_check(model, optimizer=None, loss=None, step=0, rank=0, check_grad=False):
    """
    SNN 训练健康检查（轻量级，不拖慢训练）。

    检查项目：
    1. Loss 异常：NaN/Inf
    2. 神经元发放异常：癫痫（发放率过高）/ 死寂（发放率过低）
    3. 神经元趋同：所有神经元输出相同
    4. 梯度流通异常（仅在 check_grad=True 时检查）
    5. Layer-wise LR 倍率补偿异常

    Args:
        model: 模型
        optimizer: 优化器（可选）
        loss: 当前 loss（可选）
        step: 当前步数
        rank: 进程 rank
        check_grad: 是否检查梯度（耗时，仅 checkpoint 时启用）

    Returns:
        dict: 健康状态报告
    """
    report = {
        'step': step,
        'healthy': True,
        'warnings': [],
        'errors': [],
        'stats': {}
    }
    _raw = model.module if hasattr(model, 'module') else model

    # 1. Loss 检查
    if loss is not None:
        loss_val = loss.item() if torch.is_tensor(loss) else loss
        report['stats']['loss'] = loss_val
        if math.isnan(loss_val) or math.isinf(loss_val):
            report['healthy'] = False
            report['errors'].append(f"Loss 异常: {loss_val}")

    # 2. 神经元发放率检查（通过膜电位 v 和阈值 v_th 估算）
    firing_stats = []
    dead_layers = []
    epileptic_layers = []

    for i, layer in enumerate(_raw.layers):
        snn_block = layer.snn_block
        if hasattr(snn_block, 'hidden_neuron'):
            neuron = snn_block.hidden_neuron
            # 检查是否有 v 和 v_th 属性（SelectivePLIFNode 可能没有）
            if hasattr(neuron, 'v') and hasattr(neuron, 'v_th'):
                v = neuron.v
                if isinstance(v, torch.Tensor):
                    v_th = neuron.v_th
                    v_th_val = v_th.mean().item() if isinstance(v_th, torch.Tensor) else v_th
                    # 估计发放率：v > 0.8 * v_th 的比例
                    firing_ratio = (v > v_th_val * 0.8).float().mean().item()
                    firing_stats.append({'layer': i, 'firing_rate': firing_ratio, 'v_mean': v.mean().item()})

                    if firing_ratio < 0.01:
                        dead_layers.append(i)
                    elif firing_ratio > 0.90:
                        epileptic_layers.append(i)

    report['stats']['firing'] = firing_stats
    if dead_layers:
        report['warnings'].append(f"死寂层: L{dead_layers}")
    if epileptic_layers:
        report['warnings'].append(f"癫痫层: L{epileptic_layers}")

    # 3. 神经元趋同检查
    convergent_layers = []
    for i, layer in enumerate(_raw.layers):
        snn_block = layer.snn_block
        if hasattr(snn_block, 'output_neuron') and hasattr(snn_block.output_neuron, 'v'):
            v = snn_block.output_neuron.v
            if isinstance(v, torch.Tensor) and v.numel() > 1:
                if v.std().item() < 1e-6:
                    convergent_layers.append(i)

    if convergent_layers:
        report['warnings'].append(f"趋同层: L{convergent_layers}")

    # 4. 梯度检查（仅在 check_grad=True 时，用于 checkpoint）
    if check_grad:
        grad_stats = []
        vanishing = []
        exploding = []
        nan_params = []

        for name, param in _raw.named_parameters():
            if param.grad is not None:
                g = param.grad
                norm = g.float().norm().item()
                grad_stats.append({'name': name, 'norm': norm})

                if norm < 1e-7:
                    vanishing.append(name.split('.')[-1])
                elif norm > 1e4:
                    exploding.append(name.split('.')[-1])
                if torch.isnan(g).any() or torch.isinf(g).any():
                    nan_params.append(name)
                    report['healthy'] = False
                    report['errors'].append(f"梯度NaN: {name}")

        report['stats']['grad'] = grad_stats
        if len(vanishing) > len(grad_stats) * 0.3:
            report['warnings'].append(f"梯度消失: {len(vanishing)}/{len(grad_stats)}")
        if exploding:
            report['warnings'].append(f"梯度爆炸: {exploding[:3]}")

    # 5. LR 倍率检查
    if optimizer and hasattr(optimizer, '_layer_scales'):
        scales = optimizer._layer_scales
        if scales:
            report['stats']['lr_scales'] = {'min': min(scales), 'max': max(scales)}
            ratio = max(scales) / max(min(scales), 1e-10)
            if ratio > 100:
                report['warnings'].append(f"LR倍率差异大: {ratio:.0f}x")

    # 汇总
    if report['errors']:
        report['healthy'] = False

    return report


def print_health_report(report, rank=0):
    """打印健康检查报告"""
    if rank != 0:
        return
    step = report.get('step', 0)
    if report['errors']:
        print(f"[Health@{step}] ❌ {report['errors']}")
    if report['warnings']:
        print(f"[Health@{step}] ⚠️ {report['warnings']}")


# ============================================================
# 训练循环
# ============================================================

def train_epoch(epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
                iter_per_epoch, tokens_seen, rank, world_size, writer=None):
    """训练一个 epoch（FSDP 版本）。"""
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

            # 每 5000 步重校准 layer-wise LR (EMA α=0.3)
            global_step = step + 1; _recalib = (global_step == args.accumulation_steps) or (global_step % 5000 == 0)
            if _recalib:
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
                if is_main_process(rank):
                    print(f"  Layer LR recalib: L0={_scales[0]:.2f}x → L{args.num_layers-1}={_scales[-1]:.2f}x")

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # 定期健康检查（每 500 步，或训练开始时）
            global_step = epoch * iter_per_epoch + step + 1
            if global_step == args.accumulation_steps or global_step % 500 == 0:
                health = snn_health_check(model, optimizer, loss, global_step, rank, check_grad=False)
                if not health['healthy']:
                    raise RuntimeError(f"SNN 健康检查失败: {health['errors']}")
                # TensorBoard: 健康检查指标
                if writer and health:
                    for i, info in enumerate(health.get('stats', {}).get('firing', [])):
                        writer.add_scalar(f'health/firing_rate/layer_{i}', info['firing_rate'], global_step)

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

            # TensorBoard
            if writer:
                global_step = epoch * iter_per_epoch + step
                writer.add_scalar('train/loss', batch_loss, global_step)
                writer.add_scalar('train/ppl', batch_ppl, global_step)
                writer.add_scalar('train/lr', lr, global_step)
                writer.add_scalar('train/tps', tps, global_step)
                writer.add_scalar('train/mem_gb', mem_cur, global_step)
                writer.add_scalar('train/mem_peak_gb', mem_peak, global_step)

        # 定期保存（所有 rank 参与 FSDP state_dict 聚合，仅 rank 0 写文件）
        if (step + 1) % args.save_interval == 0:
            model.eval()
            global_step = epoch * iter_per_epoch + step + 1
            _save_loss = loss.item() * args.accumulation_steps
            # 保存时执行完整健康检查（含梯度检查）
            health = snn_health_check(model, optimizer, _save_loss, global_step, rank, check_grad=True)
            print_health_report(health, rank)
            save_checkpoint(args.save_dir, model, optimizer, scaler,
                            global_step, epoch, _save_loss, tokens_seen,
                            health_report=health, rank=rank)
            model.train()

    return tokens_seen


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model Pretraining (FSDP)")

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

    # TensorBoard
    parser.add_argument('--tb_dir', default='runs', type=str, help='TensorBoard 日志根目录')

    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # ==================== 分布式初始化 ====================
    local_rank, rank, world_size = setup_distributed()

    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # 种子：每卡不同（保证数据不同），但可复现
    torch.manual_seed(42 + rank)

    # TensorBoard（仅主进程）
    writer = SummaryWriter(log_dir=os.path.join(args.tb_dir, args.out_dir)) if is_main_process(rank) else None

    # 混合精度：FSDP MixedPrecision 处理参数/通信 dtype，
    # autocast 处理计算 dtype（因为模型内部调用 forward_parallel 绕过 FSDP forward hook）
    mp_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    ctx = torch.amp.autocast('cuda', dtype=mp_dtype)

    # ==================== 模型初始化 ====================
    model, tokenizer, device = init_model(args, local_rank, rank)

    # FSDP 包装（ZeRO-3：参数/梯度/优化器全分片）
    fsdp_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={SNNDecoderLayer},
    )
    fsdp_mixed_precision = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_policy,
        mixed_precision=fsdp_mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,
        device_id=local_rank,
        limit_all_gathers=True,
    )

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
    _pg = model.get_param_groups()

    # dynamics 标记（SNNAdam Phase 1/4-10 使用）
    _dynamics_map = {'b_beta': 'b_beta', 'b_alpha': 'b_alpha', 'b_omega': 'b_omega',
                     'b_th': 'b_th'}

    # 神经元参数 lr_mult（surrogate gradient 天然较弱，需要更高 lr）
    _neuron_keys = {'b_beta', 'b_alpha', 'b_th', 'b_omega',
                    'block_output_neuron', 'ffn_neurons', 'ffn_expert_neurons'}

    # ---- Layer-wise LR: 按层拆分参数组，第一次 accumulated batch 后就地校准 ----
    _layer_map = model.get_layer_indices()
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
    Logger(f"SNN Language Model Pretraining (FSDP, {world_size} GPUs)", rank)
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

    # ==================== 训练前健康检查 ====================
    if is_main_process(rank):
        Logger("Running pre-training health check...", rank)
        pre_health = snn_health_check(model, optimizer, step=0, rank=rank, check_grad=False)
        print_health_report(pre_health, rank)
        if not pre_health['healthy']:
            raise RuntimeError(f"Pre-training health check failed: {pre_health['errors']}")
        Logger("✓ Pre-training health check passed\n", rank)

    # ==================== 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tokens_seen = train_epoch(
            epoch, model, train_loader, sampler, optimizer, scaler, ctx, args,
            iter_per_epoch, tokens_seen, rank, world_size, writer,
        )

    # 最终保存（所有 rank 参与 FSDP state_dict 聚合）
    Logger(f"\nTraining finished. Total tokens seen: {tokens_seen:,}", rank)
    Logger(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank)
    save_checkpoint(args.save_dir, model, optimizer, scaler,
                    args.epochs * iter_per_epoch, args.epochs - 1, 0.0, tokens_seen, rank=rank)

    if writer:
        writer.close()

    cleanup_distributed()
