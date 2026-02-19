"""
真实训练速度对比：legacy vs sequential scan
用真实模型、真实数据、真实训练循环，跑固定步数计时。
"""
import os
import sys
import time
import torch
import torch.optim as optim
from contextlib import nullcontext

sys.path.insert(0, '/home/dgxspark/Desktop/NeuronSpark')
import atomic_ops.parallel_scan as ps
from atomic_ops.parallel_scan import _linear_recurrence_legacy, _linear_recurrence_sequential

# ============================================================
# 初始化模型和数据（和 train.py 完全一致）
# ============================================================
from model import SNNLanguageModel
from train import PretrainDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

device = 'cuda'
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained('./tokenizer_snn/')
model = SNNLanguageModel(
    vocab_size=6144, D=1024, N=8, K=16, num_layers=20, D_ff=3072, v_th_min=0.1,
).to(device)
model.train()

train_ds = PretrainDataset('data/seq-monkey/seq_monkey_datawhale.jsonl', tokenizer, max_length=512)
train_loader = DataLoader(train_ds, batch_size=8, pin_memory=True, drop_last=False, shuffle=True, num_workers=4)

optimizer = optim.Adam(model.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler('cuda', enabled=True)
ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16)

accumulation_steps = 4
NUM_STEPS = 20  # 每种实现跑的步数

# ============================================================
# 训练 N 步的函数
# ============================================================
def run_steps(label, lr_fn, num_steps):
    """用指定的 linear_recurrence 跑 num_steps 步真实训练，返回总时间和 token 数"""
    ps.linear_recurrence = lr_fn
    optimizer.zero_grad(set_to_none=True)
    total_tokens = 0

    # warmup 2 步
    data_iter = iter(train_loader)
    for _ in range(2):
        X, Y, mask = next(data_iter)
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / accumulation_steps
            mask_flat = mask.view(-1)
            loss = torch.sum(loss * mask_flat) / mask_flat.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 正式计时
    t0 = time.perf_counter()
    for step in range(num_steps):
        X, Y, mask = next(data_iter)
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / accumulation_steps
            mask_flat = mask.view(-1)
            loss = torch.sum(loss * mask_flat) / mask_flat.sum()

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_tokens += int(mask_flat.sum().item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = total_tokens / elapsed
    ms_per_step = elapsed / num_steps * 1000
    print(f"  {label:12s}: {elapsed:.2f}s, {num_steps} steps, {ms_per_step:.0f}ms/step, TPS={tps:.1f}")
    return elapsed, tps

# ============================================================
# 对比实验
# ============================================================
print(f"Model: D=1024, N=8, K=16, layers=20, batch=8")
print(f"Steps per run: {NUM_STEPS}")
print(f"{'='*60}")

t1, tps1 = run_steps("Legacy", _linear_recurrence_legacy, NUM_STEPS)
t2, tps2 = run_steps("Sequential", _linear_recurrence_sequential, NUM_STEPS)

print(f"{'='*60}")
print(f"  Speedup: {t1/t2:.2f}x (Legacy {tps1:.1f} TPS → Sequential {tps2:.1f} TPS)")

# 恢复默认
ps.linear_recurrence = ps._linear_recurrence_sequential
