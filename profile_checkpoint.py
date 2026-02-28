"""Checkpoint 显存验证脚本 — 验证 saved_tensors 是否跨层累积。

用法: python profile_checkpoint.py

测试 3 个场景:
1. 有 checkpoint: forward 后 retained 应只有 ~346 MB (层边界 spike)
2. 无 checkpoint: forward 后 retained 应 ~50 GB (所有 saved_tensors)
3. 逐层 backward: 每层只增 ~104 MB (梯度累积), 不是 ~2.5 GB (激活累积)
"""

import os, gc, torch, types
from torch.utils.checkpoint import checkpoint as ckpt
from spikingjelly.activation_based import functional

os.environ.setdefault('TRITON_PTXAS_PATH', '/usr/local/cuda-13.0/bin/ptxas')

from model import SNNLanguageModel

D, N, K, NUM_LAYERS, D_FF = 1024, 8, 16, 20, 3072
BATCH, SEQ = 1, 512


def mem():
    return torch.cuda.memory_allocated() / 1024**2


def run(use_ckpt, label):
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    model = SNNLanguageModel(
        vocab_size=6144, D=D, N=N, K=K,
        num_layers=NUM_LAYERS, D_ff=D_FF,
    ).cuda().bfloat16()
    model_mb = mem()

    ids = torch.randint(1, 6144, (BATCH, SEQ), device='cuda')
    tgt = torch.randint(1, 6144, (BATCH, SEQ), device='cuda')

    if use_ckpt:
        out = model(ids, tgt)
    else:
        for lm in model.layers:
            functional.reset_net(lm)
        spike = model.encode(ids)
        for lm in model.layers:
            functional.reset_net(lm)
            spike = lm.forward_parallel(spike)
        logits = model.decode(spike, SEQ)
        loss_raw = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), tgt.reshape(-1),
            ignore_index=0, reduction='none',
        )
        out = type('O', (), {'last_loss': loss_raw})()

    loss = out.last_loss.mean()
    fwd_mb = mem()
    fwd_peak = torch.cuda.max_memory_allocated() / 1024**2
    retained = fwd_mb - model_mb

    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"[{label}]")
    print(f"  Model:     {model_mb:.0f} MB")
    print(f"  Retained:  {retained:.0f} MB")
    print(f"  Fwd peak:  {fwd_peak:.0f} MB")
    print(f"  Bwd peak:  {bwd_peak:.0f} MB")

    del model, out, loss, ids, tgt; gc.collect(); torch.cuda.empty_cache()
    return retained, bwd_peak


def run_per_layer():
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

    model = SNNLanguageModel(
        vocab_size=6144, D=D, N=N, K=K,
        num_layers=NUM_LAYERS, D_ff=D_FF,
    ).cuda().bfloat16()

    ids = torch.randint(1, 6144, (BATCH, SEQ), device='cuda')
    tgt = torch.randint(1, 6144, (BATCH, SEQ), device='cuda')

    layer_mem = {}
    def hook(i):
        def fn(*a):
            torch.cuda.synchronize()
            layer_mem[i] = mem()
        return fn

    def snn_fwd(self, spike_seq):
        spike = spike_seq
        def _lf(lm, x):
            functional.reset_net(lm)
            return lm.forward_parallel(x)
        for i, lm in enumerate(self.layers):
            spike = ckpt(_lf, lm, spike, use_reentrant=False)
            spike.register_hook(hook(i))
        return spike

    model.snn_forward = types.MethodType(snn_fwd, model)
    out = model(ids, tgt)
    loss = out.last_loss.mean()
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    bwd_peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n[Per-layer backward] (bwd peak: {bwd_peak:.0f} MB)")
    for i in sorted(layer_mem):
        delta = f"+{layer_mem[i] - layer_mem[i+1]:.0f}" if i + 1 in layer_mem else ""
        print(f"  L{i:>2}: {layer_mem[i]:.0f} MB  {delta}")

    per_layer_delta = (layer_mem[0] - layer_mem[NUM_LAYERS - 1]) / (NUM_LAYERS - 1)
    print(f"\n  每层平均增量: {per_layer_delta:.0f} MB")
    if per_layer_delta < 200:
        print(f"  ✓ 增量 ~{per_layer_delta:.0f} MB ≈ 层参数梯度, 不是激活累积")
    else:
        print(f"  ✗ 增量 ~{per_layer_delta:.0f} MB, 可能存在激活累积!")

    del model, out, loss, ids, tgt; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: D={D}, N={N}, K={K}, layers={NUM_LAYERS}, batch={BATCH}, seq={SEQ}")
    print(f"1 DN tensor = {SEQ*K * BATCH * D*N * 2 / 1024**2:.0f} MB")
    print()

    r1, b1 = run(True,  "WITH checkpoint")
    r2, b2 = run(False, "NO checkpoint")

    print(f"\nCheckpoint 释放: {r2 - r1:.0f} MB saved_tensors ({r2/max(r1,1):.0f}x)")
    print(f"Backward peak 差: {b2 - b1:.0f} MB")

    run_per_layer()
