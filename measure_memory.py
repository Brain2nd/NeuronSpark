"""显存测量脚本：安全地测量不同 batch_size 下的显存峰值。

先 batch_size=1,2 测量，从差值推算 max batch_size，再验证。
"""
import os, sys, gc, torch
sys.path.insert(0, os.path.dirname(__file__))

from model import SNNLanguageModel
from atomic_ops import SNNAdam

# ====== 模型配置 (对标 HappyLLM ~215M) ======
MODEL_CFG = dict(
    vocab_size=6144,
    D=512, N=8, K=16,
    num_layers=12,
    D_ff=1536,
    v_th_min=0.1,
)
SEQ_LEN = 512  # max_length (对齐 HappyLLM)
DTYPE = torch.bfloat16


def measure_one(batch_size, model, optimizer, device):
    """单次 forward + backward，返回峰值显存 (GB)。"""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    x = torch.randint(0, MODEL_CFG['vocab_size'], (batch_size, SEQ_LEN), device=device)
    y = torch.randint(0, MODEL_CFG['vocab_size'], (batch_size, SEQ_LEN), device=device)

    with torch.amp.autocast('cuda', dtype=DTYPE):
        out = model(x, y)
        loss = out.last_loss.mean()

    loss.backward()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1e9

    del x, y, out, loss
    gc.collect()
    torch.cuda.empty_cache()

    return peak


def main():
    device = torch.device('cuda:0')

    # 创建模型
    model = SNNLanguageModel(**MODEL_CFG).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params / 1e6:.1f}M params")

    # 基础显存（模型 + 优化器状态初始化前）
    model_mem = torch.cuda.memory_allocated(device) / 1e9
    print(f"Model memory: {model_mem:.2f} GB")

    # 简单优化器（只需要 step 兼容）
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # 测量 batch_size=1
    print("\n--- Measuring batch_size=1 ---")
    peak1 = measure_one(1, model, optimizer, device)
    print(f"  peak memory: {peak1:.2f} GB")

    # 测量 batch_size=2
    print("\n--- Measuring batch_size=2 ---")
    peak2 = measure_one(2, model, optimizer, device)
    print(f"  peak memory: {peak2:.2f} GB")

    # 推算
    delta_per_batch = peak2 - peak1
    total_gpu = torch.cuda.get_device_properties(device).total_memory / 1e9
    safety_margin = 4.0  # 预留 4 GB 安全裕量

    if delta_per_batch > 0:
        max_bs = int((total_gpu - safety_margin - peak1) / delta_per_batch) + 1
    else:
        max_bs = 1

    print(f"\n===== Summary =====")
    print(f"GPU total:          {total_gpu:.1f} GB")
    print(f"Safety margin:      {safety_margin:.1f} GB")
    print(f"Model baseline:     {model_mem:.2f} GB")
    print(f"peak(bs=1):         {peak1:.2f} GB")
    print(f"peak(bs=2):         {peak2:.2f} GB")
    print(f"delta/batch:        {delta_per_batch:.2f} GB")
    print(f"Estimated max bs:   {max_bs}")
    print(f"Recommended safe:   {max(1, max_bs - 2)}")

    # 验证推荐值
    safe_bs = max(1, max_bs - 2)
    if safe_bs > 2:
        print(f"\n--- Verifying batch_size={safe_bs} ---")
        peak_safe = measure_one(safe_bs, model, optimizer, device)
        print(f"  peak memory: {peak_safe:.2f} GB / {total_gpu:.1f} GB ({peak_safe/total_gpu*100:.0f}%)")

    # 也测一下中间值
    mid_bs = max(1, max_bs // 2)
    if mid_bs > 2 and mid_bs != safe_bs:
        print(f"\n--- Verifying batch_size={mid_bs} ---")
        peak_mid = measure_one(mid_bs, model, optimizer, device)
        print(f"  peak memory: {peak_mid:.2f} GB / {total_gpu:.1f} GB ({peak_mid/total_gpu*100:.0f}%)")


if __name__ == '__main__':
    main()
