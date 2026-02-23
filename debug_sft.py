"""
调试SFT训练：详细打印内部状态
"""
import os
import math
import torch
from torch import optim
from transformers import AutoTokenizer
from model import SNNLanguageModel
from dataset import SFTDataset
from torch.utils.data import DataLoader

def main():
    device = "cuda:0"
    
    # 加载模型
    print("=== 加载模型 ===")
    model = SNNLanguageModel(
        vocab_size=6144,
        D=896,
        N=8,
        K=16,
        num_layers=20,
        D_ff=2688,
    ).to(device)
    
    # 加载预训练权重
    ckpt_path = "checkpoints/ckpt_step85000.pth"
    print(f"加载checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("模型加载完成")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_snn")
    
    # 加载数据
    print("\n=== 加载数据 ===")
    dataset = SFTDataset("data/sft/sft_data.jsonl", tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 测试训练步骤
    print("\n=== 测试计算图 ===")
    model.train()
    
    X, Y, loss_mask = next(iter(loader))
    X = X.to(device)
    Y = Y.to(device)
    loss_mask = loss_mask.to(device)
    
    print(f"X requires_grad: {X.requires_grad}")
    print(f"Y requires_grad: {Y.requires_grad}")
    print(f"loss_mask requires_grad: {loss_mask.requires_grad}")
    
    # 前向传播
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(X, Y)
        raw_loss = out.last_loss
        print(f"\nraw_loss requires_grad: {raw_loss.requires_grad}")
        print(f"raw_loss grad_fn: {raw_loss.grad_fn}")
        
        loss_mask_flat = loss_mask.view(-1)
        mask_sum = loss_mask_flat.sum()
        print(f"mask_sum: {mask_sum.item()}")
        print(f"loss_mask_flat dtype: {loss_mask_flat.dtype}")
        
        # 问题可能在这里 - loss_mask是int类型
        loss_mask_float = loss_mask_flat.float()
        print(f"loss_mask_float dtype: {loss_mask_float.dtype}")
        
        masked = raw_loss * loss_mask_float
        print(f"masked requires_grad: {masked.requires_grad}")
        print(f"masked grad_fn: {masked.grad_fn}")
        
        loss = torch.sum(masked) / mask_sum.float()
        print(f"\nloss requires_grad: {loss.requires_grad}")
        print(f"loss grad_fn: {loss.grad_fn}")
        print(f"loss value: {loss.item()}")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    main()
