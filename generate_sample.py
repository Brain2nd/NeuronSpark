"""
推理/采样脚本：SNN 语言模型文本生成

支持两种模式：
  1. pretrain: 文本续写（自由补全）
  2. sft: 对话模式（ChatML 模板）

SNN 生成核心机制：
  - Prefill: forward_parallel 并行处理 prompt，建立所有神经元膜电位状态
  - Autoregressive: 逐 token 生成，每 token K=16 次 single_step_forward
  - 神经元 V 状态跨 token 连续传递，不 reset

用法：
    # 文本续写（预训练模型）
    python generate_sample.py \
        --checkpoint checkpoints/ckpt_step10000.pth \
        --mode pretrain --prompt "人工智能"

    # 对话模式（SFT 模型）
    python generate_sample.py \
        --checkpoint checkpoints_sft/ckpt_step3000.pth \
        --mode sft --prompt "什么是脉冲神经网络？"

    # 交互模式
    python generate_sample.py \
        --checkpoint checkpoints/ckpt_step10000.pth \
        --interactive
"""

import argparse

import torch
from transformers import AutoTokenizer

from model import SNNLanguageModel


def load_model(checkpoint_path, device):
    """从 checkpoint 加载模型。"""
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 中读取模型配置
    config = ckpt.get('model_config', {})
    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 6144),
        D=config.get('D', 1024),
        N=config.get('N', 8),
        K=config.get('K', 16),
        num_layers=config.get('num_layers', 20),
        D_ff=config.get('D_ff', 3072),
    )

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        model.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    model = model.to(device).eval()
    step = ckpt.get('step', '?')
    print(f"  Model loaded (step={step}, D={config.get('D')}, Layers={config.get('num_layers')})")
    return model


def pretrain_sample(model, tokenizer, prompt, max_new_tokens=256,
                    temperature=0.8, top_k=50, device='cuda'):
    """预训练模型文本续写。"""
    text = f"{tokenizer.bos_token}{prompt}"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def sft_sample(model, tokenizer, prompt, max_new_tokens=256,
               temperature=0.8, top_k=50, device='cuda'):
    """SFT 模型对话生成。"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # 提取 assistant 回复部分
    marker = "assistant\n"
    idx = full_text.rfind(marker)
    if idx >= 0:
        response = full_text[idx + len(marker):]
        # 去掉可能的结束标记
        for end_token in ["<|im_end|>", tokenizer.eos_token]:
            if end_token and response.endswith(end_token):
                response = response[:-len(end_token)]
        return response.strip()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def interactive_mode(model, tokenizer, args):
    """交互式生成模式。"""
    print(f"\n{'='*50}")
    print(f"SNN Language Model Interactive Generation")
    print(f"Mode: {args.mode} | Temperature: {args.temperature} | Top-k: {args.top_k}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Type 'quit' to exit, 'mode pretrain/sft' to switch mode")
    print(f"{'='*50}\n")

    mode = args.mode
    sample_fn = sft_sample if mode == 'sft' else pretrain_sample

    while True:
        try:
            prompt = input(f"[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == 'quit':
            break
        if prompt.lower().startswith('mode '):
            new_mode = prompt.split()[1].lower()
            if new_mode in ('pretrain', 'sft'):
                mode = new_mode
                sample_fn = sft_sample if mode == 'sft' else pretrain_sample
                print(f"  Switched to {mode} mode")
            continue

        response = sample_fn(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"\n{response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Language Model Text Generation")

    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint 路径')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_snn/')
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # 生成参数
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'sft'], help='生成模式')
    parser.add_argument('--prompt', type=str, default=None, help='输入 prompt（非交互模式）')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)

    # 交互模式
    parser.add_argument('--interactive', action='store_true', help='交互式生成')

    args = parser.parse_args()

    # 加载
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.checkpoint, args.device)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        sample_fn = sft_sample if args.mode == 'sft' else pretrain_sample
        response = sample_fn(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"\n{response}")
    else:
        print("Please specify --prompt or --interactive")
