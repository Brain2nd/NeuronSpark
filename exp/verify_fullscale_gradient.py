"""
Full-scale gradient health check for SNNLanguageModel.

Creates a model at near-production scale (D=1024, 22 layers) and verifies that
gradients flow through every parameter without NaN, None, all-zero, or vanishing
values. Reports layer-by-layer norms and the L0/L21 gradient ratio to diagnose
potential vanishing/exploding gradient issues.

Usage:
    TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
    conda run --no-capture-output -n SNN python -u exp/verify_fullscale_gradient.py
"""

import sys
import os
import gc
import math

import torch
import torch.nn as nn

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SNNLanguageModel


def grad_norm(param):
    """Compute L2 norm of gradient. Returns None if no grad."""
    if param.grad is None:
        return None
    return param.grad.float().norm().item()


def main():
    # ================================================================
    # Config
    # ================================================================
    vocab_size = 6144
    D = 1024
    N = 8
    K = 16
    num_layers = 22
    D_ff = 3072
    batch_size = 1
    seq_len = 32

    print("=" * 70)
    print("Full-Scale Gradient Health Check")
    print("=" * 70)
    print(f"Model config: vocab={vocab_size}, D={D}, N={N}, K={K}, "
          f"layers={num_layers}, D_ff={D_ff}")
    print(f"Input: batch_size={batch_size}, seq_len={seq_len}")
    print()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ================================================================
    # 1. Create model
    # ================================================================
    print("[1/5] Creating model...")
    model = SNNLanguageModel(
        vocab_size=vocab_size,
        D=D,
        N=N,
        K=K,
        num_layers=num_layers,
        D_ff=D_ff,
    ).to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable:        {trainable_params:,}")

    mem_after_model = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after model: {mem_after_model:.2f} GB")
    print()

    # ================================================================
    # 2. Forward pass
    # ================================================================
    print("[2/5] Running forward pass...")
    token_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    model.train()
    output = model(token_ids, target_ids)
    loss = output.last_loss.mean()
    print(f"  Loss: {loss.item():.4f}")

    mem_after_fwd = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after forward: {mem_after_fwd:.2f} GB")
    print()

    # ================================================================
    # 3. Backward pass
    # ================================================================
    print("[3/5] Running backward pass...")
    loss.backward()

    mem_after_bwd = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after backward: {mem_after_bwd:.2f} GB")
    print()

    # Free activations
    del output, loss, token_ids, target_ids
    torch.cuda.empty_cache()
    gc.collect()

    mem_after_cleanup = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after cleanup: {mem_after_cleanup:.2f} GB")
    print()

    # ================================================================
    # 4. Check all gradients
    # ================================================================
    print("[4/5] Checking all parameter gradients...")
    print()

    issues = []
    total_checked = 0
    total_ok = 0

    # Build a name->param dict once
    all_params = {name: param for name, param in model.named_parameters()
                  if param.requires_grad}

    # 4a. Non-layer parameters
    print("  --- Non-layer Parameters ---")
    print(f"  {'Parameter':<45} {'Shape':<25} {'Grad Norm':>12} {'Status':>8}")
    print(f"  {'-'*45} {'-'*25} {'-'*12} {'-'*8}")

    for name, param in all_params.items():
        if name.startswith("layers."):
            continue

        total_checked += 1
        gn = grad_norm(param)
        shape_str = str(list(param.shape))

        if gn is None:
            status = "NONE"
            issues.append((name, "gradient is None"))
        elif math.isnan(gn):
            status = "NaN"
            issues.append((name, "gradient contains NaN"))
        elif gn == 0.0:
            status = "ZERO"
            issues.append((name, "gradient is exactly zero"))
        elif gn < 1e-15:
            status = "VANISH"
            issues.append((name, f"gradient vanishing: {gn:.2e}"))
        else:
            status = "OK"
            total_ok += 1

        gn_str = f"{gn:.6e}" if gn is not None else "None"
        print(f"  {name:<45} {shape_str:<25} {gn_str:>12} {status:>8}")

    print()

    # 4b. Layer-by-layer analysis
    print("  --- Per-Layer Gradient Norms (Key Components) ---")
    print()

    # Key component patterns to track
    key_components = [
        ("snn_block.W_in.weight", "W_in"),
        ("snn_block.W_out.weight", "W_out"),
        ("snn_block.b_beta", "b_beta"),
        ("snn_block.b_alpha", "b_alpha"),
        ("snn_block.b_th", "b_th"),
        ("snn_block.output_neuron.w", "out_n.w"),
        ("snn_block.output_neuron.v_th", "out_n.vth"),
        ("block_out_proj.weight", "blk_proj"),
        ("ffn_out_proj.weight", "ffn_proj"),
        ("input_neuron1.w", "in_n1.w"),
        ("input_neuron2.w", "in_n2.w"),
        ("snn_ffn.gate_proj.weight", "ffn_gate"),
        ("snn_ffn.down_proj.weight", "ffn_down"),
        ("snn_ffn.output_neuron.w", "ffn_on.w"),
    ]

    # Print header
    comp_names = [c[1] for c in key_components]
    header = f"  {'Layer':>5} " + " ".join(f"{cn:>10}" for cn in comp_names)
    print(header)
    print(f"  {'-'*5} " + " ".join(f"{'-'*10}" for _ in comp_names))

    # Per-layer gradient norms for ratio computation
    layer_total_norms = {}  # layer_idx -> float

    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        row_values = []
        layer_norm_sq = 0.0

        for comp_path, comp_label in key_components:
            full_name = prefix + comp_path
            param = all_params.get(full_name, None)
            if param is not None:
                gn = grad_norm(param)
                row_values.append(gn)
                if gn is not None and not math.isnan(gn):
                    layer_norm_sq += gn ** 2
            else:
                row_values.append(None)

        row_str = f"  {layer_idx:>5} "
        for v in row_values:
            if v is None:
                row_str += f"{'N/A':>10} "
            else:
                row_str += f"{v:>10.3e} "
        print(row_str)

        layer_total_norms[layer_idx] = math.sqrt(layer_norm_sq)

        # Also check ALL params in this layer
        for name, param in all_params.items():
            if not name.startswith(prefix):
                continue
            total_checked += 1
            gn = grad_norm(param)
            if gn is None:
                issues.append((name, "gradient is None"))
            elif math.isnan(gn):
                issues.append((name, "gradient contains NaN"))
            elif gn == 0.0:
                issues.append((name, "gradient is exactly zero"))
            elif gn < 1e-15:
                issues.append((name, f"gradient vanishing: {gn:.2e}"))
            else:
                total_ok += 1

    print()

    # ================================================================
    # 5. Summary and L0/L21 gradient ratio
    # ================================================================
    print("[5/5] Summary")
    print("=" * 70)

    l0_norm = layer_total_norms.get(0, 0.0)
    l_last = num_layers - 1
    l_last_norm = layer_total_norms.get(l_last, 0.0)

    print(f"  Layer 0 total grad norm (key components):  {l0_norm:.6e}")
    print(f"  Layer {l_last} total grad norm (key components): {l_last_norm:.6e}")

    if l_last_norm > 0 and l0_norm > 0:
        ratio = l0_norm / l_last_norm
        print(f"  L0 / L{l_last} gradient ratio: {ratio:.4f}")
        if ratio < 0.01:
            print(f"  WARNING: Severe vanishing gradients (ratio < 0.01)")
        elif ratio < 0.1:
            print(f"  WARNING: Moderate gradient attenuation (ratio < 0.1)")
        elif ratio > 100:
            print(f"  WARNING: Possible exploding gradients (ratio > 100)")
        else:
            print(f"  Gradient ratio is healthy.")
    else:
        print(f"  Cannot compute ratio (zero norm detected).")

    print()
    print(f"  Total parameters checked: {total_checked}")
    print(f"  Passed: {total_ok}")
    print(f"  Failed: {total_checked - total_ok}")
    print()

    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for name, desc in issues:
            print(f"    - {name}: {desc}")
        print()

    # Overall verdict
    if len(issues) == 0:
        print("  *** OVERALL: PASS ***")
        print("  All gradients are non-None, non-NaN, non-zero, and above 1e-15.")
    else:
        print(f"  *** OVERALL: FAIL ({len(issues)} issue(s)) ***")

    print()
    print("=" * 70)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return 0 if len(issues) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
