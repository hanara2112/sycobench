#!/usr/bin/env python3
"""
Debug script for steering intervention.

This script diagnoses why steering might not be working by:
1. Verifying hooks are firing
2. Showing actual projection magnitudes
3. Testing different alpha values
4. Comparing baseline vs steered log-probabilities

Usage:
    python debug_steering.py --model Qwen/Qwen2.5-3B-Instruct

The script checks for:
- Hook activation (is the intervention actually happening?)
- Magnitude calibration (is alpha appropriate for the projection magnitudes?)
- Direction correctness (does adding the direction increase sycophancy?)
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from utils import (
    set_seed,
    get_device,
    load_model_and_tokenizer,
    build_chat_prompt,
    create_contrastive_instances,
    logp_of_candidate,
)
from steering_utils import (
    SteeringConfig,
    ActivationSteering,
    train_sycophancy_direction,
)


def diagnose_steering(
    model,
    tokenizer,
    direction: np.ndarray,
    layer_idx: int,
    test_examples: list,
    device: torch.device,
    n_samples: int = 5,
):
    """
    Diagnose steering issues.
    
    Returns diagnostic information.
    """
    print("\n" + "=" * 60)
    print("STEERING DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check direction properties
    print("\n1. DIRECTION PROPERTIES")
    print(f"   Direction shape: {direction.shape}")
    print(f"   Direction norm: {np.linalg.norm(direction):.6f}")
    print(f"   Direction dtype: {direction.dtype}")
    print(f"   Min/Max values: {direction.min():.6f} / {direction.max():.6f}")
    
    # 2. Test hook activation with debug mode
    print("\n2. TESTING HOOK ACTIVATION")
    config = SteeringConfig(
        layer_idx=layer_idx,
        direction=direction,
        alpha=1.0,
        intervention_type="subtraction",
    )
    steering = ActivationSteering(model, config, debug=True)
    
    # Run one forward pass with steering
    test_item = test_examples[0]
    prompt = build_chat_prompt(tokenizer, test_item["question"])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\n   Running forward pass with steering active...")
    with steering:
        with torch.no_grad():
            outputs = model(**inputs)
    
    if steering._intervention_count == 0:
        print("   ⚠️  WARNING: Hook did not fire! This is the bug.")
        print("   Check model architecture and hook target.")
    else:
        print(f"   ✓ Hook fired {steering._intervention_count} times")
    
    # 3. Measure projection magnitudes in the representation space
    print("\n3. PROJECTION MAGNITUDE ANALYSIS")
    
    from utils import gather_residual_activations
    
    proj_magnitudes = []
    for i, item in enumerate(test_examples[:n_samples]):
        prompt = build_chat_prompt(tokenizer, item["question"])
        full_text = prompt + item["answer_matching_behavior"]
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        acts = gather_residual_activations(model, layer_idx, inputs)
        
        # Debug: print shape on first iteration
        if i == 0:
            print(f"   Activation tensor shape: {acts.shape}")
        
        # Handle different shapes: [batch, seq, hidden] or [seq, hidden]
        if acts.dim() == 3:
            act_seq = acts[0]  # [seq, hidden]
        else:
            act_seq = acts  # Already [seq, hidden]
        
        # Project onto direction
        v = torch.tensor(direction, dtype=act_seq.dtype, device=device)
        
        if act_seq.shape[0] == 0:
            print(f"   Warning: Empty sequence at sample {i}")
            continue
            
        projections = torch.matmul(act_seq, v)  # [seq_len]
        last_token_proj = projections[-1].item()
        proj_magnitudes.append(last_token_proj)
    
    if not proj_magnitudes:
        print("   ERROR: Could not get any projections. Something is wrong with activation extraction.")
        mean_proj = 1.0  # Default value
    else:
        mean_proj = np.mean(proj_magnitudes)
        std_proj = np.std(proj_magnitudes)
        
        print(f"   Projection at last token (n={len(proj_magnitudes)}):")
        print(f"     Mean: {mean_proj:.4f}")
        print(f"     Std:  {std_proj:.4f}")
        print(f"     Range: [{min(proj_magnitudes):.4f}, {max(proj_magnitudes):.4f}]")
    
    print(f"\n   Recommended alpha values:")
    print(f"     For subtraction: α = {abs(mean_proj):.1f} to {abs(mean_proj) * 2:.1f}")
    print(f"     For clamping (projection-scaled): α = 1.0")
    
    # 4. Test alpha sweep on one example
    print("\n4. ALPHA SWEEP (single example)")
    
    test_item = test_examples[0]
    prompt = build_chat_prompt(tokenizer, test_item["question"])
    syco_answer = test_item["answer_matching_behavior"]
    non_syco_answer = test_item["answer_not_matching_behavior"]
    
    # Baseline
    logp_syco_base = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
    logp_nonsyco_base = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
    margin_base = logp_syco_base - logp_nonsyco_base
    
    print(f"\n   Baseline (no steering):")
    print(f"     log P(syco) - log P(non-syco) = {margin_base:.4f}")
    print(f"     Model prefers: {'SYCOPHANTIC' if margin_base > 0 else 'NON-SYCOPHANTIC'}")
    
    # Test different alphas with SUBTRACTION
    print(f"\n   Testing SUBTRACTION intervention:")
    alphas_to_test = [1.0, 5.0, 10.0, 20.0, 50.0, float(abs(mean_proj))]
    
    for alpha in alphas_to_test:
        config = SteeringConfig(
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            intervention_type="subtraction",
        )
        steering = ActivationSteering(model, config)
        
        with steering:
            logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
            logp_nonsyco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
        
        margin = logp_syco - logp_nonsyco
        changed = margin < margin_base
        flipped = margin * margin_base < 0
        
        status = "FLIPPED ✓✓" if flipped else ("CHANGED ✓" if changed else "no effect")
        print(f"     α={alpha:6.1f}: margin={margin:+.4f} [{status}]")
    
    # Test CLAMPING (projection-scaled)
    print(f"\n   Testing CLAMPING intervention (projection-scaled):")
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        config = SteeringConfig(
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            intervention_type="clamping",
        )
        steering = ActivationSteering(model, config)
        
        with steering:
            logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
            logp_nonsyco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
        
        margin = logp_syco - logp_nonsyco
        changed = margin < margin_base
        flipped = margin * margin_base < 0
        
        status = "FLIPPED ✓✓" if flipped else ("CHANGED ✓" if changed else "no effect")
        print(f"     α={alpha:4.1f}: margin={margin:+.4f} [{status}]")
    
    # Test ADDITION (should make model MORE sycophantic)
    print(f"\n   Testing ADDITION intervention (sanity check):")
    print(f"   (If direction is correct, this should INCREASE sycophancy)")
    
    for alpha in [10.0, 20.0, 50.0]:
        config = SteeringConfig(
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            intervention_type="addition",
        )
        steering = ActivationSteering(model, config)
        
        with steering:
            logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
            logp_nonsyco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
        
        margin = logp_syco - logp_nonsyco
        increased = margin > margin_base
        
        status = "INCREASED ✓" if increased else "⚠️ decreased (direction may be wrong!)"
        print(f"     α={alpha:4.1f}: margin={margin:+.4f} [{status}]")
    
    # 5. Summary and recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if steering._intervention_count == 0:
        print("\n🔴 CRITICAL: Hooks are not firing. Check:")
        print("   - model.model.layers[layer_idx] exists")
        print("   - Hook is registered before forward pass")
    else:
        print(f"\n   ✓ Hooks are working (fired {steering._intervention_count} times)")
    
    print(f"\n   For SUBTRACTION: try α = {abs(mean_proj):.0f}")
    print(f"   For CLAMPING: use α = 1.0 (removes full projection)")
    print(f"\n   If neither works, the direction may not be causal for behavior.")
    

def main(
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer: int | None = None,
    train_instances: int = 200,
    n_diagnostic_samples: int = 10,
):
    """Run steering diagnostics."""
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        "Anthropic/model-written-evals",
        data_files="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        split="train",
    )
    dataset = dataset.shuffle(seed=seed)
    
    # Split
    test_examples = dataset.select(range(50))
    train_examples = dataset.select(range(50, 50 + train_instances // 2))
    train_data = create_contrastive_instances(train_examples, "train")
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Determine layer
    n_layers = model.config.num_hidden_layers
    if layer is None:
        # Use layer from previous experiments or default to 3/4 of layers
        layer = int(n_layers * 0.75)
    print(f"Using layer {layer} (model has {n_layers} layers)")
    
    # Train direction
    print(f"\nTraining sycophancy direction from {len(train_data)} instances...")
    direction = train_sycophancy_direction(model, tokenizer, train_data, layer, device)
    
    # Run diagnostics
    diagnose_steering(
        model, tokenizer, direction, layer,
        list(test_examples), device,
        n_samples=n_diagnostic_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug steering intervention")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-instances", type=int, default=200)
    
    args = parser.parse_args()
    main(
        seed=args.seed,
        model_name=args.model,
        layer=args.layer,
        train_instances=args.train_instances,
    )
