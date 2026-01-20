"""
Activation steering experiment for sycophancy mitigation.

This script trains a sycophancy direction and applies steering interventions
to reduce sycophantic behavior in LLMs.

Usage:
    python run_steering.py --model Qwen/Qwen2.5-3B-Instruct --layer 16 --alpha 1.0

Methods:
    - subtraction: h' = h - α·v (remove sycophancy direction)
    - clamping: h' = h - (h·v)·v (project out direction completely)
    - addition: h' = h + α·(-v) (amplify anti-sycophancy)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
    MultiLayerSteering,
    train_sycophancy_direction,
    evaluate_steering_effect,
    generate_with_steering,
)


def measure_sycophancy_rate(
    model,
    tokenizer,
    test_examples,
    device: torch.device,
) -> tuple[float, list[dict]]:
    """
    Measure the baseline sycophancy rate of the model.
    
    Returns:
        Tuple of (sycophancy_rate, detailed_results)
    """
    results = []
    sycophantic_count = 0
    
    for item in tqdm(test_examples, desc="Measuring sycophancy rate"):
        prompt = build_chat_prompt(tokenizer, item["question"])
        
        syco_answer = item["answer_matching_behavior"]
        non_syco_answer = item["answer_not_matching_behavior"]
        
        logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
        logp_non_syco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
        
        is_sycophantic = logp_syco > logp_non_syco
        if is_sycophantic:
            sycophantic_count += 1
        
        results.append({
            "question": item["question"][:100] + "...",
            "is_sycophantic": is_sycophantic,
            "logp_syco": logp_syco,
            "logp_non_syco": logp_non_syco,
            "margin": logp_syco - logp_non_syco,
        })
    
    rate = sycophantic_count / len(test_examples)
    return rate, results


def measure_sycophancy_rate_with_steering(
    model,
    tokenizer,
    test_examples,
    steering: ActivationSteering | MultiLayerSteering,
    device: torch.device,
) -> tuple[float, list[dict]]:
    """
    Measure sycophancy rate with steering active.
    """
    results = []
    sycophantic_count = 0
    
    with steering:
        for item in tqdm(test_examples, desc="Measuring with steering"):
            prompt = build_chat_prompt(tokenizer, item["question"])
            
            syco_answer = item["answer_matching_behavior"]
            non_syco_answer = item["answer_not_matching_behavior"]
            
            logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
            logp_non_syco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
            
            is_sycophantic = logp_syco > logp_non_syco
            if is_sycophantic:
                sycophantic_count += 1
            
            results.append({
                "question": item["question"][:100] + "...",
                "is_sycophantic": is_sycophantic,
                "logp_syco": logp_syco,
                "logp_non_syco": logp_non_syco,
                "margin": logp_syco - logp_non_syco,
            })
    
    rate = sycophantic_count / len(test_examples)
    return rate, results


def run_alpha_sweep(
    model,
    tokenizer,
    test_examples,
    direction: np.ndarray,
    layer_idx: int,
    intervention_type: str,
    alphas: list[float],
    device: torch.device,
) -> list[dict]:
    """
    Sweep over different steering strengths (alpha values).
    """
    results = []
    
    for alpha in alphas:
        print(f"\n  Testing α = {alpha:.2f}...")
        
        config = SteeringConfig(
            layer_idx=layer_idx,
            direction=direction,
            alpha=alpha,
            intervention_type=intervention_type,
        )
        steering = ActivationSteering(model, config)
        
        rate, _ = measure_sycophancy_rate_with_steering(
            model, tokenizer, test_examples, steering, device
        )
        
        results.append({
            "alpha": alpha,
            "sycophancy_rate": rate,
        })
        print(f"    Sycophancy rate: {rate:.4f}")
    
    return results


def main(
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer: int | None = None,
    alpha: float = 1.0,
    intervention_type: str = "subtraction",
    train_instances: int = 500,
    test_instances: int = 200,
    alpha_sweep: bool = False,
    output_dir: str = "results",
):
    """
    Run steering experiment.
    
    Args:
        seed: Random seed
        model_name: HuggingFace model name
        layer: Layer to intervene on (if None, uses middle layer)
        alpha: Steering strength
        intervention_type: Type of intervention (subtraction, clamping, addition)
        train_instances: Number of training instances for direction
        test_instances: Number of test instances for evaluation
        alpha_sweep: Whether to sweep over alpha values
        output_dir: Output directory for results
    """
    set_seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Intervention type: {intervention_type}")
    print(f"Steering strength (α): {alpha}")
    
    # Load dataset
    print("\nLoading Anthropic/model-written-evals dataset...")
    dataset = load_dataset(
        "Anthropic/model-written-evals",
        data_files="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        split="train",
    )
    print(f"Total dataset size: {len(dataset)} examples")
    
    dataset = dataset.shuffle(seed=seed)
    
    # Calculate prompts needed
    test_prompts = test_instances // 2
    train_prompts = train_instances // 2
    total_prompts_needed = test_prompts + train_prompts
    
    if total_prompts_needed > len(dataset):
        raise ValueError(
            f"Need {total_prompts_needed} prompts but dataset only has {len(dataset)}"
        )
    
    # Create splits - keep raw examples for sycophancy rate measurement
    test_examples_raw = dataset.select(range(test_prompts))
    train_examples_raw = dataset.select(range(test_prompts, test_prompts + train_prompts))
    
    # Create contrastive instances for direction training
    train_data = create_contrastive_instances(train_examples_raw, "train")
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} instances ({train_prompts} prompts)")
    print(f"  Test: {len(test_examples_raw)} prompts")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Determine layer to use
    n_layers = model.config.num_hidden_layers
    if layer is None:
        layer = n_layers // 2  # Default to middle layer
    
    print(f"\nModel has {n_layers} layers")
    print(f"Using layer {layer} for intervention")
    
    # Measure baseline sycophancy rate
    print("\n" + "=" * 60)
    print("BASELINE MEASUREMENT")
    print("=" * 60)
    
    baseline_rate, baseline_results = measure_sycophancy_rate(
        model, tokenizer, test_examples_raw, device
    )
    print(f"\nBaseline sycophancy rate: {baseline_rate:.4f} ({int(baseline_rate * len(test_examples_raw))}/{len(test_examples_raw)})")
    
    # Train sycophancy direction
    print("\n" + "=" * 60)
    print("TRAINING SYCOPHANCY DIRECTION")
    print("=" * 60)
    
    direction = train_sycophancy_direction(
        model, tokenizer, train_data, layer, device
    )
    print(f"Direction shape: {direction.shape}")
    print(f"Direction norm (should be 1.0): {np.linalg.norm(direction):.6f}")
    
    # Create steering configuration
    config = SteeringConfig(
        layer_idx=layer,
        direction=direction,
        alpha=alpha,
        intervention_type=intervention_type,
    )
    steering = ActivationSteering(model, config)
    
    # Measure sycophancy rate with steering
    print("\n" + "=" * 60)
    print(f"STEERING MEASUREMENT (α={alpha}, type={intervention_type})")
    print("=" * 60)
    
    steered_rate, steered_results = measure_sycophancy_rate_with_steering(
        model, tokenizer, test_examples_raw, steering, device
    )
    
    print(f"\nSteered sycophancy rate: {steered_rate:.4f} ({int(steered_rate * len(test_examples_raw))}/{len(test_examples_raw)})")
    
    # Calculate impact
    reduction = baseline_rate - steered_rate
    reduction_pct = (reduction / baseline_rate * 100) if baseline_rate > 0 else 0
    
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Baseline sycophancy rate: {baseline_rate:.4f}")
    print(f"Steered sycophancy rate:  {steered_rate:.4f}")
    print(f"Absolute reduction:       {reduction:.4f}")
    print(f"Relative reduction:       {reduction_pct:.1f}%")
    
    # Count flips
    flipped_to_honest = sum(
        1 for b, s in zip(baseline_results, steered_results)
        if b["is_sycophantic"] and not s["is_sycophantic"]
    )
    flipped_to_syco = sum(
        1 for b, s in zip(baseline_results, steered_results)
        if not b["is_sycophantic"] and s["is_sycophantic"]
    )
    
    print(f"\nBehavior changes:")
    print(f"  Flipped to honest: {flipped_to_honest}")
    print(f"  Flipped to sycophantic: {flipped_to_syco}")
    print(f"  Net improvement: {flipped_to_honest - flipped_to_syco}")
    
    # Optional: Alpha sweep
    alpha_sweep_results = None
    if alpha_sweep:
        print("\n" + "=" * 60)
        print("ALPHA SWEEP")
        print("=" * 60)
        
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        alpha_sweep_results = run_alpha_sweep(
            model, tokenizer, test_examples_raw, direction, layer,
            intervention_type, alphas, device
        )
        
        print("\nAlpha sweep results:")
        print(f"{'Alpha':<10} {'Sycophancy Rate':<20}")
        print("-" * 30)
        for r in alpha_sweep_results:
            print(f"{r['alpha']:<10.2f} {r['sycophancy_rate']:<20.4f}")
        
        # Find best alpha
        best = min(alpha_sweep_results, key=lambda x: x["sycophancy_rate"])
        print(f"\nBest α: {best['alpha']} (rate: {best['sycophancy_rate']:.4f})")
    
    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "experiment": "steering",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": seed,
            "model_name": model_name,
            "layer": layer,
            "n_layers": n_layers,
            "alpha": alpha,
            "intervention_type": intervention_type,
            "train_instances": len(train_data),
            "test_instances": len(test_examples_raw),
        },
        "baseline_sycophancy_rate": baseline_rate,
        "steered_sycophancy_rate": steered_rate,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
        "flipped_to_honest": flipped_to_honest,
        "flipped_to_syco": flipped_to_syco,
        "net_improvement": flipped_to_honest - flipped_to_syco,
        "alpha_sweep_results": alpha_sweep_results,
    }
    
    # Save direction vector
    direction_file = output_path / f"sycophancy_direction_layer{layer}.npy"
    np.save(direction_file, direction)
    print(f"\nDirection saved to {direction_file}")
    
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_file = output_path / f"steering_{model_short}_{intervention_type}_results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation steering for sycophancy mitigation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to intervene on (default: middle layer)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering strength",
    )
    parser.add_argument(
        "--intervention-type",
        type=str,
        default="subtraction",
        choices=["subtraction", "clamping", "addition"],
        help="Type of intervention",
    )
    parser.add_argument("--train-instances", type=int, default=500)
    parser.add_argument("--test-instances", type=int, default=200)
    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help="Run sweep over alpha values",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    main(
        seed=args.seed,
        model_name=args.model,
        layer=args.layer,
        alpha=args.alpha,
        intervention_type=args.intervention_type,
        train_instances=args.train_instances,
        test_instances=args.test_instances,
        alpha_sweep=args.alpha_sweep,
        output_dir=args.output_dir,
    )
