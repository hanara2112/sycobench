"""
Cross-Concept Transfer Evaluation.

Tests if sycophancy direction is concept-specific
- Train sycophancy detector → Test on deception (should FAIL)
- Train deception detector → Test on sycophancy (should FAIL)

If cross-concept transfer fails → Direction is specific to the concept,
not a generic "agreement" or "manipulation" feature.

Usage:
    python run_cross_concept.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import (
    set_seed,
    get_device,
    load_model_and_tokenizer,
    load_jsonl,
)
from run_layer_sweep import (
    extract_activations,
    train_diffmean,
    compute_scores,
    find_optimal_threshold,
)


def load_concept_data(concept: str, setting: str, split: str, seed: int = 0, max_size: int | None = None) -> list[dict]:
    """Load data for a specific concept and setting."""
    data_dir = Path(__file__).parent / "data" / concept / setting / f"seed_{seed}"
    data_file = data_dir / f"{split}.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = load_jsonl(str(data_file))
    
    if max_size is not None and len(data) > max_size:
        import random
        random.seed(seed)
        random.shuffle(data)
        data = data[:max_size]
    
    return data


def compute_direction_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two directions."""
    d1_norm = d1 / (np.linalg.norm(d1) + 1e-8)
    d2_norm = d2 / (np.linalg.norm(d2) + 1e-8)
    return float(np.dot(d1_norm, d2_norm))


def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer: int | None = None,
    train_size: int = 400,
    test_size: int = 400,
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Run cross-concept transfer evaluation.
    
    Tests:
    1. Train sycophancy → Test sycophancy (should work)
    2. Train sycophancy → Test deception (should NOT work)
    3. Train deception → Test deception (should work)
    4. Train deception → Test sycophancy (should NOT work)
    """
    set_seed(seed)
    device = get_device()
    
    print("=" * 60)
    print("Cross-Concept Transfer Evaluation")
    print("Addresses Reviewer 2: Agreement vs Sycophancy Concern")
    print("=" * 60)
    print(f"Model: {model_name}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    n_layers = model.config.num_hidden_layers
    
    if layer is None:
        layer = int(n_layers * 0.6)
    layer = min(layer, n_layers - 1)
    print(f"Layer: {layer} (of {n_layers})")
    
    # Load datasets
    print("\nLoading datasets...")
    syco_train = load_concept_data("sycophancy", "political", "train", seed=0, max_size=train_size)
    syco_test = load_concept_data("sycophancy", "political", "test", seed=0, max_size=test_size)
    decep_train = load_concept_data("deception", "truthfulqa", "train", seed=0, max_size=train_size)
    decep_test = load_concept_data("deception", "truthfulqa", "test", seed=0, max_size=test_size)
    
    print(f"  Sycophancy: {len(syco_train)} train, {len(syco_test)} test")
    print(f"  Deception: {len(decep_train)} train, {len(decep_test)} test")
    
    # Train sycophancy direction
    print("\n" + "-" * 40)
    print("Training SYCOPHANCY direction...")
    print("-" * 40)
    pos_acts_syco, neg_acts_syco = extract_activations(
        model, tokenizer, syco_train, layer, device, desc="Syco Train"
    )
    syco_direction = train_diffmean(pos_acts_syco, neg_acts_syco)
    
    # Get threshold for sycophancy
    syco_train_scores, syco_train_labels = compute_scores(
        model, tokenizer, syco_train, syco_direction, layer, device
    )
    syco_threshold, _ = find_optimal_threshold(syco_train_scores, syco_train_labels)
    
    # Train deception direction
    print("\n" + "-" * 40)
    print("Training DECEPTION direction...")
    print("-" * 40)
    pos_acts_decep, neg_acts_decep = extract_activations(
        model, tokenizer, decep_train, layer, device, desc="Decep Train"
    )
    decep_direction = train_diffmean(pos_acts_decep, neg_acts_decep)
    
    # Get threshold for deception
    decep_train_scores, decep_train_labels = compute_scores(
        model, tokenizer, decep_train, decep_direction, layer, device
    )
    decep_threshold, _ = find_optimal_threshold(decep_train_scores, decep_train_labels)
    
    # Compute direction similarity
    dir_similarity = compute_direction_similarity(syco_direction, decep_direction)
    print(f"\nDirection cosine similarity: {dir_similarity:.4f}")
    
    # Evaluate all 4 combinations
    def evaluate(scores, labels, threshold):
        preds = (scores > threshold).astype(int)
        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
    
    print("\n" + "-" * 40)
    print("Evaluating cross-concept transfer...")
    print("-" * 40)
    
    # Sycophancy direction on both test sets
    syco_on_syco_scores, syco_on_syco_labels = compute_scores(
        model, tokenizer, syco_test, syco_direction, layer, device
    )
    syco_on_decep_scores, syco_on_decep_labels = compute_scores(
        model, tokenizer, decep_test, syco_direction, layer, device
    )
    
    # Deception direction on both test sets
    decep_on_decep_scores, decep_on_decep_labels = compute_scores(
        model, tokenizer, decep_test, decep_direction, layer, device
    )
    decep_on_syco_scores, decep_on_syco_labels = compute_scores(
        model, tokenizer, syco_test, decep_direction, layer, device
    )
    
    results = {
        "syco_on_syco": evaluate(syco_on_syco_scores, syco_on_syco_labels, syco_threshold),
        "syco_on_decep": evaluate(syco_on_decep_scores, syco_on_decep_labels, syco_threshold),
        "decep_on_decep": evaluate(decep_on_decep_scores, decep_on_decep_labels, decep_threshold),
        "decep_on_syco": evaluate(decep_on_syco_scores, decep_on_syco_labels, decep_threshold),
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("CROSS-CONCEPT TRANSFER RESULTS")
    print("=" * 60)
    print(f"{'Train → Test':<30} {'AUROC':<12} {'Accuracy':<12} {'F1':<12}")
    print("-" * 66)
    print(f"{'Sycophancy → Sycophancy':<30} {results['syco_on_syco']['auroc']:<12.4f} {results['syco_on_syco']['accuracy']:<12.4f} {results['syco_on_syco']['f1']:<12.4f}")
    print(f"{'Sycophancy → Deception':<30} {results['syco_on_decep']['auroc']:<12.4f} {results['syco_on_decep']['accuracy']:<12.4f} {results['syco_on_decep']['f1']:<12.4f}")
    print(f"{'Deception → Deception':<30} {results['decep_on_decep']['auroc']:<12.4f} {results['decep_on_decep']['accuracy']:<12.4f} {results['decep_on_decep']['f1']:<12.4f}")
    print(f"{'Deception → Sycophancy':<30} {results['decep_on_syco']['auroc']:<12.4f} {results['decep_on_syco']['accuracy']:<12.4f} {results['decep_on_syco']['f1']:<12.4f}")
    
    print(f"\nDirection Similarity: {dir_similarity:.4f}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION (Addresses Reviewer 2)")
    print("=" * 60)
    
    in_domain_works = (
        results['syco_on_syco']['auroc'] > 0.75 and 
        results['decep_on_decep']['auroc'] > 0.75
    )
    cross_fails = (
        results['syco_on_decep']['auroc'] < 0.65 and 
        results['decep_on_syco']['auroc'] < 0.65
    )
    directions_distinct = abs(dir_similarity) < 0.5
    
    if in_domain_works:
        print("✓ In-domain detection works (AUROC > 0.75 for both concepts)")
    else:
        print("✗ In-domain detection may not be working well")
    
    if cross_fails:
        print("✓ Cross-concept transfer fails (AUROC < 0.65)")
        print("  → Directions are CONCEPT-SPECIFIC, not generic 'agreement'")
    else:
        print("~ Cross-concept transfer partially works")
        print("  → Possible shared manipulation signal or confound")
    
    if directions_distinct:
        print(f"✓ Direction similarity = {dir_similarity:.3f} < 0.5")
        print("  → Directions are geometrically distinct")
    else:
        print(f"~ Direction similarity = {dir_similarity:.3f} >= 0.5")
        print("  → Directions may share some structure")
    
    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "experiment": "cross_concept_transfer",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "layer": layer,
            "seed": seed,
        },
        "direction_similarity": dir_similarity,
        "results": results,
        "interpretation": {
            "in_domain_works": in_domain_works,
            "cross_fails": cross_fails,
            "directions_distinct": directions_distinct,
            "conclusion": "Concept-specific" if (cross_fails and directions_distinct) else "Possible shared signal",
        },
    }
    
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_file = output_path / f"cross_concept_{model_short}_results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Concept Transfer Evaluation (Addresses Reviewer 2)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        layer=args.layer,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
