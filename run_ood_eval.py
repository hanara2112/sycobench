"""
OOD (Out-of-Domain) Generalization Evaluation.

Addresses Reviewer 1 concern: "Dataset confound - detector might be 
capturing a proxy signal that correlates with dataset structure."

Tests if the sycophancy direction generalizes across domains:
- Train: political + philpapers
- Test: nlp_survey (held-out domain)

If AUROC > 0.80 on OOD test → Feature is NOT a dataset artifact.

Usage:
    python run_ood_eval.py --model Qwen/Qwen2.5-3B-Instruct
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


def load_domain_data(domain: str, split: str, seed: int = 0, max_size: int | None = None) -> list[dict]:
    """Load data for a specific domain."""
    data_dir = Path(__file__).parent / "data" / "sycophancy" / domain / f"seed_{seed}"
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


def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer: int | None = None,
    train_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Run OOD generalization evaluation.
    
    Train on political + philpapers, test on nlp_survey (OOD).
    """
    set_seed(seed)
    device = get_device()
    
    print("=" * 60)
    print("OOD Generalization Evaluation")
    print("Addresses Reviewer 1: Dataset Confound Concern")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Train domains: political + philpapers")
    print(f"Test domain: nlp_survey (OOD)")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    n_layers = model.config.num_hidden_layers
    
    if layer is None:
        layer = int(n_layers * 0.6)  # 60% depth
    layer = min(layer, n_layers - 1)
    print(f"Layer: {layer} (of {n_layers})")
    
    # Load training data from multiple domains
    print("\nLoading train data (political + philpapers)...")
    train_political = load_domain_data("political", "train", seed=0, max_size=train_size // 2)
    train_philpapers = load_domain_data("philpapers", "train", seed=0, max_size=train_size // 2)
    train_data = train_political + train_philpapers
    print(f"  Political: {len(train_political)}")
    print(f"  Philpapers: {len(train_philpapers)}")
    print(f"  Total train: {len(train_data)}")
    
    # Load in-domain test data
    print("\nLoading in-domain test data (political)...")
    test_id = load_domain_data("political", "test", seed=0, max_size=test_size // 2)
    print(f"  In-domain test: {len(test_id)}")
    
    # Load OOD test data
    print("\nLoading OOD test data (nlp_survey)...")
    test_ood = load_domain_data("nlp_survey", "test", seed=0, max_size=test_size // 2)
    print(f"  OOD test: {len(test_ood)}")
    
    # Extract training activations
    print("\nExtracting train activations...")
    pos_acts_train, neg_acts_train = extract_activations(
        model, tokenizer, train_data, layer, device, desc="Train"
    )
    
    # Train direction
    print("Training diff-in-means direction...")
    direction = train_diffmean(pos_acts_train, neg_acts_train)
    
    # Compute training scores for threshold
    print("Computing training scores...")
    train_scores, train_labels = compute_scores(
        model, tokenizer, train_data, direction, layer, device
    )
    threshold, train_acc = find_optimal_threshold(train_scores, train_labels)
    
    # Compute in-domain test scores
    print("Computing in-domain test scores...")
    id_scores, id_labels = compute_scores(
        model, tokenizer, test_id, direction, layer, device
    )
    
    # Compute OOD test scores
    print("Computing OOD test scores...")
    ood_scores, ood_labels = compute_scores(
        model, tokenizer, test_ood, direction, layer, device
    )
    
    # Compute metrics
    def compute_metrics(scores, labels, threshold):
        preds = (scores > threshold).astype(int)
        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
    
    train_metrics = {"accuracy": train_acc, "auroc": float(roc_auc_score(train_labels, train_scores))}
    id_metrics = compute_metrics(id_scores, id_labels, threshold)
    ood_metrics = compute_metrics(ood_scores, ood_labels, threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Setting':<25} {'AUROC':<12} {'Accuracy':<12} {'F1':<12}")
    print("-" * 60)
    print(f"{'Train (ID)':<25} {train_metrics['auroc']:<12.4f} {train_metrics['accuracy']:<12.4f} {'--':<12}")
    print(f"{'Test (ID - political)':<25} {id_metrics['auroc']:<12.4f} {id_metrics['accuracy']:<12.4f} {id_metrics['f1']:<12.4f}")
    print(f"{'Test (OOD - nlp_survey)':<25} {ood_metrics['auroc']:<12.4f} {ood_metrics['accuracy']:<12.4f} {ood_metrics['f1']:<12.4f}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION (Addresses Reviewer 1)")
    print("=" * 60)
    if ood_metrics['auroc'] >= 0.80:
        print(f"✓ OOD AUROC = {ood_metrics['auroc']:.3f} >= 0.80")
        print("✓ Feature generalizes across domains → NOT a dataset artifact")
    elif ood_metrics['auroc'] >= 0.70:
        print(f"~ OOD AUROC = {ood_metrics['auroc']:.3f} (moderate generalization)")
        print("~ Partial evidence for generalization")
    else:
        print(f"✗ OOD AUROC = {ood_metrics['auroc']:.3f} < 0.70")
        print("✗ Poor generalization → Possible dataset confound")
    
    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "ood_generalization",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "layer": layer,
            "seed": seed,
            "train_domains": ["political", "philpapers"],
            "test_id_domain": "political",
            "test_ood_domain": "nlp_survey",
        },
        "results": {
            "train": train_metrics,
            "test_id": id_metrics,
            "test_ood": ood_metrics,
        },
        "interpretation": {
            "ood_generalizes": ood_metrics['auroc'] >= 0.80,
            "conclusion": "NOT a dataset artifact" if ood_metrics['auroc'] >= 0.80 else "Possible confound",
        },
    }
    
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_file = output_path / f"ood_eval_{model_short}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OOD Generalization Evaluation (Addresses Reviewer 1)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
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
