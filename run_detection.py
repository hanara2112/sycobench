"""
Unified detection benchmark script.

Imports working functions from run_layer_sweep.py and run_linear_probe.py

Usage:
    python run_detection.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --train-file data/sycophancy/political/seed_0/train.jsonl \
        --test-file data/sycophancy/political/seed_0/test.jsonl \
        --method diffmean \
        --layer 20 \
        --output results.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import (
    set_seed,
    get_device,
    load_jsonl,
    load_model_and_tokenizer,
)

from run_layer_sweep import (
    extract_activations as extract_activations_diffmean,
    train_diffmean,
    compute_scores as compute_diffmean_scores,
    find_optimal_threshold,
)
from run_linear_probe import (
    extract_activations as extract_activations_probe,
)


def run_detection(
    model_name: str,
    train_file: str,
    test_file: str,
    method: str,
    layer: int,
    seed: int = 42,
    output_file: str | None = None,
) -> dict:
    """
    Run detection experiment using the original working functions.
    """
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load data from JSONL files
    print(f"Loading train data: {train_file}")
    train_data = load_jsonl(train_file)
    print(f"Loading test data: {test_file}")
    test_data = load_jsonl(test_file)
    print(f"Train: {len(train_data)} instances, Test: {len(test_data)} instances")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers, using layer {layer}")

    if layer >= n_layers or layer < 0:
        raise ValueError(f"Layer {layer} out of range [0, {n_layers})")

    # Run detection based on method
    if method == "diffmean":
        metrics = run_diffmean_detection(
            model, tokenizer, train_data, test_data, layer, device
        )
    elif method == "probe":
        metrics = run_probe_detection(
            model, tokenizer, train_data, test_data, layer, device, seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Print results
    print("\n" + "=" * 60)
    print(f"DETECTION RESULTS ({method.upper()})")
    print("=" * 60)
    print(f"  AUROC:    {metrics['auroc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print("=" * 60)

    # Parse metadata from file path
    train_path = Path(train_file)
    parts = train_path.parts
    try:
        concept = parts[-4]
        setting = parts[-3]
        seed_dir = parts[-2]
        seed_num = int(seed_dir.split("_")[1])
    except (IndexError, ValueError):
        concept = "unknown"
        setting = "unknown"
        seed_num = seed

    # Append to CSV if output file specified
    if output_file:
        write_results_csv(
            output_file,
            concept=concept,
            setting=setting,
            method=method,
            model=model_name.split("/")[-1],
            layer=layer,
            seed=seed_num,
            auroc=metrics["auroc"],
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
        )
        print(f"Results appended to {output_file}")

    return metrics


def run_diffmean_detection(
    model,
    tokenizer,
    train_data: list[dict],
    test_data: list[dict],
    layer: int,
    device: torch.device,
) -> dict:
    """Run DiffMean detection using functions from run_layer_sweep.py."""
    # Extract training activations (using the original function)
    print("Extracting training activations...")
    pos_acts, neg_acts = extract_activations_diffmean(
        model, tokenizer, train_data, layer, device, desc="Train"
    )

    # Train direction (using the original function)
    print("Training diff-in-means direction...")
    direction = train_diffmean(pos_acts, neg_acts)

    # Compute training scores for threshold (using the original function)
    print("Computing training scores...")
    train_scores, train_labels = compute_diffmean_scores(
        model, tokenizer, train_data, direction, layer, device
    )
    threshold, train_acc = find_optimal_threshold(train_scores, train_labels)
    print(f"Train accuracy: {train_acc:.4f}, threshold: {threshold:.4f}")

    # Compute test scores
    print("Computing test scores...")
    test_scores, test_labels = compute_diffmean_scores(
        model, tokenizer, test_data, direction, layer, device
    )

    # Compute metrics
    y_pred = (test_scores > threshold).astype(int)
    auroc = roc_auc_score(test_labels, test_scores)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    # Cleanup
    del pos_acts, neg_acts, train_scores, test_scores
    torch.cuda.empty_cache()

    return {"auroc": auroc, "accuracy": accuracy, "f1": f1}


def run_probe_detection(
    model,
    tokenizer,
    train_data: list[dict],
    test_data: list[dict],
    layer: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Run Linear Probe detection using functions from run_linear_probe.py."""
    # Extract training activations (using the original function)
    print("Extracting training activations...")
    X_train, y_train = extract_activations_probe(
        model, tokenizer, train_data, layer, device, desc="Train"
    )

    # Extract test activations
    print("Extracting test activations...")
    X_test, y_test = extract_activations_probe(
        model, tokenizer, test_data, layer, device, desc="Test"
    )

    # Train logistic regression
    print("Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    print(f"Train accuracy: {train_acc:.4f}")

    # Compute test predictions and scores
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1]

    # Compute metrics
    auroc = roc_auc_score(y_test, y_scores)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Cleanup
    del X_train, X_test, y_train, y_test
    torch.cuda.empty_cache()

    return {"auroc": auroc, "accuracy": accuracy, "f1": f1}


def write_results_csv(
    output_file: str,
    concept: str,
    setting: str,
    method: str,
    model: str,
    layer: int,
    seed: int,
    auroc: float,
    accuracy: float,
    f1: float,
):
    """Append results to CSV file."""
    file_exists = os.path.exists(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "concept", "setting", "method", "model", "layer", "seed",
                "auroc", "accuracy", "f1"
            ])
        writer.writerow([
            concept, setting, method, model, layer, seed,
            f"{auroc:.4f}", f"{accuracy:.4f}", f"{f1:.4f}"
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified detection benchmark for sycophancy/deception"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffmean", "probe"],
        default="diffmean",
        help="Detection method",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer to extract activations from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV file for results",
    )

    args = parser.parse_args()

    run_detection(
        model_name=args.model,
        train_file=args.train_file,
        test_file=args.test_file,
        method=args.method,
        layer=args.layer,
        seed=args.seed,
        output_file=args.output,
    )
