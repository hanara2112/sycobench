"""
Unified detection benchmark script.

Supports both DiffMean and Linear Probe methods for detecting
sycophancy/deception from model activations.

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
from tqdm import tqdm

from utils import (
    set_seed,
    get_device,
    load_jsonl,
    load_model_and_tokenizer,
    build_chat_prompt,
    gather_residual_activations,
)


# =============================================================================
# Activation Extraction (adapted from run_layer_sweep.py)
# =============================================================================

def extract_activations_for_diffmean(
    model,
    tokenizer,
    data: list[dict],
    layer_idx: int,
    device: torch.device,
    desc: str = "Extracting",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract activations at the last PROMPT token for pos/neg instances (for DiffMean).
    
    We use the last prompt token (before the answer) because:
    - The answer tokens (A)/(B) are context-independent
    - The prompt's final token contains the model's 'decision state'
    """
    positive_activations = []
    negative_activations = []

    for item in tqdm(data, desc=desc, leave=False):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        full_text = chat_prompt + item["completion"]

        # Tokenize prompt only to find where it ends
        prompt_tokens = tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations = gather_residual_activations(model, layer_idx, inputs)

        # Use the LAST TOKEN of the PROMPT (before the answer)
        # This is where the model's "decision state" is encoded
        last_prompt_idx = prompt_len - 1
        if activations.dim() == 2:
            prompt_act = activations[last_prompt_idx, :]
        else:
            prompt_act = activations[0, last_prompt_idx, :]

        if item["label"] == 1:
            positive_activations.append(prompt_act.cpu())
        else:
            negative_activations.append(prompt_act.cpu())

    return positive_activations, negative_activations


def extract_activations_for_probe(
    model,
    tokenizer,
    data: list[dict],
    layer_idx: int,
    device: torch.device,
    desc: str = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract activations at the last PROMPT token for all instances (for Probe)."""
    activations_list = []
    labels_list = []

    for item in tqdm(data, desc=desc, leave=False):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        full_text = chat_prompt + item["completion"]

        # Tokenize prompt only to find where it ends
        prompt_tokens = tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations = gather_residual_activations(model, layer_idx, inputs)

        # Use the LAST TOKEN of the PROMPT
        last_prompt_idx = prompt_len - 1
        if activations.dim() == 2:
            prompt_act = activations[last_prompt_idx, :]
        else:
            prompt_act = activations[0, last_prompt_idx, :]

        activations_list.append(prompt_act.to(torch.float32).cpu().numpy())
        labels_list.append(item["label"])

    return np.array(activations_list), np.array(labels_list)


# =============================================================================
# DiffMean Method (from run_layer_sweep.py)
# =============================================================================

def train_diffmean(
    positive_activations: list[torch.Tensor],
    negative_activations: list[torch.Tensor],
) -> np.ndarray:
    """Train diff-in-means direction from last-token activations."""
    # Stack activations: each is [hidden_dim], so stack to [N, hidden_dim]
    all_positive = torch.stack(positive_activations, dim=0)
    all_negative = torch.stack(negative_activations, dim=0)

    mean_positive = all_positive.mean(dim=0)
    mean_negative = all_negative.mean(dim=0)

    direction = mean_positive - mean_negative
    norm = torch.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction.to(torch.float32).numpy()


def compute_diffmean_scores(
    model,
    tokenizer,
    data: list[dict],
    direction: np.ndarray,
    layer_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scores for instances using the direction vector on last PROMPT token."""
    # Keep direction on CPU to avoid multi-GPU device issues
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    scores = []
    labels = []

    for item in tqdm(data, desc="Scoring", leave=False):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        full_text = chat_prompt + item["completion"]

        # Tokenize prompt only to find where it ends
        prompt_tokens = tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # Tokenize full text
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations = gather_residual_activations(model, layer_idx, inputs)

        # Use the LAST TOKEN of the PROMPT
        last_prompt_idx = prompt_len - 1
        if activations.dim() == 2:
            prompt_act = activations[last_prompt_idx, :].to(torch.float32)
        else:
            prompt_act = activations[0, last_prompt_idx, :].to(torch.float32)

        # Move to CPU for dot product
        prompt_act = prompt_act.cpu()
        score = (prompt_act @ direction_tensor).item()
        scores.append(score)
        labels.append(item["label"])

    return np.array(scores), np.array(labels)


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find optimal threshold to maximize accuracy."""
    thresholds = np.unique(scores)
    thresholds = np.concatenate([[scores.min() - 1], thresholds, [scores.max() + 1]])

    best_threshold = 0.0
    best_acc = 0.0

    for t in thresholds:
        preds = (scores > t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


# =============================================================================
# Main Detection Function
# =============================================================================

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
    Run detection experiment.
    
    Args:
        model_name: HuggingFace model name
        train_file: Path to training JSONL
        test_file: Path to test JSONL
        method: Detection method ('diffmean' or 'probe')
        layer: Layer index to extract activations from
        seed: Random seed
        output_file: Optional CSV file to append results
    
    Returns:
        Dictionary of metrics
    """
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
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
    # Expected: data/<concept>/<setting>/seed_<N>/train.jsonl
    parts = train_path.parts
    try:
        concept = parts[-4]  # e.g., 'sycophancy' or 'deception'
        setting = parts[-3]  # e.g., 'political' or 'truthfulqa'
        seed_dir = parts[-2]  # e.g., 'seed_0'
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
    """Run DiffMean detection."""
    # Extract training activations
    print("Extracting training activations...")
    pos_acts, neg_acts = extract_activations_for_diffmean(
        model, tokenizer, train_data, layer, device, desc="Train"
    )

    # Train direction
    print("Training diff-in-means direction...")
    direction = train_diffmean(pos_acts, neg_acts)

    # Compute training scores for threshold
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
    """Run Linear Probe detection."""
    # Extract training activations
    print("Extracting training activations...")
    X_train, y_train = extract_activations_for_probe(
        model, tokenizer, train_data, layer, device, desc="Train"
    )

    # Extract test activations
    print("Extracting test activations...")
    X_test, y_test = extract_activations_for_probe(
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
