"""
Layer sweep experiment for diff-in-means sycophancy detection.

Tests which transformer layer works best for detecting sycophancy using
the diff-in-means approach from representation engineering.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

from utils import (
    set_seed,
    get_device,
    load_model_and_tokenizer,
    build_chat_prompt,
    gather_residual_activations,
    create_contrastive_instances,
)


def extract_activations(
    model,
    tokenizer,
    data: list[dict],
    layer_idx: int,
    device: torch.device,
    prefix_length: int = 1,
    desc: str = "Extracting",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract mean-pooled activations for positive and negative instances.
    
    Each instance's activations are MEAN-POOLED to a single vector before
    being returned. This prevents dilution when computing class means.
    """
    positive_activations = []
    negative_activations = []

    for item in tqdm(data, desc=desc, leave=False):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        full_text = chat_prompt + item["completion"]

        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations = gather_residual_activations(model, layer_idx, inputs)

        if activations.dim() == 2:
            seq_len = activations.shape[0]
            item_activations = activations[prefix_length:seq_len, :]
        else:
            seq_len = activations.shape[1]
            item_activations = activations[0, prefix_length:seq_len, :]

        # CRITICAL: Mean-pool per instance to get a single representation vector
        # This prevents signal dilution when computing class means
        instance_repr = item_activations.mean(dim=0)  # Shape: (hidden_dim,)

        if item["label"] == 1:
            positive_activations.append(instance_repr.cpu())
        else:
            negative_activations.append(instance_repr.cpu())

    return positive_activations, negative_activations


def train_diffmean(
    positive_activations: list[torch.Tensor],
    negative_activations: list[torch.Tensor],
) -> np.ndarray:
    """
    Train diff-in-means direction.
    
    Input: Lists of mean-pooled per-instance activation vectors.
    Output: Normalized direction vector (positive - negative).
    """
    # Stack per-instance representations (each already mean-pooled)
    all_positive = torch.stack(positive_activations, dim=0)  # (n_pos, hidden_dim)
    all_negative = torch.stack(negative_activations, dim=0)  # (n_neg, hidden_dim)

    # Mean over instances
    mean_positive = all_positive.mean(dim=0)
    mean_negative = all_negative.mean(dim=0)

    direction = mean_positive - mean_negative
    norm = torch.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction.to(torch.float32).numpy()


def compute_scores(
    model,
    tokenizer,
    data: list[dict],
    direction: np.ndarray,
    layer_idx: int,
    device: torch.device,
    prefix_length: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scores for instances using the direction vector."""
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

    scores = []
    labels = []

    for item in tqdm(data, desc="Scoring", leave=False):
        chat_prompt = build_chat_prompt(tokenizer, item["prompt"])
        full_text = chat_prompt + item["completion"]

        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations = gather_residual_activations(model, layer_idx, inputs)

        if activations.dim() == 2:
            seq_len = activations.shape[0]
            item_activations = activations[prefix_length:seq_len, :].to(torch.float32)
        else:
            seq_len = activations.shape[1]
            item_activations = activations[0, prefix_length:seq_len, :].to(torch.float32)

        token_scores = item_activations @ direction_tensor
        mean_score = token_scores.mean().item()  # Use mean for stability across seeds
        scores.append(mean_score)
        labels.append(item["label"])

    return np.array(scores), np.array(labels)


def find_optimal_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
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


def main(
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layers: list[int] | None = None,
    train_instances: int = 2000,
    test_instances: int = 1000,
    output_dir: str = "results",
):
    """
    Run layer sweep experiment.

    Args:
        seed: Random seed
        model_name: HuggingFace model name
        layers: List of layers to test (if None, tests key layers)
        train_instances: Number of training instances
        test_instances: Number of test instances
        output_dir: Output directory
    """
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    print("Loading Anthropic/model-written-evals dataset...")
    dataset = load_dataset(
        "Anthropic/model-written-evals",
        data_files="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        split="train",
    )
    print(f"Total dataset size: {len(dataset)} examples")

    dataset = dataset.shuffle(seed=seed)

    # Calculate prompts needed (each prompt creates 2 instances)
    test_prompts = test_instances // 2
    train_prompts = train_instances // 2
    total_prompts_needed = test_prompts + train_prompts

    if total_prompts_needed > len(dataset):
        raise ValueError(
            f"Need {total_prompts_needed} prompts but dataset only has {len(dataset)}"
        )

    # Create train/test splits
    test_examples = dataset.select(range(test_prompts))
    train_examples = dataset.select(range(test_prompts, test_prompts + train_prompts))

    test_data = create_contrastive_instances(test_examples, "test")
    train_data = create_contrastive_instances(train_examples, "train")

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} instances ({train_prompts} prompts)")
    print(f"  Test: {len(test_data)} instances ({test_prompts} prompts)")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Determine layers to test
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")

    if layers is None:
        # Test every 4th layer plus key layers
        layers = list(range(0, n_layers, 4))
        for l in [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            if l not in layers:
                layers.append(l)
        layers = sorted(set(layers))

    layers = [l for l in layers if 0 <= l < n_layers]
    print(f"Testing layers: {layers}")

    # Results storage
    results = []

    for layer_idx in layers:
        print(f"\n{'=' * 60}")
        print(f"Layer {layer_idx}")
        print(f"{'=' * 60}")

        # Extract training activations
        print("Extracting training activations...")
        pos_acts, neg_acts = extract_activations(
            model, tokenizer, train_data, layer_idx, device,
            desc=f"Train L{layer_idx}"
        )

        # Train direction
        print("Training diff-in-means direction...")
        direction = train_diffmean(pos_acts, neg_acts)

        # Compute training scores for threshold
        print("Computing training scores...")
        train_scores, train_labels = compute_scores(
            model, tokenizer, train_data, direction, layer_idx, device
        )
        threshold, train_acc = find_optimal_threshold(train_scores, train_labels)

        # Compute test scores
        print("Computing test scores...")
        test_scores, test_labels = compute_scores(
            model, tokenizer, test_data, direction, layer_idx, device
        )

        # Compute metrics
        y_pred = (test_scores > threshold).astype(int)
        y_true = test_labels

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auroc = roc_auc_score(y_true, test_scores)

        print(f"\nLayer {layer_idx} Results:")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Test Acc:  {accuracy:.4f}")
        print(f"  Test F1:   {f1:.4f}")
        print(f"  Test AUROC: {auroc:.4f}")

        results.append({
            "layer": layer_idx,
            "threshold": float(threshold),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_auroc": float(auroc),
        })

        # Clean up
        del pos_acts, neg_acts, train_scores, test_scores
        torch.cuda.empty_cache()

    # Find best layer
    best_result = max(results, key=lambda x: x["test_auroc"])
    best_layer = best_result["layer"]

    # Print summary
    print("\n" + "=" * 80)
    print("LAYER SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'Layer':<8} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUROC':<12}")
    print("-" * 56)
    for r in results:
        marker = " <-- BEST" if r["layer"] == best_layer else ""
        print(
            f"{r['layer']:<8} {r['train_accuracy']:<12.4f} "
            f"{r['test_accuracy']:<12.4f} {r['test_f1']:<12.4f} "
            f"{r['test_auroc']:<12.4f}{marker}"
        )

    print(f"\nBest layer: {best_layer} (AUROC = {best_result['test_auroc']:.4f})")

    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    full_results = {
        "experiment": "layer_sweep",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": seed,
            "model_name": model_name,
            "n_layers": n_layers,
            "layers_tested": layers,
            "train_instances": len(train_data),
            "test_instances": len(test_data),
        },
        "best_layer": best_layer,
        "best_auroc": best_result["test_auroc"],
        "results": results,
    }

    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_file = output_path / f"layer_sweep_{model_short}_results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer sweep for diff-in-means")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to test (default: auto-select)",
    )
    parser.add_argument("--train-instances", type=int, default=2000)
    parser.add_argument("--test-instances", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()
    main(
        seed=args.seed,
        model_name=args.model,
        layers=args.layers,
        train_instances=args.train_instances,
        test_instances=args.test_instances,
        output_dir=args.output_dir,
    )
