"""
Linear probe experiment for sycophancy detection.

Trains logistic regression classifiers on activations from each layer
to detect sycophantic responses.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
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
    desc: str = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract last-token activations and labels for all instances.
    
    Uses last token representation (the model's decision point) rather than
    mean-pooling over all tokens, which provides richer sycophancy signal.
    """
    activations_list = []
    labels_list = []

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

        # Use last token representation (model's decision point)
        if activations.dim() == 2:
            item_activations = activations[-1, :]
        else:
            item_activations = activations[0, -1, :]

        activations_list.append(item_activations.cpu().numpy())
        labels_list.append(item["label"])

    return np.array(activations_list), np.array(labels_list)


def main(
    seed: int = 42,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layers: list[int] | None = None,
    train_instances: int = 2000,
    test_instances: int = 1000,
    output_dir: str = "results",
):
    """
    Run linear probe experiment.

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

    # Calculate prompts needed
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
        layers = [
            n_layers // 4,
            n_layers // 2,
            3 * n_layers // 4,
            n_layers - 1,
        ]
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
        X_train, y_train = extract_activations(
            model, tokenizer, train_data, layer_idx, device,
            desc=f"Train L{layer_idx}"
        )

        # Extract test activations
        print("Extracting test activations...")
        X_test, y_test = extract_activations(
            model, tokenizer, test_data, layer_idx, device,
            desc=f"Test L{layer_idx}"
        )

        # Train logistic regression
        print("Training logistic regression...")
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)

        # Compute test predictions and scores
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:, 1]

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auroc = roc_auc_score(y_test, y_scores)

        print(f"\nLayer {layer_idx} Results:")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Test Acc:  {accuracy:.4f}")
        print(f"  Test F1:   {f1:.4f}")
        print(f"  Test AUROC: {auroc:.4f}")

        results.append({
            "layer": layer_idx,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_auroc": float(auroc),
        })

        # Clean up
        del X_train, X_test, y_train, y_test
        torch.cuda.empty_cache()

    # Find best layer
    best_result = max(results, key=lambda x: x["test_auroc"])
    best_layer = best_result["layer"]

    # Print summary
    print("\n" + "=" * 80)
    print("LINEAR PROBE SUMMARY")
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
        "experiment": "linear_probe",
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
    results_file = output_path / f"linear_probe_{model_short}_results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe for sycophancy detection")
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
        help="Layers to test (default: 25%, 50%, 75%, last)",
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
