"""
Direction Similarity Analysis.

Computes cosine similarity between directions learned on different 
domains/concepts to verify:
1. Same concept, different domains → HIGH similarity (direction is consistent)
2. Different concepts → LOW similarity (directions are distinct)

Usage:
    python run_direction_sim.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from utils import (
    set_seed,
    get_device,
    load_model_and_tokenizer,
    load_jsonl,
)
from run_layer_sweep import (
    extract_activations,
    train_diffmean,
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


def compute_cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two directions."""
    d1_norm = d1 / (np.linalg.norm(d1) + 1e-8)
    d2_norm = d2 / (np.linalg.norm(d2) + 1e-8)
    return float(np.dot(d1_norm, d2_norm))


def train_direction(model, tokenizer, data, layer, device, name: str) -> np.ndarray:
    """Train diff-in-means direction on given data."""
    print(f"  Training {name} direction...")
    pos_acts, neg_acts = extract_activations(
        model, tokenizer, data, layer, device, desc=name
    )
    return train_diffmean(pos_acts, neg_acts)


def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer: int | None = None,
    train_size: int = 400,
    seed: int = 42,
    output_dir: str = "results",
):
    """
    Compute direction similarity matrix.
    """
    set_seed(seed)
    device = get_device()
    
    print("=" * 60)
    print("Direction Similarity Analysis")
    print("=" * 60)
    print(f"Model: {model_name}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    n_layers = model.config.num_hidden_layers
    
    if layer is None:
        layer = int(n_layers * 0.6)
    layer = min(layer, n_layers - 1)
    print(f"Layer: {layer} (of {n_layers})")
    
    # Define datasets to compare
    datasets = {
        "syco_political": ("sycophancy", "political"),
        "syco_philpapers": ("sycophancy", "philpapers"),
        "syco_nlp_survey": ("sycophancy", "nlp_survey"),
        "deception": ("deception", "truthfulqa"),
    }
    
    # Train directions for each dataset
    print("\nTraining directions...")
    directions = {}
    for name, (concept, setting) in datasets.items():
        try:
            data = load_concept_data(concept, setting, "train", seed=0, max_size=train_size)
            directions[name] = train_direction(model, tokenizer, data, layer, device, name)
            print(f"    {name}: {len(data)} instances")
        except FileNotFoundError as e:
            print(f"    {name}: SKIPPED ({e})")
    
    # Compute similarity matrix
    print("\n" + "=" * 60)
    print("DIRECTION SIMILARITY MATRIX")
    print("=" * 60)
    
    dir_names = list(directions.keys())
    n = len(dir_names)
    
    # Header
    header = f"{'':>20}" + "".join(f"{name:>18}" for name in dir_names)
    print(header)
    print("-" * len(header))
    
    # Similarity matrix
    similarity_matrix = {}
    for i, name1 in enumerate(dir_names):
        row = f"{name1:>20}"
        for j, name2 in enumerate(dir_names):
            sim = compute_cosine_similarity(directions[name1], directions[name2])
            similarity_matrix[f"{name1}_vs_{name2}"] = sim
            row += f"{sim:>18.4f}"
        print(row)
    
    # Key comparisons
    print("\n" + "=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)
    
    # Same concept, different domains
    if "syco_political" in directions and "syco_philpapers" in directions:
        same_concept_sim = compute_cosine_similarity(
            directions["syco_political"], 
            directions["syco_philpapers"]
        )
        print(f"Same concept (syco_political ↔ syco_philpapers): {same_concept_sim:.4f}")
        if same_concept_sim > 0.7:
            print("  ✓ High similarity → Direction is consistent across domains")
        else:
            print("  ~ Low similarity → Direction may vary across domains")
    
    if "syco_political" in directions and "syco_nlp_survey" in directions:
        same_concept_sim2 = compute_cosine_similarity(
            directions["syco_political"], 
            directions["syco_nlp_survey"]
        )
        print(f"Same concept (syco_political ↔ syco_nlp_survey): {same_concept_sim2:.4f}")
    
    # Different concepts
    if "syco_political" in directions and "deception" in directions:
        diff_concept_sim = compute_cosine_similarity(
            directions["syco_political"], 
            directions["deception"]
        )
        print(f"\nDifferent concepts (sycophancy ↔ deception): {diff_concept_sim:.4f}")
        if abs(diff_concept_sim) < 0.3:
            print("  ✓ Low similarity → Directions are distinct")
        elif abs(diff_concept_sim) < 0.5:
            print("  ~ Moderate similarity → Some shared structure")
        else:
            print("  ✗ High similarity → May be detecting same signal")
    
    # Save results
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "direction_similarity",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "layer": layer,
            "seed": seed,
            "train_size": train_size,
        },
        "datasets_used": list(directions.keys()),
        "similarity_matrix": similarity_matrix,
    }
    
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_file = output_path / f"direction_sim_{model_short}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direction Similarity Analysis"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        layer=args.layer,
        train_size=args.train_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
