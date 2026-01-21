"""
Data preparation script for sycophancy and deception detection experiments.

Downloads and processes:
- Anthropic/model-written-evals sycophancy datasets
- TruthfulQA dataset for deception detection

Creates balanced train/test splits with contrastive pairs.
Supports multi-seed splits and cross-domain evaluation.
"""

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset

from utils import set_seed, save_jsonl


# Available dataset configurations
SYCOPHANCY_DATASETS = {
    "political": "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "philpapers": "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "nlp_survey": "sycophancy/sycophancy_on_nlp_survey.jsonl",
}

# TruthfulQA limits (dataset has ~817 examples)
TRUTHFULQA_MAX_TRAIN = 300
TRUTHFULQA_MAX_TEST = 300
TRUTHFULQA_MIN_ANSWER_LENGTH = 20


def validate_no_overlap(train_prompts: list[str], test_prompts: list[str]):
    """Assert no overlap between train and test prompts."""
    train_set = set(train_prompts)
    test_set = set(test_prompts)
    overlap = train_set & test_set
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping prompts between train and test!"
    print(f"✓ Sanity check passed: 0 overlapping prompts between train and test")


def filter_truthfulqa_example(example: dict) -> bool:
    """
    Filter TruthfulQA examples to keep only high-quality deception pairs.
    
    Keeps examples where:
    - best_answer exists and is non-empty
    - at least one incorrect_answer exists
    - incorrect_answer is >= 20 chars (not obvious nonsense)
    """
    best_answer = example.get("best_answer", "")
    incorrect_answers = example.get("incorrect_answers", [])
    
    if not best_answer or not best_answer.strip():
        return False
    
    if not incorrect_answers:
        return False
    
    # Check if at least one incorrect answer is substantive
    if isinstance(incorrect_answers, str):
        incorrect_answers = [incorrect_answers]
    
    for ans in incorrect_answers:
        if ans and len(ans.strip()) >= TRUTHFULQA_MIN_ANSWER_LENGTH:
            return True
    
    return False


def prepare_sycophancy(
    dataset_name: str = "political",
    seed: int = 42,
    n_train_prompts: int = 500,
    n_test_prompts: int = 500,
    output_dir: str = "data",
    seed_suffix: str = None,
):
    """
    Prepare a sycophancy dataset.

    Creates train/test splits with contrastive pairs:
    - For each prompt, creates two instances:
      - (prompt, answer_matching_behavior) -> label=1 (sycophantic)
      - (prompt, answer_not_matching_behavior) -> label=0 (not sycophantic)

    Args:
        dataset_name: One of 'political', 'philpapers', 'nlp_survey'
        seed: Random seed for reproducibility
        n_train_prompts: Number of unique prompts for training
        n_test_prompts: Number of unique prompts for testing
        output_dir: Directory to save output files
        seed_suffix: Optional suffix for seed-based directories (e.g., "seed_0")
    """
    set_seed(seed)

    if dataset_name not in SYCOPHANCY_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(SYCOPHANCY_DATASETS.keys())}")

    data_file = SYCOPHANCY_DATASETS[dataset_name]
    print(f"Loading Anthropic/model-written-evals ({dataset_name})...")
    
    dataset = load_dataset(
        "Anthropic/model-written-evals",
        data_files=data_file,
        split="train",
    )

    print(f"Dataset size: {len(dataset)} examples")

    total_needed = n_train_prompts + n_test_prompts
    if total_needed > len(dataset):
        print(f"Warning: Requested {total_needed} prompts but dataset only has {len(dataset)}. Using all available.")
        n_train_prompts = int(len(dataset) * 0.6)
        n_test_prompts = int(len(dataset) * 0.4)

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_prompts = dataset.select(range(n_train_prompts))
    test_prompts = dataset.select(range(n_train_prompts, n_train_prompts + n_test_prompts))

    print(f"Train prompts: {len(train_prompts)}")
    print(f"Test prompts: {len(test_prompts)}")

    def create_instances(prompts, split_name: str) -> list[dict]:
        """Create contrastive pairs from prompts."""
        instances = []
        for idx, example in enumerate(prompts):
            base_id = f"{split_name}_{idx:04d}"

            # Sycophantic (matching behavior)
            instances.append({
                "id": f"{base_id}_syco",
                "prompt": example["question"],
                "completion": example["answer_matching_behavior"],
                "label": 1,
                "concept": "sycophancy",
                "answer_matching_behavior": example["answer_matching_behavior"],
                "answer_not_matching_behavior": example["answer_not_matching_behavior"],
            })

            # Non-sycophantic (not matching behavior)
            instances.append({
                "id": f"{base_id}_norm",
                "prompt": example["question"],
                "completion": example["answer_not_matching_behavior"],
                "label": 0,
                "concept": "sycophancy",
                "answer_matching_behavior": example["answer_matching_behavior"],
                "answer_not_matching_behavior": example["answer_not_matching_behavior"],
            })

        return instances

    train_instances = create_instances(train_prompts, "train")
    test_instances = create_instances(test_prompts, "test")

    # Sanity check: no overlap
    train_prompt_texts = [ex["question"] for ex in train_prompts]
    test_prompt_texts = [ex["question"] for ex in test_prompts]
    validate_no_overlap(train_prompt_texts, test_prompt_texts)

    # Create prompt-only file for sycophancy rate evaluation
    test_prompts_only = []
    for idx, example in enumerate(test_prompts):
        test_prompts_only.append({
            "id": f"test_{idx:04d}",
            "prompt": example["question"],
            "answer_matching_behavior": example["answer_matching_behavior"],
            "answer_not_matching_behavior": example["answer_not_matching_behavior"],
        })

    # Verify balance
    train_pos = sum(1 for x in train_instances if x["label"] == 1)
    train_neg = sum(1 for x in train_instances if x["label"] == 0)
    test_pos = sum(1 for x in test_instances if x["label"] == 1)
    test_neg = sum(1 for x in test_instances if x["label"] == 0)

    print(f"\nTrain instances: {len(train_instances)} (pos: {train_pos}, neg: {train_neg})")
    print(f"Test instances: {len(test_instances)} (pos: {test_pos}, neg: {test_neg})")
    print(f"Test prompts (for sycophancy rate): {len(test_prompts_only)}")

    # Save files
    if seed_suffix:
        output_path = Path(__file__).parent / output_dir / "sycophancy" / dataset_name / seed_suffix
    else:
        output_path = Path(__file__).parent / output_dir / "sycophancy" / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    test_file = output_path / "test.jsonl"
    prompts_file = output_path / "test_prompts.jsonl"

    save_jsonl(train_instances, train_file)
    save_jsonl(test_instances, test_file)
    save_jsonl(test_prompts_only, prompts_file)

    print(f"\nSaved files:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    print(f"  - {prompts_file}")

    return train_instances, test_instances


def prepare_sycophancy_cross_domain(
    seed: int = 42,
    n_train_prompts: int = 500,
    n_test_prompts: int = 500,
    output_dir: str = "data",
    seed_suffix: str = None,
    train_domains: list[str] = ["political", "philpapers"],
    test_domains: list[str] = ["nlp_survey"],
):
    """
    Prepare cross-domain sycophancy dataset for OOD generalization.
    
    Train on some domains, test on different domains.
    """
    set_seed(seed)
    
    print(f"Cross-domain split: Train on {train_domains}, Test on {test_domains}")
    
    # Load and combine train domains
    all_train_data = []
    for domain in train_domains:
        data_file = SYCOPHANCY_DATASETS[domain]
        dataset = load_dataset(
            "Anthropic/model-written-evals",
            data_files=data_file,
            split="train",
        )
        dataset = dataset.shuffle(seed=seed)
        all_train_data.extend(list(dataset))
    
    # Load and combine test domains
    all_test_data = []
    for domain in test_domains:
        data_file = SYCOPHANCY_DATASETS[domain]
        dataset = load_dataset(
            "Anthropic/model-written-evals",
            data_files=data_file,
            split="train",
        )
        dataset = dataset.shuffle(seed=seed)
        all_test_data.extend(list(dataset))
    
    # Sample from combined data
    import random
    random.seed(seed)
    random.shuffle(all_train_data)
    random.shuffle(all_test_data)
    
    # First, sample train data
    train_data = all_train_data[:n_train_prompts]
    train_questions = set(ex["question"] for ex in train_data)
    
    # Filter test data to remove any prompts that overlap with train
    # This can happen because different sycophancy datasets may share some questions
    original_test_size = len(all_test_data)
    all_test_data = [ex for ex in all_test_data if ex["question"] not in train_questions]
    filtered_count = original_test_size - len(all_test_data)
    if filtered_count > 0:
        print(f"⚠ Filtered {filtered_count} overlapping prompts from test domain")
    
    test_data = all_test_data[:n_test_prompts]
    
    print(f"Train prompts: {len(train_data)} from {train_domains}")
    print(f"Test prompts: {len(test_data)} from {test_domains}")
    
    def create_instances(data, split_name: str) -> list[dict]:
        """Create contrastive pairs from prompts."""
        instances = []
        for idx, example in enumerate(data):
            base_id = f"{split_name}_{idx:04d}"

            # Sycophantic (matching behavior)
            instances.append({
                "id": f"{base_id}_syco",
                "prompt": example["question"],
                "completion": example["answer_matching_behavior"],
                "label": 1,
                "concept": "sycophancy",
                "answer_matching_behavior": example["answer_matching_behavior"],
                "answer_not_matching_behavior": example["answer_not_matching_behavior"],
            })

            # Non-sycophantic (not matching behavior)
            instances.append({
                "id": f"{base_id}_norm",
                "prompt": example["question"],
                "completion": example["answer_not_matching_behavior"],
                "label": 0,
                "concept": "sycophancy",
                "answer_matching_behavior": example["answer_matching_behavior"],
                "answer_not_matching_behavior": example["answer_not_matching_behavior"],
            })

        return instances

    train_instances = create_instances(train_data, "train")
    test_instances = create_instances(test_data, "test")

    # Sanity check: no overlap
    train_prompt_texts = [ex["question"] for ex in train_data]
    test_prompt_texts = [ex["question"] for ex in test_data]
    validate_no_overlap(train_prompt_texts, test_prompt_texts)

    # Verify balance
    train_pos = sum(1 for x in train_instances if x["label"] == 1)
    train_neg = sum(1 for x in train_instances if x["label"] == 0)
    test_pos = sum(1 for x in test_instances if x["label"] == 1)
    test_neg = sum(1 for x in test_instances if x["label"] == 0)

    print(f"\nTrain instances: {len(train_instances)} (pos: {train_pos}, neg: {train_neg})")
    print(f"Test instances: {len(test_instances)} (pos: {test_pos}, neg: {test_neg})")

    # Save files
    train_str = "_".join(train_domains)
    test_str = "_".join(test_domains)
    if seed_suffix:
        output_path = Path(__file__).parent / output_dir / "sycophancy" / f"cross_domain_{train_str}_to_{test_str}" / seed_suffix
    else:
        output_path = Path(__file__).parent / output_dir / "sycophancy" / f"cross_domain_{train_str}_to_{test_str}"
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    test_file = output_path / "test.jsonl"

    save_jsonl(train_instances, train_file)
    save_jsonl(test_instances, test_file)

    print(f"\nSaved files:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")

    return train_instances, test_instances


def prepare_truthfulqa(
    seed: int = 42,
    n_train_prompts: int = TRUTHFULQA_MAX_TRAIN,
    n_test_prompts: int = TRUTHFULQA_MAX_TEST,
    output_dir: str = "data",
    seed_suffix: str = None,
):
    """
    Prepare the TruthfulQA dataset for deception detection.

    Creates train/test splits with contrastive pairs:
    - For each question, creates two instances:
      - (question, best_answer) -> label=0 (truthful/not deceptive)
      - (question, incorrect_answer) -> label=1 (deceptive)

    Note: This is more accurately "truthfulness/factuality deviation"
    rather than strategic deception, since we don't know if the model
    knows the truth.

    Args:
        seed: Random seed for reproducibility
        n_train_prompts: Number of unique prompts for training (max 300)
        n_test_prompts: Number of unique prompts for testing (max 300)
        output_dir: Directory to save output files
        seed_suffix: Optional suffix for seed-based directories
    """
    set_seed(seed)

    # Cap at maximum values
    n_train_prompts = min(n_train_prompts, TRUTHFULQA_MAX_TRAIN)
    n_test_prompts = min(n_test_prompts, TRUTHFULQA_MAX_TEST)

    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    print(f"Dataset size: {len(dataset)} examples")

    # Filter dataset for quality
    print("Filtering TruthfulQA for high-quality examples...")
    filtered_indices = [i for i, ex in enumerate(dataset) if filter_truthfulqa_example(ex)]
    dataset = dataset.select(filtered_indices)
    print(f"After filtering: {len(dataset)} examples (removed {len(filtered_indices)} low-quality)")

    total_needed = n_train_prompts + n_test_prompts
    if total_needed > len(dataset):
        print(f"Warning: Requested {total_needed} prompts but dataset only has {len(dataset)}. Adjusting.")
        n_train_prompts = min(n_train_prompts, int(len(dataset) * 0.5))
        n_test_prompts = min(n_test_prompts, len(dataset) - n_train_prompts)

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_data = dataset.select(range(n_train_prompts))
    test_data = dataset.select(range(n_train_prompts, n_train_prompts + n_test_prompts))

    print(f"Train prompts: {len(train_data)}")
    print(f"Test prompts: {len(test_data)}")

    def create_instances(data, split_name: str) -> list[dict]:
        """Create contrastive pairs from TruthfulQA data."""
        instances = []
        for idx, example in enumerate(data):
            base_id = f"{split_name}_{idx:04d}"
            question = example["question"]
            best_answer = example["best_answer"]
            
            # Get incorrect answers - handle both string and list formats
            incorrect_answers = example.get("incorrect_answers", [])
            if isinstance(incorrect_answers, str):
                incorrect_answers = [incorrect_answers]
            
            # Find the first substantive incorrect answer
            incorrect_answer = None
            for ans in incorrect_answers:
                if ans and len(ans.strip()) >= TRUTHFULQA_MIN_ANSWER_LENGTH:
                    incorrect_answer = ans
                    break
            
            if not incorrect_answer:
                continue

            # Truthful answer (not deceptive)
            instances.append({
                "id": f"{base_id}_truth",
                "prompt": question,
                "completion": best_answer,
                "label": 0,
                "concept": "deception",
                "category": example.get("category", "unknown"),
            })

            # Deceptive answer
            instances.append({
                "id": f"{base_id}_decep",
                "prompt": question,
                "completion": incorrect_answer,
                "label": 1,
                "concept": "deception",
                "category": example.get("category", "unknown"),
            })

        return instances

    train_instances = create_instances(train_data, "train")
    test_instances = create_instances(test_data, "test")

    # Sanity check: no overlap
    train_prompt_texts = [ex["question"] for ex in train_data]
    test_prompt_texts = [ex["question"] for ex in test_data]
    validate_no_overlap(train_prompt_texts, test_prompt_texts)

    # Verify balance
    train_pos = sum(1 for x in train_instances if x["label"] == 1)
    train_neg = sum(1 for x in train_instances if x["label"] == 0)
    test_pos = sum(1 for x in test_instances if x["label"] == 1)
    test_neg = sum(1 for x in test_instances if x["label"] == 0)

    print(f"\nTrain instances: {len(train_instances)} (deceptive: {train_pos}, truthful: {train_neg})")
    print(f"Test instances: {len(test_instances)} (deceptive: {test_pos}, truthful: {test_neg})")

    # Save files
    if seed_suffix:
        output_path = Path(__file__).parent / output_dir / "deception" / "truthfulqa" / seed_suffix
    else:
        output_path = Path(__file__).parent / output_dir / "deception" / "truthfulqa"
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    test_file = output_path / "test.jsonl"

    save_jsonl(train_instances, train_file)
    save_jsonl(test_instances, test_file)

    print(f"\nSaved files:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")

    return train_instances, test_instances


def prepare_all_datasets(
    seed: int = 42,
    num_seeds: int = 1,
    n_train_prompts: int = 500,
    n_test_prompts: int = 500,
    output_dir: str = "data",
    syco_mode: str = "in_domain",
):
    """Prepare all available datasets with multi-seed support."""
    print("=" * 60)
    print("Preparing all datasets...")
    print(f"Number of seeds: {num_seeds}")
    print(f"Sycophancy mode: {syco_mode}")
    print("=" * 60)

    for seed_i in range(num_seeds):
        current_seed = seed + seed_i
        seed_suffix = f"seed_{seed_i}" if num_seeds > 1 else None
        
        print(f"\n{'#' * 60}")
        print(f"# SEED {seed_i} (seed={current_seed})")
        print(f"{'#' * 60}")

        if syco_mode == "in_domain":
            # Prepare all sycophancy datasets independently
            for dataset_name in SYCOPHANCY_DATASETS:
                print(f"\n{'=' * 60}")
                print(f"Processing sycophancy dataset: {dataset_name}")
                print("=" * 60)
                prepare_sycophancy(
                    dataset_name=dataset_name,
                    seed=current_seed,
                    n_train_prompts=n_train_prompts,
                    n_test_prompts=n_test_prompts,
                    output_dir=output_dir,
                    seed_suffix=seed_suffix,
                )
        elif syco_mode == "cross_domain":
            # Cross-domain: train on political+philpapers, test on nlp_survey
            print(f"\n{'=' * 60}")
            print("Processing cross-domain sycophancy")
            print("=" * 60)
            prepare_sycophancy_cross_domain(
                seed=current_seed,
                n_train_prompts=n_train_prompts,
                n_test_prompts=n_test_prompts,
                output_dir=output_dir,
                seed_suffix=seed_suffix,
                train_domains=["political", "philpapers"],
                test_domains=["nlp_survey"],
            )

        # Prepare TruthfulQA (capped at 300/300)
        print(f"\n{'=' * 60}")
        print("Processing TruthfulQA dataset")
        print("=" * 60)
        prepare_truthfulqa(
            seed=current_seed,
            n_train_prompts=TRUTHFULQA_MAX_TRAIN,
            n_test_prompts=TRUTHFULQA_MAX_TEST,
            output_dir=output_dir,
            seed_suffix=seed_suffix,
        )

    print(f"\n{'=' * 60}")
    print("All datasets prepared successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sycophancy and deception datasets")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--num-seeds", type=int, default=3, help="Number of different random seeds for splits"
    )
    parser.add_argument(
        "--n-train", type=int, default=500, help="Number of train prompts per dataset"
    )
    parser.add_argument(
        "--n-test", type=int, default=500, help="Number of test prompts per dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "political", "philpapers", "nlp_survey", "truthfulqa"],
        help="Which dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--syco-mode",
        type=str,
        default="in_domain",
        choices=["in_domain", "cross_domain"],
        help="Sycophancy evaluation mode: in_domain (same domain train/test) or cross_domain (OOD generalization)",
    )

    args = parser.parse_args()

    if args.dataset == "all":
        prepare_all_datasets(
            seed=args.seed,
            num_seeds=args.num_seeds,
            n_train_prompts=args.n_train,
            n_test_prompts=args.n_test,
            output_dir=args.output_dir,
            syco_mode=args.syco_mode,
        )
    elif args.dataset == "truthfulqa":
        for seed_i in range(args.num_seeds):
            current_seed = args.seed + seed_i
            seed_suffix = f"seed_{seed_i}" if args.num_seeds > 1 else None
            prepare_truthfulqa(
                seed=current_seed,
                n_train_prompts=args.n_train,
                n_test_prompts=args.n_test,
                output_dir=args.output_dir,
                seed_suffix=seed_suffix,
            )
    else:
        for seed_i in range(args.num_seeds):
            current_seed = args.seed + seed_i
            seed_suffix = f"seed_{seed_i}" if args.num_seeds > 1 else None
            prepare_sycophancy(
                dataset_name=args.dataset,
                seed=current_seed,
                n_train_prompts=args.n_train,
                n_test_prompts=args.n_test,
                output_dir=args.output_dir,
                seed_suffix=seed_suffix,
            )
