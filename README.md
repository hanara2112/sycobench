# Sycophancy Detection via Representation Engineering

Detection and mitigation of sycophantic behavior in LLMs using activation-based methods.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run layer sweep (find best layer)
python run_layer_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct --train-instances 200 --test-instances 100

# 3. Run detection benchmark
python run_detection.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train-file data/sycophancy/political/seed_0/train.jsonl \
    --test-file data/sycophancy/political/seed_0/test.jsonl \
    --method diffmean --layer 21
```

## Key Results

| Method                  | Test AUROC          | Notes              |
| ----------------------- | ------------------- | ------------------ |
| Random Baseline         | ~50%                | Sanity check       |
| LLM-as-Judge            | 0.69                | Black-box baseline |
| Linear Probe            | 0.88-0.96           | White-box          |
| **Diff-in-Means** | **0.90-0.98** | Best performing    |

---

## Experiments

### Core Detection

```bash
# Layer sweep - find best layer
python run_layer_sweep.py --model Qwen/Qwen2.5-3B-Instruct

# Linear probe
python run_linear_probe.py --model Qwen/Qwen2.5-3B-Instruct

# Unified detection benchmark
python run_detection.py \
    --train-file data/sycophancy/political/seed_0/train.jsonl \
    --test-file data/sycophancy/political/seed_0/test.jsonl \
    --method diffmean --layer 20
```

### Generalization Experiments

**OOD Generalization** - Train on multiple domains, test on held-out domain:

```bash
python run_ood_eval.py --model Qwen/Qwen2.5-3B-Instruct
# Train: political + philpapers → Test: nlp_survey
# Expected: AUROC > 0.80 → NOT a dataset artifact
```

**Cross-Concept Transfer** - Test if sycophancy ≠ deception:

```bash
python run_cross_concept.py --model Qwen/Qwen2.5-3B-Instruct
# Train sycophancy → Test deception (should FAIL)
# Expected: AUROC < 0.60 → Concept-specific direction
```

**Direction Similarity** - Verify directions are distinct:

```bash
python run_direction_sim.py --model Qwen/Qwen2.5-3B-Instruct
# Expected: cos(syco, deception) < 0.3
```

### Mitigation

```bash
# Steering experiment with alpha sweep
python run_steering.py --model Qwen/Qwen2.5-3B-Instruct --alpha-sweep
```

### Baselines

```bash
# LLM-as-judge baseline
python run_llm_judge_baseline.py --model Qwen/Qwen2.5-3B-Instruct

# Random baseline
python run_random_baseline.py

# Model sycophancy rate
python run_model_sycophancy_rate.py --model Qwen/Qwen2.5-3B-Instruct
```

---

## Project Structure

```
sycobench/
├── data_prep.py              # Dataset preparation
├── utils.py                  # Shared utilities
├── steering_utils.py         # Steering utilities
│
├── run_layer_sweep.py        # Diff-in-means layer sweep
├── run_linear_probe.py       # Linear probe experiment
├── run_detection.py          # Unified detection benchmark
│
├── run_ood_eval.py           # OOD generalization test
├── run_cross_concept.py      # Cross-concept transfer test
├── run_direction_sim.py      # Direction similarity analysis
│
├── run_steering.py           # Activation steering
├── run_llm_judge_baseline.py # LLM judge baseline
├── run_model_sycophancy_rate.py
├── run_random_baseline.py
│
├── data/                     # Datasets
│   ├── sycophancy/
│   │   ├── political/
│   │   ├── philpapers/
│   │   ├── nlp_survey/
│   │   └── cross_domain_political_philpapers_to_nlp_survey/
│   └── deception/
│       └── truthfulqa/
└── results/                  # Output
```

---

## Datasets

| Concept    | Setting      | Train | Test |
| ---------- | ------------ | ----- | ---- |
| Sycophancy | political    | 1000  | 1000 |
| Sycophancy | philpapers   | 1000  | 1000 |
| Sycophancy | nlp_survey   | 1000  | 1000 |
| Sycophancy | cross_domain | 1000  | 1000 |
| Deception  | truthfulqa   | 600   | 600  |

---

## Common Arguments

| Argument              | Description       | Default                      |
| --------------------- | ----------------- | ---------------------------- |
| `--model`           | HuggingFace model | `Qwen/Qwen2.5-3B-Instruct` |
| `--seed`            | Random seed       | 42                           |
| `--train-instances` | Training size     | 2000                         |
| `--test-instances`  | Test size         | 1000                         |
| `--layer`           | Layer index       | Auto-select                  |
| `--output-dir`      | Results directory | `results`                  |

### Supported Models

- Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B}-Instruct
- google/gemma-2-{2b,9b}-it

---

## Methods

### Diff-in-Means

1. Extract activations for sycophantic (label=1) and non-sycophantic (label=0) responses
2. Compute direction: `v = mean(h | syco) - mean(h | non-syco)`
3. Score by projection: `score = h · v`
4. Classify using optimal threshold

### Linear Probe

Logistic regression on mean-pooled activations.

### Activation Steering

Subtract or clamp along the sycophancy direction during generation.

---

## Citation

```bibtex
@software{sycobench,
  title = {Sycophancy Detection via Representation Engineering},
  author = {Vamshi Krishna Bonagiri, Aryaman Bahl},
  year = {2026},
  url = {https://github.com/hanara2112/sycobench}
}
```

## References

- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [Towards Understanding Sycophancy](https://arxiv.org/abs/2310.13548) - Sharma et al., 2023
- [Anthropic Model-Written Evals](https://huggingface.co/datasets/Anthropic/model-written-evals)
