"""
Sycophancy Geometry Diagnostic
==============================

Proper diagnosis of sycophancy geometry following these scenarios:

1. Multi-dimensional: Is sycophancy multi-faceted? (within-class PCA)
2. Nonlinear boundary: Linear vs MLP probe comparison
3. Distributed across layers: Multi-layer probing
4. Multiple clusters: Cluster analysis per class

The goal is to understand WHY steering fails, not just that it fails.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Local imports
from utils import (
    set_seed, get_device, load_model_and_tokenizer,
    build_chat_prompt,
)

# Configuration
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES = 200
OUTPUT_DIR = Path("geometry_diagnostics")
OUTPUT_DIR.mkdir(exist_ok=True)

set_seed(SEED)
device = get_device()

print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

# =============================================================================
# SECTION 1: Data Loading
# =============================================================================
print("\n" + "="*60)
print("SECTION 1: Loading Data")
print("="*60)

from datasets import load_dataset

dataset = load_dataset(
    "Anthropic/model-written-evals",
    data_files="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    split="train",
)

def create_pairs(data, n_prompts):
    pairs = []
    for i, item in enumerate(data):
        if i >= n_prompts:
            break
        pairs.append({
            "question": item["question"],
            "syco_answer": item["answer_matching_behavior"],
            "honest_answer": item["answer_not_matching_behavior"],
        })
    return pairs

pairs = create_pairs(dataset, N_SAMPLES)
print(f"Loaded {len(pairs)} pairs")

# =============================================================================
# SECTION 2: Model Loading
# =============================================================================
print("\n" + "="*60)
print("SECTION 2: Loading Model")
print("="*60)

model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Model loaded: {n_layers} layers, hidden_dim={hidden_dim}")

# =============================================================================
# SECTION 3: Extract Activations from Multiple Layers
# =============================================================================
print("\n" + "="*60)
print("SECTION 3: Extract Multi-Layer Activations")
print("="*60)

def extract_multi_layer_activations(pairs, label_type, layers_to_extract):
    """Extract activations from multiple layers for multi-layer analysis."""
    layer_acts = {l: [] for l in layers_to_extract}
    
    for pair in tqdm(pairs, desc=f"Extracting {label_type}"):
        prompt = build_chat_prompt(tokenizer, pair["question"])
        answer = pair["syco_answer"] if label_type == "syco" else pair["honest_answer"]
        full_text = prompt + answer
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract LAST token (contains full context)
        for layer_idx in layers_to_extract:
            hidden = outputs.hidden_states[layer_idx + 1]
            act = hidden[0, -1, :].float().cpu().numpy()
            layer_acts[layer_idx].append(act)
    
    return {l: np.array(acts) for l, acts in layer_acts.items()}

# Extract from early, middle, late layers
LAYERS = [8, 16, 21, 24]  # Representative layers
print(f"Extracting from layers: {LAYERS}")

syco_acts = extract_multi_layer_activations(pairs, "syco", LAYERS)
honest_acts = extract_multi_layer_activations(pairs, "honest", LAYERS)

# =============================================================================
# SECTION 4: DIAGNOSTIC 1 - Within-Class Dimensionality
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 1: Within-Class Dimensionality")
print("="*60)

# This answers: Is sycophancy multi-faceted?
# If within-class variance requires many PCs → multi-dimensional concept

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

within_class_dims = {}
for idx, layer in enumerate(LAYERS):
    ax = axes.flatten()[idx]
    
    # PCA on sycophantic examples only
    syco_centered = syco_acts[layer] - syco_acts[layer].mean(axis=0)
    pca = PCA(n_components=min(50, len(syco_centered)))
    pca.fit(syco_centered)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find dims for 80%, 90%, 95%
    dim_80 = np.argmax(cumsum >= 0.80) + 1
    dim_90 = np.argmax(cumsum >= 0.90) + 1
    dim_95 = np.argmax(cumsum >= 0.95) + 1
    
    within_class_dims[layer] = {"80%": dim_80, "90%": dim_90, "95%": dim_95}
    
    ax.plot(range(1, len(cumsum)+1), cumsum, 'b-o', markersize=3)
    ax.axhline(y=0.80, color='g', linestyle='--', alpha=0.5, label='80%')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance')
    ax.set_title(f'Layer {layer}: Within-Class PCA\n(90%={dim_90} dims, 95%={dim_95} dims)')
    ax.set_xlim(0, 50)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("DIAGNOSTIC 1: Is Sycophancy Multi-Dimensional?\n(Higher dims = multi-faceted concept)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_within_class_dimensionality.png", dpi=150)
plt.close()

print("Within-class dimensionality:")
for layer, dims in within_class_dims.items():
    print(f"  Layer {layer}: 80%={dims['80%']}, 90%={dims['90%']}, 95%={dims['95%']} components")

# =============================================================================
# SECTION 5: DIAGNOSTIC 2 - Linear vs Nonlinear Probing
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 2: Linear vs Nonlinear Boundary")
print("="*60)

# This answers: Is the decision boundary curved?
# If MLP >> Linear, there's nonlinearity to exploit

linear_vs_mlp = {}

for layer in LAYERS:
    X = np.vstack([syco_acts[layer], honest_acts[layer]])
    y = np.array([1] * len(syco_acts[layer]) + [0] * len(honest_acts[layer]))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    
    # Linear probe
    linear = LogisticRegression(max_iter=1000, random_state=SEED)
    linear.fit(X_train, y_train)
    linear_acc = accuracy_score(y_test, linear.predict(X_test))
    
    # MLP probe (nonlinear)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=SEED)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    
    # RBF SVM (local nonlinearity)
    rbf_svm = SVC(kernel='rbf', random_state=SEED)
    rbf_svm.fit(X_train, y_train)
    rbf_acc = accuracy_score(y_test, rbf_svm.predict(X_test))
    
    linear_vs_mlp[layer] = {
        "linear": linear_acc,
        "mlp": mlp_acc,
        "rbf_svm": rbf_acc,
        "gap": mlp_acc - linear_acc
    }
    
    print(f"  Layer {layer}: Linear={linear_acc:.3f}, MLP={mlp_acc:.3f}, RBF-SVM={rbf_acc:.3f} (gap={mlp_acc-linear_acc:+.3f})")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(LAYERS))
width = 0.25

linear_accs = [linear_vs_mlp[l]["linear"] for l in LAYERS]
mlp_accs = [linear_vs_mlp[l]["mlp"] for l in LAYERS]
rbf_accs = [linear_vs_mlp[l]["rbf_svm"] for l in LAYERS]

ax.bar(x - width, linear_accs, width, label='Linear Probe', color='#3498db')
ax.bar(x, mlp_accs, width, label='MLP Probe', color='#e74c3c')
ax.bar(x + width, rbf_accs, width, label='RBF-SVM', color='#2ecc71')

ax.set_xlabel('Layer')
ax.set_ylabel('Accuracy')
ax.set_title('DIAGNOSTIC 2: Is the Boundary Curved?\n(MLP >> Linear → Nonlinear boundary)')
ax.set_xticks(x)
ax.set_xticklabels([f'L{l}' for l in LAYERS])
ax.legend()
ax.set_ylim(0.4, 1.05)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_linear_vs_nonlinear.png", dpi=150)
plt.close()

# =============================================================================
# SECTION 6: DIAGNOSTIC 3 - Multi-Layer Probing
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 3: Is Information Distributed Across Layers?")
print("="*60)

# Concatenate activations from all layers
X_multi = np.concatenate([
    np.vstack([syco_acts[l], honest_acts[l]])
    for l in LAYERS
], axis=-1)  # Shape: [N, hidden_dim * num_layers]

y = np.array([1] * len(pairs) + [0] * len(pairs))

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.3, random_state=SEED, stratify=y
)

# Multi-layer linear probe
multi_probe = LogisticRegression(max_iter=1000, random_state=SEED)
multi_probe.fit(X_train, y_train)
multi_acc = accuracy_score(y_test, multi_probe.predict(X_test))

# Compare with best single-layer
best_single = max(linear_vs_mlp[l]["linear"] for l in LAYERS)
best_layer = max(LAYERS, key=lambda l: linear_vs_mlp[l]["linear"])

print(f"  Best single-layer (L{best_layer}): {best_single:.3f}")
print(f"  Multi-layer concat: {multi_acc:.3f}")
print(f"  Improvement: {multi_acc - best_single:+.3f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
methods = [f'L{l}' for l in LAYERS] + ['Multi-Layer']
accs = [linear_vs_mlp[l]["linear"] for l in LAYERS] + [multi_acc]
colors = ['#3498db'] * len(LAYERS) + ['#e74c3c']

bars = ax.bar(methods, accs, color=colors)
ax.set_ylabel('Accuracy')
ax.set_title('DIAGNOSTIC 3: Is Information Distributed?\n(Multi >> Best Single → Distributed)')
ax.set_ylim(0.4, 1.05)
ax.grid(True, axis='y', alpha=0.3)

# Annotate improvement
ax.annotate(f'+{multi_acc - best_single:.3f}', 
            xy=(len(methods)-1, multi_acc), 
            xytext=(len(methods)-1, multi_acc + 0.03),
            ha='center', fontsize=10, color='darkred')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_multi_layer_probing.png", dpi=150)
plt.close()

# =============================================================================
# SECTION 7: DIAGNOSTIC 4 - Cluster Structure Per Class
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSTIC 4: Are There Multiple Types of Sycophancy?")
print("="*60)

# Cluster within each class
analysis_layer = 21
X_syco = syco_acts[analysis_layer]
X_honest = honest_acts[analysis_layer]

def find_best_k(X, max_k=6):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        scores.append(silhouette_score(X, labels))
    return scores

syco_scores = find_best_k(X_syco)
honest_scores = find_best_k(X_honest)

best_k_syco = np.argmax(syco_scores) + 2
best_k_honest = np.argmax(honest_scores) + 2

print(f"  Sycophantic: Best k={best_k_syco} (silhouette={syco_scores[best_k_syco-2]:.3f})")
print(f"  Honest: Best k={best_k_honest} (silhouette={honest_scores[best_k_honest-2]:.3f})")

# Interpretation
if syco_scores[best_k_syco-2] < 0.1:
    cluster_interpretation = "Weak clustering - no clear sycophancy types"
elif best_k_syco > 2:
    cluster_interpretation = f"Multiple sycophancy types ({best_k_syco} clusters)"
else:
    cluster_interpretation = "Single sycophancy mode"

print(f"  Interpretation: {cluster_interpretation}")

# =============================================================================
# SECTION 8: Summary and Recommendations
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: GEOMETRY DIAGNOSIS")
print("="*60)

# Compile results
diagnosis = {
    "within_class_dims": within_class_dims,
    "linear_vs_mlp": linear_vs_mlp,
    "multi_layer_improvement": multi_acc - best_single,
    "best_k_syco": int(best_k_syco),
    "silhouette_syco": float(syco_scores[best_k_syco-2]),
}

with open(OUTPUT_DIR / "diagnosis_results.json", "w") as f:
    json.dump(diagnosis, f, indent=2, default=float)

# Determine geometry type
avg_within_dim_90 = np.mean([within_class_dims[l]["90%"] for l in LAYERS])
max_gap = max(linear_vs_mlp[l]["gap"] for l in LAYERS)
multi_improvement = multi_acc - best_single

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    GEOMETRY DIAGNOSIS                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Within-class dimensionality (90%): ~{avg_within_dim_90:.0f} dimensions       │
│    → {'HIGH' if avg_within_dim_90 > 10 else 'LOW'}: Sycophancy is {'multi-faceted' if avg_within_dim_90 > 10 else 'compact'}                │
│                                                             │
│ 2. Linear vs MLP gap: max {max_gap:+.3f}                              │
│    → {'Nonlinear boundary detected' if max_gap > 0.05 else 'Boundary is approximately linear'}                       │
│                                                             │
│ 3. Multi-layer improvement: {multi_improvement:+.3f}                           │
│    → {'Information distributed across layers' if multi_improvement > 0.02 else 'Single layer sufficient'}             │
│                                                             │
│ 4. Cluster structure: k={best_k_syco}, silhouette={syco_scores[best_k_syco-2]:.3f}            │
│    → {cluster_interpretation}          │
└─────────────────────────────────────────────────────────────┘

RECOMMENDED INTERVENTION:
""")

# Determine recommended approach
if avg_within_dim_90 > 15:
    print("  1. Subspace projection (PLS with r=10-20) instead of single direction")
if max_gap > 0.05:
    print("  2. Learned steering vector (gradient-based) to handle nonlinearity")
if multi_improvement > 0.02:
    print("  3. Multi-layer intervention (intervene at multiple layers)")
if best_k_syco > 2 and syco_scores[best_k_syco-2] > 0.1:
    print("  4. Cluster-aware steering (different vectors per cluster)")

print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Diagnosis complete!")
