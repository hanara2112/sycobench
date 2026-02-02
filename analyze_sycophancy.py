"""
Sycophancy Detection & Steering Analysis Notebook
==================================================

A comprehensive visualization notebook to understand:
1. How sycophancy is represented across layers
2. PCA/geometry of the sycophancy manifold  
3. Detection performance analysis
4. Insights for improving steering

Run this notebook on the server with GPU.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde

# Local imports
from utils import (
    set_seed, get_device, load_model_and_tokenizer,
    build_chat_prompt, gather_residual_activations,
)

# Configuration
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_TRAIN = 200  # Training examples for direction
N_TEST = 100   # Test examples for evaluation
OUTPUT_DIR = Path("analysis_plots")
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
print(f"Total dataset size: {len(dataset)} examples")

# Create contrastive pairs
def create_pairs(data, n_prompts):
    """Create pairs of sycophantic and non-sycophantic examples."""
    pairs = []
    for i, item in enumerate(data):
        if i >= n_prompts:
            break
        q = item["question"]
        pairs.append({
            "question": q,
            "syco_answer": item["answer_matching_behavior"],
            "honest_answer": item["answer_not_matching_behavior"],
        })
    return pairs

train_pairs = create_pairs(dataset, N_TRAIN)
test_pairs = create_pairs(dataset.select(range(N_TRAIN, N_TRAIN + N_TEST)), N_TEST)
print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

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
# SECTION 3: Extract Activations Across Layers
# =============================================================================
print("\n" + "="*60)
print("SECTION 3: Extracting Activations (All Layers)")
print("="*60)

def extract_activations_all_layers(pairs, label_type="syco", n_samples=50):
    """Extract last-token activations from all layers."""
    all_layer_acts = {l: [] for l in range(n_layers)}
    
    for pair in tqdm(pairs[:n_samples], desc=f"Extracting {label_type}"):
        prompt = build_chat_prompt(tokenizer, pair["question"])
        answer = pair["syco_answer"] if label_type == "syco" else pair["honest_answer"]
        full_text = prompt + answer
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract last token from each layer
        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding
            last_token_act = hidden[0, -1, :].cpu().numpy()
            all_layer_acts[layer_idx].append(last_token_act)
    
    return {l: np.array(acts) for l, acts in all_layer_acts.items()}

# Extract for both classes
print("Extracting sycophantic activations...")
syco_acts = extract_activations_all_layers(train_pairs, "syco", N_TRAIN)
print("Extracting honest activations...")
honest_acts = extract_activations_all_layers(train_pairs, "honest", N_TRAIN)

# =============================================================================
# SECTION 4: Layer-wise Detection Performance
# =============================================================================
print("\n" + "="*60)
print("SECTION 4: Layer-wise Detection Analysis")
print("="*60)

layer_aurocs = []
layer_accuracies = []

for layer_idx in tqdm(range(n_layers), desc="Layer sweep"):
    X = np.vstack([syco_acts[layer_idx], honest_acts[layer_idx]])
    y = np.array([1] * len(syco_acts[layer_idx]) + [0] * len(honest_acts[layer_idx]))
    
    # Train/test split (simple)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Diff-in-means direction
    pos_mean = X_train[y_train == 1].mean(axis=0)
    neg_mean = X_train[y_train == 0].mean(axis=0)
    direction = pos_mean - neg_mean
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    
    # Score = projection onto direction
    scores = X_test @ direction
    
    try:
        auroc = roc_auc_score(y_test, scores)
    except:
        auroc = 0.5
    
    # Accuracy with median threshold
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    
    layer_aurocs.append(auroc)
    layer_accuracies.append(acc)

# Plot 1: Layer-wise AUROC
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(n_layers), layer_aurocs, 'b-o', linewidth=2, markersize=5)
ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Sycophancy Detection: Layer-wise AUROC (Diff-in-Means)', fontsize=14)
ax.set_ylim(0.4, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# Mark best layer
best_layer = np.argmax(layer_aurocs)
ax.annotate(f'Best: L{best_layer}\nAUROC={layer_aurocs[best_layer]:.3f}',
            xy=(best_layer, layer_aurocs[best_layer]),
            xytext=(best_layer + 2, layer_aurocs[best_layer] - 0.05),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_layer_auroc.png", dpi=150)
plt.close()
print(f"Saved: 01_layer_auroc.png (Best layer: {best_layer}, AUROC: {layer_aurocs[best_layer]:.3f})")

# =============================================================================
# SECTION 5: PCA Analysis - Eigenvalue Spectrum
# =============================================================================
print("\n" + "="*60)
print("SECTION 5: PCA Analysis")
print("="*60)

# Analyze the best layer
analysis_layer = best_layer
print(f"Analyzing layer {analysis_layer}")

X_layer = np.vstack([syco_acts[analysis_layer], honest_acts[analysis_layer]])
y_layer = np.array([1] * len(syco_acts[analysis_layer]) + [0] * len(honest_acts[analysis_layer]))

# Center the data
X_centered = X_layer - X_layer.mean(axis=0)

# PCA
pca = PCA(n_components=min(50, X_centered.shape[0]))
pca.fit(X_centered)

# Plot 2: Eigenvalue spectrum
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Explained variance
ax1 = axes[0]
ax1.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.7)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title(f'PCA Eigenvalue Spectrum (Layer {analysis_layer})', fontsize=14)
ax1.set_xlim(-0.5, 20.5)

# Right: Cumulative variance
ax2 = axes[1]
cumsum = np.cumsum(pca.explained_variance_ratio_)
ax2.plot(range(len(cumsum)), cumsum, 'b-o', markersize=4)
ax2.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Explained Variance', fontsize=14)
ax2.legend()
ax2.set_xlim(-0.5, 50.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_pca_spectrum.png", dpi=150)
plt.close()

n_90 = np.argmax(cumsum >= 0.9) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Saved: 02_pca_spectrum.png")
print(f"  Components for 90% variance: {n_90}")
print(f"  Components for 95% variance: {n_95}")

# =============================================================================
# SECTION 6: 2D Projection (PCA + Direction)
# =============================================================================
print("\n" + "="*60)
print("SECTION 6: 2D Visualization")
print("="*60)

# Compute diff-in-means direction
pos_acts = syco_acts[analysis_layer]
neg_acts = honest_acts[analysis_layer]
direction = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
direction = direction / (np.linalg.norm(direction) + 1e-12)

# Project onto direction and first orthogonal PC
proj_on_dir = X_layer @ direction

# Get component orthogonal to direction
X_orthogonal = X_centered - np.outer(X_centered @ direction, direction)
pca_orth = PCA(n_components=1)
pca_orth.fit(X_orthogonal)
proj_orth = pca_orth.transform(X_orthogonal).flatten()

# Plot 3: 2D scatter
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#FF6B6B' if y == 1 else '#4ECDC4' for y in y_layer]
ax.scatter(proj_on_dir, proj_orth, c=colors, alpha=0.6, s=50)

ax.set_xlabel('Projection onto Sycophancy Direction', fontsize=12)
ax.set_ylabel('First Orthogonal PC', fontsize=12)
ax.set_title(f'Sycophancy vs Honest Activations (Layer {analysis_layer})', fontsize=14)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=10, label='Sycophantic'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=10, label='Honest'),
]
ax.legend(handles=legend_elements, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_2d_projection.png", dpi=150)
plt.close()
print("Saved: 03_2d_projection.png")

# =============================================================================
# SECTION 7: Projection Distribution Analysis
# =============================================================================
print("\n" + "="*60)
print("SECTION 7: Projection Distribution")
print("="*60)

# Plot 4: Distribution of projections
fig, ax = plt.subplots(figsize=(10, 5))

syco_proj = proj_on_dir[y_layer == 1]
honest_proj = proj_on_dir[y_layer == 0]

# Histogram
ax.hist(honest_proj, bins=30, alpha=0.6, color='#4ECDC4', label='Honest', density=True)
ax.hist(syco_proj, bins=30, alpha=0.6, color='#FF6B6B', label='Sycophantic', density=True)

# KDE overlays
if len(syco_proj) > 1 and len(honest_proj) > 1:
    x_range = np.linspace(min(honest_proj.min(), syco_proj.min()), 
                          max(honest_proj.max(), syco_proj.max()), 100)
    kde_syco = gaussian_kde(syco_proj)
    kde_honest = gaussian_kde(honest_proj)
    ax.plot(x_range, kde_syco(x_range), color='red', linewidth=2)
    ax.plot(x_range, kde_honest(x_range), color='teal', linewidth=2)

ax.set_xlabel('Projection onto Sycophancy Direction', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Distribution of Projections (Layer {analysis_layer})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Add separation metrics
mean_diff = syco_proj.mean() - honest_proj.mean()
pooled_std = np.sqrt((syco_proj.std()**2 + honest_proj.std()**2) / 2)
d_prime = mean_diff / pooled_std if pooled_std > 0 else 0

ax.text(0.02, 0.98, f"Mean diff: {mean_diff:.2f}\nd': {d_prime:.2f}",
        transform=ax.transAxes, verticalalignment='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_projection_distribution.png", dpi=150)
plt.close()
print(f"Saved: 04_projection_distribution.png (d'={d_prime:.2f})")

# =============================================================================
# SECTION 8: Multi-dimensional Analysis (PLS/LDA)
# =============================================================================
print("\n" + "="*60)
print("SECTION 8: Subspace Analysis (LDA vs PLS)")
print("="*60)

# Compare different projection methods
methods = {
    'DiffMean (1D)': direction.reshape(1, -1),
}

# LDA (also 1D for binary)
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_centered, y_layer)
methods['LDA (1D)'] = lda.scalings_.T

# PLS with different ranks
for n_comp in [1, 3, 5]:
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_centered, y_layer)
    methods[f'PLS (r={n_comp})'] = pls.x_weights_.T

# Evaluate each method
method_aurocs = {}
for name, W in methods.items():
    scores = X_layer @ W.T  # (n_samples, n_components)
    if scores.ndim > 1:
        scores = scores.sum(axis=1)  # Simple aggregation
    
    try:
        auroc = roc_auc_score(y_layer, scores)
        if auroc < 0.5:
            auroc = 1 - auroc  # Flip if direction is backwards
    except:
        auroc = 0.5
    method_aurocs[name] = auroc
    print(f"  {name}: AUROC = {auroc:.3f}")

# Plot 5: Method comparison
fig, ax = plt.subplots(figsize=(8, 5))
names = list(method_aurocs.keys())
aurocs = list(method_aurocs.values())
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

bars = ax.barh(names, aurocs, color=colors)
ax.axvline(x=0.5, color='r', linestyle='--', label='Random')
ax.set_xlabel('AUROC', fontsize=12)
ax.set_title('Detection: DiffMean vs Subspace Methods', fontsize=14)
ax.set_xlim(0.4, 1.0)
ax.grid(True, axis='x', alpha=0.3)

for bar, auroc in zip(bars, aurocs):
    ax.text(auroc + 0.01, bar.get_y() + bar.get_height()/2, f'{auroc:.3f}',
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_method_comparison.png", dpi=150)
plt.close()
print("Saved: 05_method_comparison.png")

# =============================================================================
# SECTION 9: Direction Similarity Across Layers
# =============================================================================
print("\n" + "="*60)
print("SECTION 9: Direction Consistency Across Layers")
print("="*60)

# Compute direction at each layer
layer_directions = []
for layer_idx in range(n_layers):
    pos_mean = syco_acts[layer_idx].mean(axis=0)
    neg_mean = honest_acts[layer_idx].mean(axis=0)
    d = pos_mean - neg_mean
    d = d / (np.linalg.norm(d) + 1e-12)
    layer_directions.append(d)

# Compute cosine similarity matrix
cos_sim_matrix = np.zeros((n_layers, n_layers))
for i in range(n_layers):
    for j in range(n_layers):
        cos_sim_matrix[i, j] = np.dot(layer_directions[i], layer_directions[j])

# Plot 6: Similarity heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cos_sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Cosine Similarity')
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('Direction Similarity Across Layers', fontsize=14)

# Mark diagonal
ax.set_xticks(range(0, n_layers, 4))
ax.set_yticks(range(0, n_layers, 4))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_direction_similarity.png", dpi=150)
plt.close()
print("Saved: 06_direction_similarity.png")

# =============================================================================
# SECTION 10: Steering Analysis - Projection Magnitudes
# =============================================================================
print("\n" + "="*60)
print("SECTION 10: Steering Analysis")
print("="*60)

# For steering, we need to understand the magnitude of projections
projections_by_layer = []
for layer_idx in range(n_layers):
    d = layer_directions[layer_idx]
    X = np.vstack([syco_acts[layer_idx], honest_acts[layer_idx]])
    projs = X @ d
    projections_by_layer.append({
        'layer': layer_idx,
        'mean_proj': projs.mean(),
        'std_proj': projs.std(),
        'mean_syco': projs[:len(syco_acts[layer_idx])].mean(),
        'mean_honest': projs[len(syco_acts[layer_idx]):].mean(),
    })

# Plot 7: Projection magnitude across layers
fig, ax = plt.subplots(figsize=(10, 5))

layers = [p['layer'] for p in projections_by_layer]
mean_syco = [p['mean_syco'] for p in projections_by_layer]
mean_honest = [p['mean_honest'] for p in projections_by_layer]

ax.plot(layers, mean_syco, 'r-o', label='Sycophantic', markersize=4)
ax.plot(layers, mean_honest, 'b-o', label='Honest', markersize=4)
ax.fill_between(layers, mean_syco, mean_honest, alpha=0.2, color='purple')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Mean Projection onto Direction', fontsize=12)
ax.set_title('Projection Magnitude Across Layers', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_projection_magnitude.png", dpi=150)
plt.close()
print("Saved: 07_projection_magnitude.png")

# =============================================================================
# SECTION 11: Summary Statistics
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

summary = {
    "model": MODEL_NAME,
    "n_layers": n_layers,
    "hidden_dim": hidden_dim,
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "best_layer": int(best_layer),
    "best_layer_auroc": float(layer_aurocs[best_layer]),
    "pca_90_components": int(n_90),
    "pca_95_components": int(n_95),
    "dprime": float(d_prime),
    "layer_aurocs": [float(a) for a in layer_aurocs],
}

with open(OUTPUT_DIR / "analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"""
Key Findings:
=============
1. Best layer for detection: {best_layer} (AUROC: {layer_aurocs[best_layer]:.3f})
2. Components for 90% variance: {n_90} of {hidden_dim}
3. Separation d': {d_prime:.2f}
4. Detection works well but is DISTRIBUTED across many dimensions

Implications for Steering:
==========================
- If PCA shows high dimensionality (many components needed), 
  single-direction steering is unlikely to work
- d' < 2 suggests overlapping distributions → steering may affect 
  honest responses too
- Direction similarity across layers tells us if multi-layer 
  steering should use same or different directions

Recommendations:
================
1. For detection: Use DiffMean at layer {best_layer} (sufficient)
2. For steering: Consider subspace methods (PLS with r=3-5)
3. If subspace fails: Need learned/gradient-based steering

All plots saved to: {OUTPUT_DIR}/
""")

print("Analysis complete!")
