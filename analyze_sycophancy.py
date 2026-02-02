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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform

# Optional: UMAP for visualization
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: umap-learn not installed. UMAP visualization will be skipped.")

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
    """
    Extract PROMPT's last-token activations from all layers.
    
    We extract at the prompt's last token (model's decision point) to capture
    the CONCEPT of sycophancy, not the trivial token differences in answers.
    """
    all_layer_acts = {l: [] for l in range(n_layers)}
    
    for pair in tqdm(pairs[:n_samples], desc=f"Extracting {label_type}"):
        # Build prompt only (without the answer)
        prompt = build_chat_prompt(tokenizer, pair["question"])
        answer = pair["syco_answer"] if label_type == "syco" else pair["honest_answer"]
        
        # Tokenize full text to simulate complete context
        full_text = prompt + answer
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        
        # Find where the prompt ends (position of last prompt token)
        prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract at PROMPT's last token (model's decision point, before answer)
        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding
            decision_token_act = hidden[0, prompt_length - 1, :].float().cpu().numpy()
            all_layer_acts[layer_idx].append(decision_token_act)
    
    return {l: np.array(acts) for l, acts in all_layer_acts.items()}

# Extract for both classes
print("Extracting sycophantic activations (at prompt's last token)...")
syco_acts = extract_activations_all_layers(train_pairs, "syco", N_TRAIN)
print("Extracting honest activations (at prompt's last token)...")
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
    
    # Train/test split (stratified to ensure both classes appear)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    
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

# KDE overlays (only if data has variance)
if len(syco_proj) > 1 and len(honest_proj) > 1 and syco_proj.std() > 1e-10 and honest_proj.std() > 1e-10:
    try:
        x_range = np.linspace(min(honest_proj.min(), syco_proj.min()), 
                              max(honest_proj.max(), syco_proj.max()), 100)
        kde_syco = gaussian_kde(syco_proj)
        kde_honest = gaussian_kde(honest_proj)
        ax.plot(x_range, kde_syco(x_range), color='red', linewidth=2)
        ax.plot(x_range, kde_honest(x_range), color='teal', linewidth=2)
    except Exception as e:
        print(f"  KDE skipped (low variance): {e}")

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
# SECTION 11: Intrinsic Dimensionality Estimation
# =============================================================================
print("\n" + "="*60)
print("SECTION 11: Intrinsic Dimensionality")
print("="*60)

def mle_intrinsic_dim(X, k=10):
    """
    Maximum Likelihood Estimator for intrinsic dimensionality (Levina-Bickel).
    Uses k-nearest neighbor distances.
    """
    from scipy.spatial.distance import cdist
    
    n = X.shape[0]
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    
    # Sort to get k nearest neighbors
    sorted_dists = np.sort(distances, axis=1)[:, :k]
    
    # MLE estimate
    log_ratios = []
    for i in range(n):
        dists = sorted_dists[i, :]
        if dists[-1] > 0:
            log_ratio = np.log(dists[-1] / dists[:-1]).mean()
            if log_ratio > 0:
                log_ratios.append(1.0 / log_ratio)
    
    return np.mean(log_ratios) if log_ratios else 0

def two_nn_intrinsic_dim(X):
    """
    Two-NN estimator for intrinsic dimensionality.
    Uses ratio of 2nd to 1st nearest neighbor distances.
    """
    from scipy.spatial.distance import cdist
    
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    sorted_dists = np.sort(distances, axis=1)
    
    # Ratio of 2nd to 1st nearest neighbor
    mu = sorted_dists[:, 1] / (sorted_dists[:, 0] + 1e-12)
    mu = mu[mu > 1]  # Filter invalid ratios
    
    if len(mu) == 0:
        return 0
    
    # MLE: d = n * log(mu) / sum(log(mu))
    log_mu = np.log(mu)
    d_estimate = len(mu) / np.sum(log_mu)
    
    return d_estimate

# Estimate intrinsic dimensionality for each class
X_syco = syco_acts[analysis_layer]
X_honest = honest_acts[analysis_layer]

print("Computing intrinsic dimensionality...")

# MLE estimator
id_mle_syco = mle_intrinsic_dim(X_syco[:100], k=5)  # Use subset for speed
id_mle_honest = mle_intrinsic_dim(X_honest[:100], k=5)
id_mle_combined = mle_intrinsic_dim(X_layer[:200], k=5)

# Two-NN estimator  
id_2nn_syco = two_nn_intrinsic_dim(X_syco[:100])
id_2nn_honest = two_nn_intrinsic_dim(X_honest[:100])
id_2nn_combined = two_nn_intrinsic_dim(X_layer[:200])

print(f"  MLE estimator - Syco: {id_mle_syco:.1f}, Honest: {id_mle_honest:.1f}, Combined: {id_mle_combined:.1f}")
print(f"  Two-NN estimator - Syco: {id_2nn_syco:.1f}, Honest: {id_2nn_honest:.1f}, Combined: {id_2nn_combined:.1f}")

# Plot 8: Intrinsic dimensionality comparison
fig, ax = plt.subplots(figsize=(8, 5))

categories = ['Sycophantic', 'Honest', 'Combined']
mle_dims = [id_mle_syco, id_mle_honest, id_mle_combined]
twonn_dims = [id_2nn_syco, id_2nn_honest, id_2nn_combined]

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, mle_dims, width, label='MLE (Levina-Bickel)', color='#3498db')
ax.bar(x + width/2, twonn_dims, width, label='Two-NN', color='#e74c3c')

ax.set_xlabel('Data Subset', fontsize=12)
ax.set_ylabel('Estimated Intrinsic Dimension', fontsize=12)
ax.set_title(f'Intrinsic Dimensionality Estimation (Layer {analysis_layer})', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Add text annotation
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1D (single direction)')
ax.text(2.5, 1, '← 1D', fontsize=9, va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_intrinsic_dimensionality.png", dpi=150)
plt.close()
print("Saved: 08_intrinsic_dimensionality.png")

# =============================================================================
# SECTION 12: UMAP / t-SNE Visualization
# =============================================================================
print("\n" + "="*60)
print("SECTION 12: Manifold Visualization (UMAP/t-SNE)")
print("="*60)

# Use t-SNE (always available) and optionally UMAP
fig, axes = plt.subplots(1, 2 if HAS_UMAP else 1, figsize=(12 if HAS_UMAP else 8, 6))
if not HAS_UMAP:
    axes = [axes]

# t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
X_tsne = tsne.fit_transform(X_layer)

ax = axes[0]
for label, color, name in [(1, '#FF6B6B', 'Sycophantic'), (0, '#4ECDC4', 'Honest')]:
    mask = y_layer == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, alpha=0.6, s=40, label=name)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f't-SNE Visualization (Layer {analysis_layer})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# UMAP if available
if HAS_UMAP:
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
    X_umap = reducer.fit_transform(X_layer)
    
    ax = axes[1]
    for label, color, name in [(1, '#FF6B6B', 'Sycophantic'), (0, '#4ECDC4', 'Honest')]:
        mask = y_layer == label
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, alpha=0.6, s=40, label=name)
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'UMAP Visualization (Layer {analysis_layer})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_manifold_visualization.png", dpi=150)
plt.close()
print("Saved: 09_manifold_visualization.png")

# =============================================================================
# SECTION 13: Cluster Analysis
# =============================================================================
print("\n" + "="*60)
print("SECTION 13: Cluster Analysis")
print("="*60)

# Is sycophancy one cluster or multiple?
# Test on each class separately

def analyze_clusters(X, name, max_k=6):
    """Analyze clustering structure using K-means with silhouette scores."""
    silhouettes = []
    inertias = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
        inertias.append(kmeans.inertia_)
    
    return silhouettes, inertias

print("Analyzing sycophantic cluster structure...")
syco_sil, syco_inertia = analyze_clusters(X_syco, "Syco")
print("Analyzing honest cluster structure...")
honest_sil, honest_inertia = analyze_clusters(X_honest, "Honest")

# Plot 10: Cluster analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Silhouette scores
ax1 = axes[0]
ks = range(2, 7)
ax1.plot(ks, syco_sil, 'r-o', label='Sycophantic', markersize=6)
ax1.plot(ks, honest_sil, 'b-o', label='Honest', markersize=6)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Silhouette Score', fontsize=12)
ax1.set_title('Cluster Quality by K', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add interpretation
best_k_syco = np.argmax(syco_sil) + 2
best_k_honest = np.argmax(honest_sil) + 2
ax1.axvline(x=best_k_syco, color='r', linestyle='--', alpha=0.5)
ax1.axvline(x=best_k_honest, color='b', linestyle='--', alpha=0.5)

# Elbow plot
ax2 = axes[1]
ax2.plot(ks, syco_inertia, 'r-o', label='Sycophantic', markersize=6)
ax2.plot(ks, honest_inertia, 'b-o', label='Honest', markersize=6)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Inertia (Within-cluster SS)', fontsize=12)
ax2.set_title('Elbow Plot', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_cluster_analysis.png", dpi=150)
plt.close()
print(f"Saved: 10_cluster_analysis.png")
print(f"  Best k for sycophantic: {best_k_syco} (silhouette={syco_sil[best_k_syco-2]:.3f})")
print(f"  Best k for honest: {best_k_honest} (silhouette={honest_sil[best_k_honest-2]:.3f})")

# =============================================================================
# SECTION 14: Per-class Cluster Visualization
# =============================================================================
print("\n" + "="*60)
print("SECTION 14: Per-class Cluster Structure")
print("="*60)

# Visualize clusters in t-SNE space
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sycophantic clusters
kmeans_syco = KMeans(n_clusters=best_k_syco, random_state=SEED, n_init=10)
syco_labels = kmeans_syco.fit_predict(X_syco)

# Project to 2D
tsne_syco = TSNE(n_components=2, perplexity=min(30, len(X_syco)-1), random_state=SEED)
X_syco_2d = tsne_syco.fit_transform(X_syco[:100])  # Subset for speed

ax = axes[0]
scatter = ax.scatter(X_syco_2d[:, 0], X_syco_2d[:, 1], c=syco_labels[:100], cmap='Set1', alpha=0.7, s=50)
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f'Sycophantic: {best_k_syco} Clusters', fontsize=14)
ax.grid(True, alpha=0.3)

# Honest clusters
kmeans_honest = KMeans(n_clusters=best_k_honest, random_state=SEED, n_init=10)
honest_labels = kmeans_honest.fit_predict(X_honest)

tsne_honest = TSNE(n_components=2, perplexity=min(30, len(X_honest)-1), random_state=SEED)
X_honest_2d = tsne_honest.fit_transform(X_honest[:100])

ax = axes[1]
scatter = ax.scatter(X_honest_2d[:, 0], X_honest_2d[:, 1], c=honest_labels[:100], cmap='Set1', alpha=0.7, s=50)
ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f'Honest: {best_k_honest} Clusters', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_per_class_clusters.png", dpi=150)
plt.close()
print("Saved: 11_per_class_clusters.png")

# =============================================================================
# SECTION 15: Summary Statistics
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
    "intrinsic_dim_mle_syco": float(id_mle_syco),
    "intrinsic_dim_mle_honest": float(id_mle_honest),
    "intrinsic_dim_2nn_syco": float(id_2nn_syco),
    "intrinsic_dim_2nn_honest": float(id_2nn_honest),
    "best_k_syco": int(best_k_syco),
    "best_k_honest": int(best_k_honest),
    "layer_aurocs": [float(a) for a in layer_aurocs],
}

with open(OUTPUT_DIR / "analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"""
Key Geometric Findings:
=======================
1. Best layer for detection: {best_layer} (AUROC: {layer_aurocs[best_layer]:.3f})
2. Separation d': {d_prime:.2f}

PCA Analysis:
-------------
- Components for 90% variance: {n_90} of {hidden_dim}
- Components for 95% variance: {n_95} of {hidden_dim}

Intrinsic Dimensionality:
-------------------------
- MLE: Syco={id_mle_syco:.1f}, Honest={id_mle_honest:.1f}
- Two-NN: Syco={id_2nn_syco:.1f}, Honest={id_2nn_honest:.1f}
- If ID >> 1, sycophancy is multi-dimensional → single-direction steering fails

Cluster Structure:
------------------
- Sycophantic: {best_k_syco} clusters (silhouette={syco_sil[best_k_syco-2]:.3f})
- Honest: {best_k_honest} clusters (silhouette={honest_sil[best_k_honest-2]:.3f})
- Multiple clusters → context-dependent sycophancy modes

IMPLICATIONS FOR STEERING:
==========================
1. If intrinsic dim > 5: Need subspace/manifold approach, not single direction
2. If clusters > 1: Different contexts may need different steering vectors  
3. If d' < 2: Steering may cause collateral damage to honest responses

RECOMMENDATIONS:
================
1. Detection: DiffMean at layer {best_layer} is sufficient (AUROC {layer_aurocs[best_layer]:.3f})
2. Steering (if ID > 5): Use PLS with r={min(int(id_mle_syco), 10)} components
3. Steering (if clusters > 2): Consider context-conditioned steering

All plots saved to: {OUTPUT_DIR}/
""")

print("=" * 60)
print("PLOTS GENERATED:")
print("=" * 60)
print("""
01_layer_auroc.png          - Detection by layer
02_pca_spectrum.png         - Eigenvalue spectrum  
03_2d_projection.png        - 2D scatter (Syco vs Honest)
04_projection_distribution.png - Projection distributions
05_method_comparison.png    - DiffMean vs PLS vs LDA
06_direction_similarity.png - Cross-layer direction consistency
07_projection_magnitude.png - Magnitude across layers
08_intrinsic_dimensionality.png - ID estimates
09_manifold_visualization.png - t-SNE/UMAP
10_cluster_analysis.png     - Silhouette & elbow
11_per_class_clusters.png   - Cluster visualization
""")

print("Analysis complete!")

