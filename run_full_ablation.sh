#!/bin/bash
# Comprehensive Ablation Study for Sycophancy Detection & Steering
# 
# Usage:
#   ./run_full_ablation.sh [MODEL] [OUTPUT_DIR]
#
# Example:
#   ./run_full_ablation.sh Qwen/Qwen2.5-7B-Instruct ablation_results

set -e  # Exit on error

MODEL=${1:-"Qwen/Qwen2.5-7B-Instruct"}
OUTPUT_DIR=${2:-"ablation_results/$(date +%Y%m%d_%H%M%S)"}

echo "============================================================"
echo "COMPREHENSIVE ABLATION STUDY"
echo "============================================================"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# Extract model short name for file naming
MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

# ========================================
# PHASE 1: LAYER SWEEP
# ========================================
echo ""
echo "############################################################"
echo "# PHASE 1: LAYER SWEEP"
echo "############################################################"

python run_layer_sweep.py \
    --model "$MODEL" \
    --train-instances 1000 \
    --test-instances 500 \
    --output-dir "$OUTPUT_DIR"

# ========================================
# PHASE 2: OOD GENERALIZATION
# ========================================
echo ""
echo "############################################################"
echo "# PHASE 2: OOD GENERALIZATION"
echo "############################################################"

python run_ood_eval.py \
    --model "$MODEL" \
    --train-size 500 \
    --test-size 500 \
    --output-dir "$OUTPUT_DIR"

# ========================================
# PHASE 3: CROSS-CONCEPT TRANSFER
# ========================================
echo ""
echo "############################################################"
echo "# PHASE 3: CROSS-CONCEPT TRANSFER"
echo "############################################################"

python run_cross_concept.py \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR"

# ========================================
# PHASE 4: STEERING ABLATION
# ========================================
echo ""
echo "############################################################"
echo "# PHASE 4: STEERING ABLATION"
echo "############################################################"

BEST_LAYER=21  # From layer sweep results

# Single-layer steering with different alphas
for ALPHA in 1 10 25 50 100; do
    for INTERVENTION in subtraction clamping; do
        echo ""
        echo "--- Steering: $INTERVENTION, alpha=$ALPHA, single-layer ---"
        python run_steering.py \
            --model "$MODEL" \
            --layer $BEST_LAYER \
            --intervention-type $INTERVENTION \
            --alpha $ALPHA \
            --test-instances 100 \
            --output-dir "$OUTPUT_DIR" || true
    done
done

# Multi-layer steering with different alphas
for ALPHA in 10 25 50; do
    for INTERVENTION in subtraction clamping; do
        echo ""
        echo "--- Steering: $INTERVENTION, alpha=$ALPHA, multi-layer ---"
        python run_steering.py \
            --model "$MODEL" \
            --layer $BEST_LAYER \
            --intervention-type $INTERVENTION \
            --alpha $ALPHA \
            --multi-layer \
            --test-instances 100 \
            --output-dir "$OUTPUT_DIR" || true
    done
done

# ========================================
# SUMMARY
# ========================================
echo ""
echo "============================================================"
echo "ABLATION STUDY COMPLETE"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR"

# Create summary file
echo "Model: $MODEL" > "$OUTPUT_DIR/ablation_summary.txt"
echo "Timestamp: $(date)" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "Experiments completed:" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "  - Layer sweep" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "  - OOD generalization" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "  - Cross-concept transfer" >> "$OUTPUT_DIR/ablation_summary.txt"
echo "  - Steering ablation (single + multi-layer)" >> "$OUTPUT_DIR/ablation_summary.txt"

echo ""
echo "Summary saved to: $OUTPUT_DIR/ablation_summary.txt"
