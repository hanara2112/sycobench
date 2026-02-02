#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Sycophancy Detection & Steering.

Runs all combinations of:
- Domains: political, philpapers, nlp_survey
- Models: Qwen2.5-7B-Instruct (and optionally smaller models)
- Layers: sweep across all layers
- Detection methods: DiffMean
- Steering methods: subtraction, clamping (single-layer and multi-layer)
- Alpha values: 1, 10, 25, 50, 100

Usage:
    # Full ablation (takes several hours)
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct

    # Quick test run
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --quick

    # Detection only
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --detection-only

    # Steering only
    python run_ablation.py --model Qwen/Qwen2.5-7B-Instruct --steering-only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from itertools import product

# Configuration
DOMAINS = ["political", "philpapers", "nlp_survey"]
ALPHAS = [1.0, 10.0, 25.0, 50.0, 100.0]
INTERVENTION_TYPES = ["subtraction", "clamping"]


def run_command(cmd: list[str], description: str) -> dict:
    """Run a command and capture output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("="*60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        return {"success": True, "output": result.stdout}
    
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command took too long")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"success": False, "error": str(e)}


def run_layer_sweep(model: str, domain: str, output_dir: str) -> dict:
    """Run layer sweep for detection."""
    cmd = [
        sys.executable, "run_layer_sweep.py",
        "--model", model,
        "--domain", domain,
        "--output-dir", output_dir,
    ]
    return run_command(cmd, f"Layer Sweep - Domain: {domain}")


def run_detection(model: str, domain: str, layer: int, output_dir: str) -> dict:
    """Run detection benchmark."""
    cmd = [
        sys.executable, "run_detection.py",
        "--model", model,
        "--domain", domain,
        "--layer", str(layer),
        "--output-dir", output_dir,
    ]
    return run_command(cmd, f"Detection - Domain: {domain}, Layer: {layer}")


def run_ood_eval(model: str, layer: int, output_dir: str) -> dict:
    """Run OOD generalization evaluation."""
    cmd = [
        sys.executable, "run_ood_eval.py",
        "--model", model,
        "--layer", str(layer),
        "--output-dir", output_dir,
    ]
    return run_command(cmd, f"OOD Evaluation - Layer: {layer}")


def run_steering(
    model: str,
    layer: int,
    alpha: float,
    intervention_type: str,
    multi_layer: bool,
    output_dir: str,
) -> dict:
    """Run steering experiment."""
    cmd = [
        sys.executable, "run_steering.py",
        "--model", model,
        "--layer", str(layer),
        "--alpha", str(alpha),
        "--intervention-type", intervention_type,
        "--output-dir", output_dir,
    ]
    if multi_layer:
        cmd.append("--multi-layer")
    
    layer_desc = f"multi-layer {layer-3}-{layer+3}" if multi_layer else f"layer {layer}"
    return run_command(
        cmd,
        f"Steering - {intervention_type}, α={alpha}, {layer_desc}"
    )


def run_cross_concept(model: str, layer: int, output_dir: str) -> dict:
    """Run cross-concept transfer analysis."""
    cmd = [
        sys.executable, "run_cross_concept.py",
        "--model", model,
        "--layer", str(layer),
        "--output-dir", output_dir,
    ]
    return run_command(cmd, f"Cross-Concept Transfer - Layer: {layer}")


def main(
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "ablation_results",
    quick: bool = False,
    detection_only: bool = False,
    steering_only: bool = False,
    layer: int | None = None,
):
    """Run comprehensive ablation study."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE ABLATION STUDY")
    print("="*60)
    print(f"Model: {model}")
    print(f"Output: {output_path}")
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print(f"Detection only: {detection_only}")
    print(f"Steering only: {steering_only}")
    print("="*60)
    
    results = {
        "config": {
            "model": model,
            "timestamp": timestamp,
            "quick": quick,
        },
        "layer_sweep": {},
        "detection": {},
        "ood_eval": {},
        "steering": {},
        "cross_concept": {},
    }
    
    # Determine best layer (from layer sweep or provided)
    best_layer = layer
    
    # ========================================
    # 1. LAYER SWEEP (per domain)
    # ========================================
    if not steering_only:
        print("\n" + "#"*60)
        print("# PHASE 1: LAYER SWEEP")
        print("#"*60)
        
        domains_to_test = DOMAINS[:1] if quick else DOMAINS
        
        for domain in domains_to_test:
            result = run_layer_sweep(model, domain, str(output_path))
            results["layer_sweep"][domain] = result
        
        # TODO: Parse layer sweep results to find best layer
        if best_layer is None:
            # Default to 75% depth for now
            best_layer = 21  # For 28-layer model
    
    if best_layer is None:
        best_layer = 21
    
    print(f"\nUsing layer {best_layer} for subsequent experiments")
    
    # ========================================
    # 2. DETECTION BENCHMARK (per domain)
    # ========================================
    if not steering_only:
        print("\n" + "#"*60)
        print("# PHASE 2: DETECTION BENCHMARK")
        print("#"*60)
        
        domains_to_test = DOMAINS[:1] if quick else DOMAINS
        
        for domain in domains_to_test:
            result = run_detection(model, domain, best_layer, str(output_path))
            results["detection"][domain] = result
    
    # ========================================
    # 3. OOD GENERALIZATION
    # ========================================
    if not steering_only:
        print("\n" + "#"*60)
        print("# PHASE 3: OOD GENERALIZATION")
        print("#"*60)
        
        result = run_ood_eval(model, best_layer, str(output_path))
        results["ood_eval"] = result
    
    # ========================================
    # 4. CROSS-CONCEPT TRANSFER
    # ========================================
    if not steering_only:
        print("\n" + "#"*60)
        print("# PHASE 4: CROSS-CONCEPT TRANSFER")
        print("#"*60)
        
        result = run_cross_concept(model, best_layer, str(output_path))
        results["cross_concept"] = result
    
    # ========================================
    # 5. STEERING ABLATION
    # ========================================
    if not detection_only:
        print("\n" + "#"*60)
        print("# PHASE 5: STEERING ABLATION")
        print("#"*60)
        
        alphas_to_test = [25.0, 50.0] if quick else ALPHAS
        interventions_to_test = ["clamping"] if quick else INTERVENTION_TYPES
        multi_layer_options = [False, True]
        
        for intervention, alpha, multi in product(
            interventions_to_test, alphas_to_test, multi_layer_options
        ):
            key = f"{intervention}_alpha{alpha}_{'multi' if multi else 'single'}"
            result = run_steering(
                model, best_layer, alpha, intervention, multi, str(output_path)
            )
            results["steering"][key] = result
    
    # ========================================
    # SAVE FINAL RESULTS
    # ========================================
    results_file = output_path / "ablation_summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n### SUMMARY ###")
    print(f"Layer sweeps: {len(results['layer_sweep'])}")
    print(f"Detection runs: {len(results['detection'])}")
    print(f"OOD eval: {'Done' if results['ood_eval'] else 'Skipped'}")
    print(f"Cross-concept: {'Done' if results['cross_concept'] else 'Skipped'}")
    print(f"Steering runs: {len(results['steering'])}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive ablation study"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ablation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test subset of configurations",
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Run only detection experiments",
    )
    parser.add_argument(
        "--steering-only",
        action="store_true",
        help="Run only steering experiments",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to use (skips layer sweep)",
    )
    
    args = parser.parse_args()
    
    main(
        model=args.model,
        output_dir=args.output_dir,
        quick=args.quick,
        detection_only=args.detection_only,
        steering_only=args.steering_only,
        layer=args.layer,
    )
