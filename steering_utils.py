"""
Activation steering utilities for sycophancy mitigation.

Implements intervention methods to steer LLMs away from sycophantic behavior
using learned "sycophancy direction" vectors.

Intervention types:
- Subtraction: h' = h - α·v (subtract scaled direction)
- Clamping: h' = h - (h·v)·v (remove direction component completely)
- Addition: h' = h + α·v (amplify anti-sycophancy)

References:
- Representation Engineering: https://arxiv.org/abs/2310.01405
- AxBench: https://arxiv.org/abs/2501.17148
- pyvene: https://arxiv.org/abs/2403.07809
"""

import torch
import numpy as np
from typing import Callable, Literal
from dataclasses import dataclass
from functools import partial


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""
    
    layer_idx: int
    direction: np.ndarray
    alpha: float = 1.0  # Steering strength
    intervention_type: Literal["subtraction", "clamping", "addition"] = "subtraction"
    
    def __post_init__(self):
        if not isinstance(self.direction, np.ndarray):
            self.direction = np.array(self.direction)
        # Ensure direction is normalized
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm


class ActivationSteering:
    """
    Activation steering for LLM behavior modification.
    
    Uses forward hooks to intercept and modify hidden states during generation.
    This approach follows the patterns from AxBench/pyvene for intervention.
    """
    
    def __init__(self, model, config: SteeringConfig):
        """
        Initialize steering with model and configuration.
        
        Args:
            model: HuggingFace model
            config: Steering configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Convert direction to tensor
        self.direction_tensor = torch.tensor(
            config.direction, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Handle for the registered hook
        self._hook_handle = None
        self._is_active = False
    
    def _subtraction_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Subtraction intervention: h' = h - α·v
        
        Subtracts the scaled sycophancy direction from activations,
        effectively reducing sycophantic behavior.
        """
        v = self.direction_tensor.to(hidden_states.dtype)
        # Broadcast: hidden_states is [batch, seq, hidden], v is [hidden]
        return hidden_states - self.config.alpha * v
    
    def _clamping_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Clamping intervention: h' = h - (h·v)·v
        
        Projects out the sycophancy direction completely from activations.
        More aggressive than subtraction - removes all projection onto direction.
        """
        v = self.direction_tensor.to(hidden_states.dtype)
        # Compute projection coefficient: (h·v) for each position
        # hidden_states: [batch, seq, hidden], v: [hidden]
        projection_coef = torch.matmul(hidden_states, v)  # [batch, seq]
        
        # Scale by alpha to control intervention strength
        projection_coef = projection_coef * self.config.alpha
        
        # Reconstruct: h - (h·v)·v
        # projection_coef: [batch, seq], v: [hidden]
        correction = projection_coef.unsqueeze(-1) * v  # [batch, seq, hidden]
        return hidden_states - correction
    
    def _addition_intervention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Addition intervention: h' = h + α·v
        
        Adds the scaled anti-sycophancy direction (negative of sycophancy direction)
        to amplify truthful behavior. Note: direction should be negated if originally
        computed as sycophantic - non-sycophantic.
        """
        v = self.direction_tensor.to(hidden_states.dtype)
        # For addition, we ADD the direction (could negate alpha to subtract)
        return hidden_states + self.config.alpha * v
    
    def _create_hook(self) -> Callable:
        """Create the forward hook function for intervention."""
        
        intervention_fn = {
            "subtraction": self._subtraction_intervention,
            "clamping": self._clamping_intervention,
            "addition": self._addition_intervention,
        }[self.config.intervention_type]
        
        def hook(module, inputs, outputs):
            """
            Forward hook that applies intervention to layer outputs.
            
            Note: Transformer layer outputs are typically (hidden_states, ...).
            We only modify the hidden_states (first element).
            """
            if not self._is_active:
                return outputs
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                modified = intervention_fn(hidden_states)
                return (modified,) + outputs[1:]
            else:
                return intervention_fn(outputs)
        
        return hook
    
    def activate(self) -> None:
        """
        Activate steering by registering the intervention hook.
        
        Call this before running inference/generation.
        """
        if self._hook_handle is not None:
            self.deactivate()
        
        # Register hook on target layer
        target_layer = self.model.model.layers[self.config.layer_idx]
        self._hook_handle = target_layer.register_forward_hook(self._create_hook())
        self._is_active = True
    
    def deactivate(self) -> None:
        """
        Deactivate steering by removing the intervention hook.
        
        Call this after inference/generation to clean up.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._is_active = False
    
    def __enter__(self):
        """Context manager entry - activates steering."""
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deactivates steering."""
        self.deactivate()
        return False


class MultiLayerSteering:
    """
    Apply steering intervention across multiple layers simultaneously.
    
    This can be more effective than single-layer intervention for some behaviors.
    """
    
    def __init__(
        self, 
        model, 
        layer_indices: list[int], 
        direction: np.ndarray,
        alpha: float = 1.0,
        intervention_type: Literal["subtraction", "clamping", "addition"] = "subtraction",
    ):
        """
        Initialize multi-layer steering.
        
        Args:
            model: HuggingFace model
            layer_indices: List of layer indices to intervene on
            direction: Steering direction vector
            alpha: Steering strength
            intervention_type: Type of intervention
        """
        self.steerers = []
        for layer_idx in layer_indices:
            config = SteeringConfig(
                layer_idx=layer_idx,
                direction=direction,
                alpha=alpha,
                intervention_type=intervention_type,
            )
            self.steerers.append(ActivationSteering(model, config))
    
    def activate(self) -> None:
        """Activate all steering hooks."""
        for steerer in self.steerers:
            steerer.activate()
    
    def deactivate(self) -> None:
        """Deactivate all steering hooks."""
        for steerer in self.steerers:
            steerer.deactivate()
    
    def __enter__(self):
        self.activate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()
        return False


def train_sycophancy_direction(
    model,
    tokenizer,
    train_data: list[dict],
    layer_idx: int,
    device: torch.device,
    prefix_length: int = 1,
) -> np.ndarray:
    """
    Train the sycophancy direction using diff-in-means.
    
    This is a convenience wrapper that computes:
    v = mean(h | sycophantic) - mean(h | non-sycophantic)
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        train_data: List of training instances with 'prompt', 'completion', 'label'
        layer_idx: Layer to extract activations from
        device: Device to use
        prefix_length: Number of prefix tokens to skip
    
    Returns:
        Normalized sycophancy direction vector
    """
    from utils import gather_residual_activations, build_chat_prompt
    from tqdm import tqdm
    
    positive_activations = []
    negative_activations = []
    
    for item in tqdm(train_data, desc="Extracting activations for direction"):
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
        
        if item["label"] == 1:  # Sycophantic
            positive_activations.append(item_activations.cpu())
        else:  # Non-sycophantic
            negative_activations.append(item_activations.cpu())
    
    # Compute diff-in-means
    all_positive = torch.cat(positive_activations, dim=0)
    all_negative = torch.cat(negative_activations, dim=0)
    
    mean_positive = all_positive.mean(dim=0)
    mean_negative = all_negative.mean(dim=0)
    
    direction = mean_positive - mean_negative
    norm = torch.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction.to(torch.float32).numpy() ## numpy does not support converting BFloat16 to float32


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering: ActivationSteering | MultiLayerSteering,
    max_new_tokens: int = 50,
    **generate_kwargs,
) -> str:
    """
    Generate text with activation steering applied.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        steering: Steering object (single or multi-layer)
        max_new_tokens: Maximum tokens to generate
        **generate_kwargs: Additional arguments for model.generate()
    
    Returns:
        Generated text (completion only, not including prompt)
    """
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    
    # Generate with steering active
    with steering:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **generate_kwargs,
        )
    
    # Decode only the generated part
    generated_ids = output_ids[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def evaluate_steering_effect(
    model,
    tokenizer,
    test_data: list[dict],
    steering: ActivationSteering | MultiLayerSteering,
    device: torch.device,
) -> dict:
    """
    Evaluate the effect of steering on model behavior.
    
    Compares sycophancy rate before and after steering.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        test_data: Test instances with 'prompt', 'answer_matching_behavior', 'answer_not_matching_behavior'
        steering: Steering object
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
    from utils import logp_of_candidate, build_chat_prompt
    from tqdm import tqdm
    
    results_without_steering = []
    results_with_steering = []
    
    for item in tqdm(test_data, desc="Evaluating steering effect"):
        prompt = build_chat_prompt(tokenizer, item["prompt"])
        syco_answer = item.get("answer_matching_behavior", item.get("completion"))
        non_syco_answer = item.get("answer_not_matching_behavior")
        
        if non_syco_answer is None:
            continue
        
        # Without steering
        logp_syco = logp_of_candidate(model, tokenizer, prompt, syco_answer, device)
        logp_non_syco = logp_of_candidate(model, tokenizer, prompt, non_syco_answer, device)
        is_sycophantic_baseline = logp_syco > logp_non_syco
        results_without_steering.append(is_sycophantic_baseline)
        
        # With steering
        with steering:
            logp_syco_steered = logp_of_candidate(
                model, tokenizer, prompt, syco_answer, device
            )
            logp_non_syco_steered = logp_of_candidate(
                model, tokenizer, prompt, non_syco_answer, device
            )
        is_sycophantic_steered = logp_syco_steered > logp_non_syco_steered
        results_with_steering.append(is_sycophantic_steered)
    
    # Compute metrics
    n_samples = len(results_without_steering)
    sycophancy_rate_before = sum(results_without_steering) / n_samples
    sycophancy_rate_after = sum(results_with_steering) / n_samples
    
    # How many changed from sycophantic to non-sycophantic
    flipped_to_honest = sum(
        1 for b, a in zip(results_without_steering, results_with_steering)
        if b and not a  # Was sycophantic, now isn't
    )
    
    return {
        "n_samples": n_samples,
        "sycophancy_rate_before": sycophancy_rate_before,
        "sycophancy_rate_after": sycophancy_rate_after,
        "reduction": sycophancy_rate_before - sycophancy_rate_after,
        "reduction_pct": (
            (sycophancy_rate_before - sycophancy_rate_after) / sycophancy_rate_before * 100
            if sycophancy_rate_before > 0 else 0
        ),
        "flipped_to_honest": flipped_to_honest,
        "flip_rate": flipped_to_honest / n_samples,
    }
