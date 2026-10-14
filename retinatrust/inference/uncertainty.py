from typing import Any, Dict, List, Optional
import numpy as np
import torch

from retinatrust.config import UncertaintyConfig

class UncertaintyResult:
    """Encapsulates model confidence, predictive entropy, and uncertainty flags."""

    def __init__(
        self,
        predicted_grade: int,
        confidence: float,
        entropy: float,
        is_uncertain: bool,
        reasons: List[str],
        probabilities: Dict[int, float],
        is_calibrated: bool = False,
    ):
        self.predicted_grade = predicted_grade
        self.confidence = confidence
        self.entropy = entropy
        self.is_uncertain = is_uncertain
        self.reasons = reasons
        self.probabilities = probabilities
        self.is_calibrated = is_calibrated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_grade": self.predicted_grade,
            "estimated_confidence": round(self.confidence, 4),
            "predictive_entropy": round(self.entropy, 4),
            "is_uncertain": self.is_uncertain,
            "uncertainty_reasons": self.reasons,
            "class_probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_calibrated": self.is_calibrated,
            "disclaimer": (
                "Uncertainty indicators are research signals highlighting ambiguous "
                "or out-of-distribution feature patterns. Not certified for clinical triage."
            ),
        }

def compute_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Shannon predictive entropy in bits:

    H(p) = - sum(p_c * log2(p_c + eps))
    For a 5-class distribution, max entropy is log2(5) approx 2.32 bits (uniform distribution).
    """
    probs = np.clip(probabilities, eps, 1.0)
    entropy = -float(np.sum(probs * np.log2(probs)))
    return max(0.0, entropy)

def quantify_uncertainty(
    logits: torch.Tensor,
    temperature: float = 1.0,
    config: Optional[UncertaintyConfig] = None,
) -> UncertaintyResult:
    """Computes calibrated confidence, predictive entropy, and evaluates uncertainty flags.

    Args:
        logits: PyTorch tensor of shape [num_classes] or [1, num_classes].
        temperature: Learned temperature scalar T (T > 0). If T != 1.0, considered calibrated.
        config: Optional UncertaintyConfig with threshold parameters.

    Returns:
        UncertaintyResult instance.
    """
    cfg = config or UncertaintyConfig()

    # Flatten logits if batch dimension is present
    flat_logits = logits.squeeze().detach().cpu()
    if flat_logits.ndim != 1:
        raise ValueError(f"Expected 1D logits tensor, got shape {flat_logits.shape}")

    # Apply temperature scaling
    t = max(0.01, float(temperature))
    scaled_logits = flat_logits / t
    probs_tensor = torch.softmax(scaled_logits, dim=0)
    probs_np = probs_tensor.numpy()

    predicted_grade = int(np.argmax(probs_np))
    confidence = float(np.max(probs_np))
    entropy = compute_predictive_entropy(probs_np)

    reasons: List[str] = []
    if confidence < cfg.min_confidence_threshold:
        reasons.append(
            f"Low confidence score ({confidence:.2%} < threshold {cfg.min_confidence_threshold:.2%}). "
            "Model is divided among multiple candidate grades."
        )

    if entropy > cfg.max_entropy_threshold:
        reasons.append(
            f"High predictive entropy ({entropy:.2f} bits > threshold {cfg.max_entropy_threshold:.2f} bits). "
            "Class probability mass is dispersed rather than concentrated."
        )

    is_uncertain = len(reasons) > 0
    is_calibrated = abs(t - 1.0) > 1e-3

    prob_dict = {i: float(probs_np[i]) for i in range(len(probs_np))}

    return UncertaintyResult(
        predicted_grade=predicted_grade,
        confidence=confidence,
        entropy=entropy,
        is_uncertain=is_uncertain,
        reasons=reasons,
        probabilities=prob_dict,
        is_calibrated=is_calibrated,
    )
