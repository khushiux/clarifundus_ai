import numpy as np
import pytest
import torch
from retinatrust.inference.uncertainty import (
    compute_predictive_entropy,
    quantify_uncertainty,
)
from retinatrust.config import UncertaintyConfig

def test_compute_predictive_entropy():
    # Deterministic distribution -> entropy near 0
    p_certain = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    h_certain = compute_predictive_entropy(p_certain)
    assert h_certain == pytest.approx(0.0, abs=1e-4)

    # Uniform distribution (5 classes) -> max entropy log2(5) approx 2.3219
    p_uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    h_uniform = compute_predictive_entropy(p_uniform)
    assert h_uniform == pytest.approx(np.log2(5), abs=1e-3)

def test_quantify_uncertainty_confident():
    # Dominant logit on class 2
    logits = torch.tensor([0.1, 0.2, 5.0, 0.3, 0.1])
    res = quantify_uncertainty(logits, temperature=1.0)
    assert res.predicted_grade == 2
    assert res.confidence > 0.80
    assert res.is_uncertain is False
    assert len(res.reasons) == 0

def test_quantify_uncertainty_ambiguous():
    # Nearly flat logits -> high entropy & low confidence
    logits = torch.tensor([0.5, 0.52, 0.49, 0.51, 0.48])
    cfg = UncertaintyConfig(min_confidence_threshold=0.50, max_entropy_threshold=1.50)
    res = quantify_uncertainty(logits, temperature=1.0, config=cfg)
    assert res.is_uncertain is True
    assert len(res.reasons) > 0
    assert any("confidence" in r.lower() or "entropy" in r.lower() for r in res.reasons)
