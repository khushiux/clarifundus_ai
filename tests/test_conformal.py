import numpy as np
import pytest
from retinatrust.inference.conformal import ConformalPredictor, ConformalPredictionResult

def test_conformal_predictor_certain():
    # Model is 99% confident in Grade 0
    probs = np.array([0.99, 0.005, 0.003, 0.001, 0.001])
    engine = ConformalPredictor(q_hat_90=0.90, q_hat_95=0.95)

    result = engine.predict_set(probs)

    assert isinstance(result, ConformalPredictionResult)
    assert result.predicted_grade == 0
    # At 99% confidence, 95% set is a singleton {0}
    assert result.prediction_set_95 == [0]
    assert result.is_singleton_95 is True
    assert "Routine" in result.triage_recommendation

def test_conformal_predictor_ambiguous():
    # Dispersed probabilities across Grade 2 and Grade 3
    probs = np.array([0.05, 0.10, 0.45, 0.35, 0.05])
    engine = ConformalPredictor(q_hat_90=0.85, q_hat_95=0.95)

    result = engine.predict_set(probs)

    assert result.is_singleton_95 is False
    assert len(result.prediction_set_95) >= 2
    assert 2 in result.prediction_set_95
    assert 3 in result.prediction_set_95
    assert "Referral" in result.triage_recommendation or "Review" in result.triage_recommendation

def test_conformal_fit_from_calibration():
    # 50 samples
    np.random.seed(42)
    val_probs = np.random.dirichlet(np.ones(5), size=50)
    val_labels = np.random.randint(0, 5, size=50)

    engine = ConformalPredictor.fit_from_calibration(val_probs, val_labels)

    assert 0.0 < engine.q_hat_90 <= 1.0
    assert 0.0 < engine.q_hat_95 <= 1.0
    assert engine.q_hat_90 <= engine.q_hat_95
