import numpy as np
import pytest
from retinatrust.training.metrics import calculate_metrics

def test_calculate_metrics_perfect_predictions():
    y_true = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["qwk"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["binary_screening"]["sensitivity"] == 1.0
    assert metrics["binary_screening"]["specificity"] == 1.0

def test_calculate_metrics_imperfect_predictions():
    y_true = [0, 0, 2, 3, 4]
    y_pred = [0, 1, 2, 2, 4]

    metrics = calculate_metrics(y_true, y_pred)
    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 < metrics["qwk"] < 1.0
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 5
    assert len(metrics["confusion_matrix"]) == 5

def test_binary_screening_referable_dr():
    # True: 2 non-referable (0, 1), 2 referable (2, 4)
    # Pred: 1 non-referable (0), 3 referable (2, 3, 4) -> 1 false positive
    y_true = [0, 1, 2, 4]
    y_pred = [0, 2, 2, 4]

    metrics = calculate_metrics(y_true, y_pred)
    # Sensitivity: TP=2, FN=0 -> 2/2 = 1.0
    assert metrics["binary_screening"]["sensitivity"] == 1.0
    # Specificity: TN=1, FP=1 -> 1/2 = 0.5
    assert metrics["binary_screening"]["specificity"] == 0.5
