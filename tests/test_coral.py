import numpy as np
import pytest
import torch
from retinatrust.models.coral import (
    CoralLayer,
    CoralLoss,
    coral_logits_to_probs,
    coral_predict_grade,
    label_to_coral_levels,
)

def test_label_to_coral_levels():
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    levels = label_to_coral_levels(labels, num_classes=5)

    expected = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    assert torch.equal(levels, expected)

def test_coral_layer_forward():
    layer = CoralLayer(in_features=64, num_classes=5)
    dummy_feat = torch.randn(8, 64)
    logits = layer(dummy_feat)

    assert logits.shape == (8, 4)

def test_coral_logits_to_probs():
    # Large positive logits -> high severity grade
    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    probs = coral_logits_to_probs(logits)

    assert probs.shape == (1, 5)
    assert probs[0, 4] > 0.80
    assert torch.isclose(probs.sum(), torch.tensor(1.0))

    # Large negative logits -> Grade 0
    logits_neg = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])
    probs_neg = coral_logits_to_probs(logits_neg)
    assert probs_neg[0, 0] > 0.80
    assert torch.isclose(probs_neg.sum(), torch.tensor(1.0))

def test_coral_loss():
    loss_fn = CoralLoss(num_classes=5)
    logits = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 2, 3, 4], dtype=torch.long)

    loss = loss_fn(logits, labels)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
