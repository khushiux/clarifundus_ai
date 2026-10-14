from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralLayer(nn.Module):
    """Consistent Rank Logits (CORAL) classification head for ordinal regression.

    Uses a single shared linear weight vector with K-1 independent rank biases.
    """

    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_classes - 1  # 4 binary tasks for 5 classes
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection: [N, 1]
        proj = self.linear(x)
        # Broadcast across tasks: [N, num_tasks]
        logits = proj + self.bias
        return logits

def label_to_coral_levels(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Converts integer grade labels (0 to 4) into binary ordinal targets (K-1 tasks).

    E.g. Grade 0 -> [0, 0, 0, 0]
         Grade 1 -> [1, 0, 0, 0]
         Grade 2 -> [1, 1, 0, 0]
         Grade 3 -> [1, 1, 1, 0]
         Grade 4 -> [1, 1, 1, 1]
    """
    num_tasks = num_classes - 1
    levels = []
    for label in labels:
        target = [1.0 if label > k else 0.0 for k in range(num_tasks)]
        levels.append(target)
    return torch.tensor(levels, dtype=torch.float32, device=labels.device)

class CoralLoss(nn.Module):
    """Computes binary cross-entropy loss across all K-1 ordinal tasks."""

    def __init__(self, num_classes: int = 5, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        levels = label_to_coral_levels(labels, num_classes=self.num_classes)
        loss = F.binary_cross_entropy_with_logits(logits, levels, reduction=self.reduction)
        return loss

def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Converts K-1 binary rank logits into a normalized 5-class probability distribution.

    Args:
        logits: Tensor of shape [N, num_classes - 1] or [num_classes - 1].

    Returns:
        Tensor of shape [N, num_classes] with class probabilities summing to 1.
    """
    was_1d = (logits.ndim == 1)
    if was_1d:
        logits = logits.unsqueeze(0)

    # Sigmoid to get P(Y > k): shape [N, num_tasks]
    p_cum = torch.sigmoid(logits)
    n, num_tasks = p_cum.shape
    num_classes = num_tasks + 1

    probs = torch.zeros((n, num_classes), dtype=torch.float32, device=logits.device)
    # P(Y = 0) = 1 - P(Y > 0)
    probs[:, 0] = torch.clamp(1.0 - p_cum[:, 0], min=0.0)

    # Intermediate classes: P(Y = k) = P(Y > k-1) - P(Y > k)
    for k in range(1, num_tasks):
        probs[:, k] = torch.clamp(p_cum[:, k - 1] - p_cum[:, k], min=0.0)

    # P(Y = K-1) = P(Y > K-2)
    probs[:, -1] = torch.clamp(p_cum[:, -1], min=0.0)

    # Normalize to ensure sum is exactly 1.0
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = torch.where(row_sums > 0, probs / row_sums, torch.ones_like(probs) / num_classes)

    return probs.squeeze(0) if was_1d else probs

def coral_predict_grade(logits: torch.Tensor) -> torch.Tensor:
    """Predicts integer class grade (0 to 4) by counting binary thresholds > 0 (prob > 0.5)."""
    # Count how many logits are > 0 (i.e. sigmoid(logits) > 0.5)
    return (logits > 0.0).sum(dim=-1).long()
