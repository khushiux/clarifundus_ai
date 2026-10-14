from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASS_LABELS_DEFAULT = [
    "0: No DR",
    "1: Mild",
    "2: Moderate",
    "3: Severe",
    "4: Proliferative",
]

def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Calculates comprehensive multi-class and binary screening metrics.

    Args:
        y_true: List of ground-truth integer labels [0, 1, 2, 3, 4].
        y_pred: List of predicted integer labels [0, 1, 2, 3, 4].
        labels: Ordered list of labels (default [0, 1, 2, 3, 4]).

    Returns:
        Structured dictionary of evaluation metrics.
    """
    valid_labels = labels or [0, 1, 2, 3, 4]
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    # 1. Quadratic Weighted Kappa (QWK)
    try:
        qwk = float(cohen_kappa_score(y_t, y_p, weights="quadratic", labels=valid_labels))
        if np.isnan(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0

    # 2. Overall Accuracy
    accuracy = float(accuracy_score(y_t, y_p))

    # 3. Macro Averaged Precision, Recall, F1
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )

    # 4. Per-class metrics
    per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(
        y_t, y_p, labels=valid_labels, average=None, zero_division=0
    )

    per_class_results = {}
    for idx, grade in enumerate(valid_labels):
        per_class_results[int(grade)] = {
            "precision": float(round(per_prec[idx], 4)),
            "recall": float(round(per_rec[idx], 4)),
            "f1_score": float(round(per_f1[idx], 4)),
            "support": int(per_support[idx]),
        }

    # 5. Confusion Matrix
    cm = confusion_matrix(y_t, y_p, labels=valid_labels)

    # 6. Exploratory Binary Screening Analysis: Referable DR (Grade >= 2) vs Non-Referable (Grade < 2)
    binary_true = (y_t >= 2).astype(int)
    binary_pred = (y_p >= 2).astype(int)

    # TP, FP, TN, FN
    tp = int(np.sum((binary_true == 1) & (binary_pred == 1)))
    fp = int(np.sum((binary_true == 0) & (binary_pred == 1)))
    tn = int(np.sum((binary_true == 0) & (binary_pred == 0)))
    fn = int(np.sum((binary_true == 1) & (binary_pred == 0)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "qwk": float(round(qwk, 4)),
        "accuracy": float(round(accuracy, 4)),
        "macro_precision": float(round(macro_prec, 4)),
        "macro_recall": float(round(macro_rec, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "binary_screening": {
            "definition": "Referable DR (Grade >= 2) vs Non-Referable DR (Grade < 2)",
            "sensitivity": float(round(sensitivity, 4)),
            "specificity": float(round(specificity, 4)),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
    }

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    class_names: Optional[List[str]] = None,
    title: str = "RetinaTrust Confusion Matrix",
) -> None:
    """Plots and saves normalized confusion matrix with raw counts and percentages."""
    names = class_names or CLASS_LABELS_DEFAULT
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(norm_cm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(names)):
        for j in range(len(names)):
            raw_val = int(cm_arr[i, j])
            pct_val = norm_cm[i, j] * 100
            color = "white" if norm_cm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{raw_val}\n({pct_val:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="left", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted DR Grade", fontsize=11, labelpad=10)
    ax.set_ylabel("True DR Grade", fontsize=11)
    ax.set_title(title, fontsize=13, pad=20)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
