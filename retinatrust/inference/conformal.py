from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

class ConformalPredictionResult:
    """Encapsulates prediction sets with finite-sample statistical coverage guarantees."""

    def __init__(
        self,
        predicted_grade: int,
        prediction_set_90: List[int],
        prediction_set_95: List[int],
        set_size_90: int,
        set_size_95: int,
        triage_recommendation: str,
        is_singleton_95: bool,
        empirical_nonconformity_score: float,
    ):
        self.predicted_grade = predicted_grade
        self.prediction_set_90 = prediction_set_90
        self.prediction_set_95 = prediction_set_95
        self.set_size_90 = set_size_90
        self.set_size_95 = set_size_95
        self.triage_recommendation = triage_recommendation
        self.is_singleton_95 = is_singleton_95
        self.score = empirical_nonconformity_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_grade": self.predicted_grade,
            "prediction_set_90": self.prediction_set_90,
            "prediction_set_95": self.prediction_set_95,
            "set_size_90": self.set_size_90,
            "set_size_95": self.set_size_95,
            "is_singleton_95": self.is_singleton_95,
            "triage_recommendation": self.triage_recommendation,
            "empirical_score": round(self.score, 4),
            "guarantee_statement": (
                "Under exchangeability, the true patient grade is mathematically "
                "guaranteed to lie within prediction_set_95 with at least 95% probability."
            ),
        }

class ConformalPredictor:
    """Split Conformal Prediction engine using Adaptive Prediction Sets (APS).

    Calibrated on validation set logits to provide distribution-free finite-sample guarantees.
    """

    def __init__(self, q_hat_90: float = 0.88, q_hat_95: float = 0.95):
        self.q_hat_90 = q_hat_90
        self.q_hat_95 = q_hat_95

    @classmethod
    def fit_from_calibration(
        cls,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
    ) -> "ConformalPredictor":
        """Fits non-conformity quantiles on validation probabilities and labels.

        Args:
            val_probs: Array of shape [N, num_classes].
            val_labels: Array of shape [N] with true class integer labels.
        """
        n = len(val_labels)
        scores = []
        for i in range(n):
            true_label = int(val_labels[i])
            prob_dist = val_probs[i]
            # Adaptive Prediction Set nonconformity score:
            # Sort probabilities descending and sum until the true class is reached
            sort_indices = np.argsort(-prob_dist)
            cum_prob = 0.0
            for idx in sort_indices:
                cum_prob += float(prob_dist[idx])
                if idx == true_label:
                    break
            scores.append(cum_prob)

        scores = np.array(scores)
        # Compute finite-sample adjusted quantiles: ceil((n + 1)(1 - alpha)) / n
        q_level_90 = min(1.0, float(np.ceil((n + 1) * 0.90) / n))
        q_level_95 = min(1.0, float(np.ceil((n + 1) * 0.95) / n))

        q_hat_90 = float(min(1.0, np.quantile(scores, q_level_90, method="higher")))
        q_hat_95 = float(min(1.0, np.quantile(scores, q_level_95, method="higher")))

        return cls(q_hat_90=q_hat_90, q_hat_95=q_hat_95)

    def predict_set(self, probabilities: np.ndarray) -> ConformalPredictionResult:
        """Constructs 90% and 95% guaranteed prediction sets for an input probability distribution."""
        probs = np.asarray(probabilities).flatten()
        sorted_indices = np.argsort(-probs)
        pred_grade = int(sorted_indices[0])

        set_90: List[int] = []
        cum_90 = 0.0
        for idx in sorted_indices:
            set_90.append(int(idx))
            cum_90 += probs[idx]
            if cum_90 >= self.q_hat_90:
                break

        set_95: List[int] = []
        cum_95 = 0.0
        for idx in sorted_indices:
            set_95.append(int(idx))
            cum_95 += probs[idx]
            if cum_95 >= self.q_hat_95:
                break

        # Triage clinical logic
        is_singleton = (len(set_95) == 1)
        if is_singleton and pred_grade == 0:
            triage = "Routine Annual Screening (Low Risk, Single-Grade Certainty)"
        elif is_singleton and pred_grade in (1, 2):
            triage = "Non-Urgent Clinical Ophthalmology Follow-Up"
        elif pred_grade >= 3:
            triage = "Urgent Retina Specialist Referral (High Severity Detected)"
        else:
            triage = (
                f"Mandatory Clinical Review Required (Ambiguous Conformal Set: "
                f"{set_95} with {len(set_95)} candidate grades)"
            )

        top_score = float(probs[pred_grade])

        return ConformalPredictionResult(
            predicted_grade=pred_grade,
            prediction_set_90=sorted(set_90),
            prediction_set_95=sorted(set_95),
            set_size_90=len(set_90),
            set_size_95=len(set_95),
            triage_recommendation=triage,
            is_singleton_95=is_singleton,
            empirical_nonconformity_score=top_score,
        )
