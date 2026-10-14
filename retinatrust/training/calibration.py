import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from retinatrust.config import load_config, resolve_path
from retinatrust.data.dataset import RetinaDataset
from retinatrust.data.transforms import get_eval_transforms
from retinatrust.models.build_model import build_model, get_device
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.calibration")

class ModelWithTemperature(nn.Module):
    """A thin decorator that scales logits by a learned positive scalar temperature T.

    Fitted strictly on validation data using negative log-likelihood (NLL).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialize temperature at 1.0 (log(1.0) = 0.0 or parameter = 1.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamping temperature to avoid division by zero or extreme scaling
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temp

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins.

    Args:
        probs: Array of shape [N, num_classes] with class probabilities.
        labels: Array of shape [N] with true integer class labels.
        n_bins: Number of equal-width confidence intervals in [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(round(ece, 4))

def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 5) -> float:
    """Computes multiclass Brier score: Mean squared error of predicted probabilities vs one-hot targets."""
    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=float)
    one_hot[np.arange(n), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(round(brier, 4))

def calibrate_model(
    checkpoint_path: str,
    val_manifest_path: str,
    output_dir: Optional[str] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Finds optimal temperature T on validation logits and updates model metadata."""
    ckpt_file = resolve_path(checkpoint_path)
    val_manifest = resolve_path(val_manifest_path)

    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    if not val_manifest.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")

    device = get_device(device_name)
    logger.info(f"Loading checkpoint for calibration: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)

    arch = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 5)
    input_size = tuple(checkpoint.get("input_size", (224, 224)))
    norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])

    model, _ = build_model(
        architecture=arch,
        num_classes=num_classes,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    eval_transform = get_eval_transforms(input_size=input_size, norm_mean=norm_mean, norm_std=norm_std)
    dataset = RetinaDataset(val_manifest, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Collect all validation logits and labels
    logger.info(f"Extracting validation logits from {len(dataset)} samples...")
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            logits = model(imgs)
            logits_list.append(logits)
            labels_list.append(lbls)

    val_logits = torch.cat(logits_list).to(device)
    val_labels = torch.cat(labels_list).to(device)

    # Uncalibrated baseline metrics
    uncal_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
    np_labels = val_labels.cpu().numpy()

    uncal_ece = compute_ece(uncal_probs, np_labels)
    uncal_brier = compute_brier_score(uncal_probs, np_labels, num_classes=num_classes)
    logger.info(f"Uncalibrated (T=1.0) -> ECE: {uncal_ece:.4f} | Brier: {uncal_brier:.4f}")

    # Learn optimal temperature via L-BFGS
    temp_module = ModelWithTemperature(model).to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS([temp_module.temperature], lr=0.01, max_iter=100)

    def eval_closure():
        optimizer.zero_grad()
        loss = nll_criterion(temp_module.temperature_scale(val_logits), val_labels)
        loss.backward()
        return loss

    optimizer.step(eval_closure)

    optimal_t = float(torch.clamp(temp_module.temperature, min=0.01, max=10.0).item())
    logger.info(f"Optimized Temperature T = {optimal_t:.4f}")

    # Calibrated metrics
    with torch.no_grad():
        cal_logits = val_logits / optimal_t
        cal_probs = torch.softmax(cal_logits, dim=1).cpu().numpy()

    cal_ece = compute_ece(cal_probs, np_labels)
    cal_brier = compute_brier_score(cal_probs, np_labels, num_classes=num_classes)
    logger.info(f"Calibrated (T={optimal_t:.3f}) -> ECE: {cal_ece:.4f} | Brier: {cal_brier:.4f}")

    result = {
        "architecture": arch,
        "checkpoint_path": str(ckpt_file),
        "temperature": round(optimal_t, 4),
        "uncalibrated_ece": uncal_ece,
        "calibrated_ece": cal_ece,
        "uncalibrated_brier": uncal_brier,
        "calibrated_brier": cal_brier,
    }

    # Save to checkpoint metadata
    checkpoint["temperature"] = optimal_t
    checkpoint["calibration_metrics"] = result
    torch.save(checkpoint, ckpt_file)
    logger.info(f"Updated checkpoint {ckpt_file} with calibration temperature {optimal_t:.4f}.")

    # Save standalone calibration json
    out_dir = Path(output_dir) if output_dir else ckpt_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"calibration_{arch}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved calibration report to: {json_path}")

    return result

def main():
    parser = argparse.ArgumentParser(description="RetinaTrust Temperature Scaling Calibration")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/resnet18_best.pt", help="Model checkpoint")
    parser.add_argument("--val-manifest", type=str, default="artifacts/manifests/val_manifest.csv", help="Validation manifest")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    args = parser.parse_args()

    try:
        calibrate_model(
            checkpoint_path=args.checkpoint,
            val_manifest_path=args.val_manifest,
            device_name=args.device,
        )
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
