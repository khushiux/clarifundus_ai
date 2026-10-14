import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from retinatrust.config import load_config, load_datasets_config, resolve_path
from retinatrust.data.dataset import RetinaDataset
from retinatrust.data.transforms import get_eval_transforms
from retinatrust.models.build_model import build_model, get_device
from retinatrust.training.metrics import calculate_metrics, plot_confusion_matrix
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.evaluate")

def evaluate_checkpoint(
    checkpoint_path: str,
    manifest_path: str,
    dataset_name: str = "APTOS 2019 Test Set",
    output_dir: Optional[str] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Loads a trained checkpoint and evaluates it independently on a test manifest.

    Never tunes hyperparameters or thresholds during evaluation.
    """
    ckpt_file = resolve_path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    manifest_file = resolve_path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Test manifest file not found: {manifest_file}")

    device = get_device(device_name)
    logger.info(f"Loading checkpoint from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)

    arch = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 5)
    input_size = tuple(checkpoint.get("input_size", (224, 224)))
    norm_mean = checkpoint.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std = checkpoint.get("norm_std", [0.229, 0.224, 0.225])

    # Construct model and load weights
    model, _ = build_model(
        architecture=arch,
        num_classes=num_classes,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load dataset
    eval_transform = get_eval_transforms(
        input_size=input_size,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    test_dataset = RetinaDataset(manifest_file, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Evaluating {len(test_dataset)} samples from {dataset_name} on {device}...")

    y_trues: List[int] = []
    y_preds: List[int] = []
    all_logits: List[List[float]] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_preds.extend(preds)
            y_trues.extend(labels.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    metrics = calculate_metrics(y_trues, y_preds)

    logger.info("========== Evaluation Results ==========")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Quadratic Weighted Kappa (QWK): {metrics['qwk']:.4f}")
    logger.info(f"Accuracy:                       {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1-Score:                 {metrics['macro_f1']:.4f}")
    logger.info(f"Referable DR Sensitivity:       {metrics['binary_screening']['sensitivity']:.4f}")
    logger.info(f"Referable DR Specificity:       {metrics['binary_screening']['specificity']:.4f}")

    # Output directory
    app_cfg = load_config()
    out_path = Path(output_dir) if output_dir else resolve_path("reports/figures")
    out_path.mkdir(parents=True, exist_ok=True)

    slug = dataset_name.lower().replace(" ", "_").replace("/", "_")
    cm_path = out_path / f"{slug}_confusion_matrix.png"
    plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        save_path=cm_path,
        title=f"Confusion Matrix: {dataset_name}",
    )
    logger.info(f"Saved confusion matrix chart to: {cm_path}")

    # Generate Markdown Report
    report_md_path = out_path / f"{slug}_report.md"
    report_content = r"""# Independent Evaluation Report: {dataset_name}

- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Model Checkpoint:** `{checkpoint_path}`
- **Architecture:** `{arch}`
- **Total Test Samples:** `{len(test_dataset)}`

## Primary Multi-Class Metrics
| Metric | Value |
|:---|:---:|
| **Quadratic Weighted Kappa (QWK)** | **{metrics['qwk']}** |
| **Accuracy** | **{metrics['accuracy']}** |
| **Macro Precision** | {metrics['macro_precision']} |
| **Macro Recall** | {metrics['macro_recall']} |
| **Macro F1-Score** | {metrics['macro_f1']} |

## Per-Class Breakdown
| Grade | Severity Level | Support | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
    class_names = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
    for grade, pdata in metrics["per_class"].items():
        name = class_names[grade] if grade < len(class_names) else f"Grade {grade}"
        report_content += f"| {grade} | {name} | {pdata['support']} | {pdata['precision']} | {pdata['recall']} | {pdata['f1_score']} |\n"

    report_content += f"""
## Exploratory Binary Screening Analysis
- **Definition:** Referable DR (Grade $\ge$ 2) vs Non-Referable DR (Grade < 2)
- **Sensitivity:** {metrics['binary_screening']['sensitivity']}
- **Specificity:** {metrics['binary_screening']['specificity']}
- **True Positives:** {metrics['binary_screening']['true_positives']} | **False Positives:** {metrics['binary_screening']['false_positives']}
- **True Negatives:** {metrics['binary_screening']['true_negatives']} | **False Negatives:** {metrics['binary_screening']['false_negatives']}

## Confusion Matrix
![Confusion Matrix]({cm_path})
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Saved evaluation report to: {report_md_path}")

    # Save summary JSON
    summary_json_path = out_path / f"{slug}_metrics.json"
    full_summary = {
        "dataset_name": dataset_name,
        "checkpoint_path": str(ckpt_file),
        "manifest_path": str(manifest_file),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    return full_summary

def main():
    parser = argparse.ArgumentParser(description="RetinaTrust Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default="artifacts/models/resnet18_best.pt", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="artifacts/manifests/test_manifest.csv", help="Test manifest CSV")
    parser.add_argument("--name", type=str, default="APTOS 2019 Test Set", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device")
    args = parser.parse_args()

    try:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            dataset_name=args.name,
            output_dir=args.output_dir,
            device_name=args.device,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
