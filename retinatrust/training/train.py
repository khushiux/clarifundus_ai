import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from retinatrust.config import load_config, load_datasets_config, resolve_path
from retinatrust.data.dataset import get_dataloaders
from retinatrust.data.manifest import create_stratified_manifests
from retinatrust.data.transforms import get_eval_transforms, get_train_transforms
from retinatrust.models.build_model import build_model, get_device
from retinatrust.training.metrics import calculate_metrics, plot_confusion_matrix
from retinatrust.utils.logging_utils import setup_logger
from retinatrust.utils.reproducibility import set_seed

logger = setup_logger("retinatrust.train")

def compute_class_weights(train_df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    """Computes balanced inverse-frequency weights: w_c = N / (C * N_c)"""
    counts = train_df["label"].value_counts().to_dict()
    total_samples = len(train_df)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            w = total_samples / (num_classes * cnt)
        else:
            w = 1.0
        weights.append(w)
    # Normalize weights so mean is 1.0
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_training: bool = True,
) -> Tuple[float, List[int], List[int]]:
    """Runs a single training or validation epoch over a DataLoader."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_trues: List[int] = []
    y_preds: List[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        y_preds.extend(preds.tolist())
        y_trues.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_trues, y_preds

def train_model(
    config_path: Optional[str] = None,
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_arch: Optional[str] = None,
    override_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrates model training, validation, early stopping, and checkpoint generation."""
    cfg = load_config(config_path)
    datasets_cfg = load_datasets_config()

    # Reproducibility
    seed = cfg.training.seed
    set_seed(seed=seed, deterministic=True)

    arch = override_arch or cfg.model.architecture
    epochs = override_epochs or (cfg.smoke_test.epochs if smoke_test else cfg.training.epochs)
    batch_size = override_batch_size or (cfg.smoke_test.batch_size if smoke_test else cfg.training.batch_size)
    device_req = override_device or cfg.training.device
    device = get_device(device_req)

    run_id = f"run_{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{'_smoke' if smoke_test else ''}"
    run_dir = resolve_path(f"reports/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"========== Starting RetinaTrust Training Run: {run_id} ==========")
    logger.info(f"Architecture: {arch} | Epochs: {epochs} | Batch Size: {batch_size} | Device: {device}")

    # Check for manifests
    manifest_dir = resolve_path(cfg.paths.manifest_dir)
    train_manifest = manifest_dir / "train_manifest.csv"
    val_manifest = manifest_dir / "val_manifest.csv"
    test_manifest = manifest_dir / "test_manifest.csv"

    if not (train_manifest.is_file() and val_manifest.is_file()):
        logger.info("Manifests not found. Generating stratified splits...")
        create_stratified_manifests(
            dataset_key="aptos",
            output_dir=str(manifest_dir),
            train_ratio=cfg.data_split.train_ratio,
            val_ratio=cfg.data_split.val_ratio,
            test_ratio=cfg.data_split.test_ratio,
            seed=seed,
        )

    # Load manifest dataframes
    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)

    if smoke_test:
        logger.info(f"Smoke-test enabled: subsetting train to {cfg.smoke_test.max_samples} samples.")
        train_df = train_df.head(cfg.smoke_test.max_samples)
        val_df = val_df.head(min(20, len(val_df)))

    # Transforms & Dataloaders
    train_transform = get_train_transforms(
        input_size=tuple(cfg.model.input_size),
        norm_mean=cfg.model.norm_mean,
        norm_std=cfg.model.norm_std,
    )
    eval_transform = get_eval_transforms(
        input_size=tuple(cfg.model.input_size),
        norm_mean=cfg.model.norm_mean,
        norm_std=cfg.model.norm_std,
    )

    loaders = get_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_manifest if not smoke_test else None,
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    # Class weighting
    if cfg.training.class_weighting == "balanced" and not smoke_test:
        class_weights = compute_class_weights(train_df, num_classes=cfg.model.num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info(f"Applied balanced class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Model & Optimizer
    model, target_layer = build_model(
        architecture=arch,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Tracking
    best_qwk = -1.0
    patience = cfg.training.early_stopping_patience
    patience_counter = 0
    best_checkpoint_path = resolve_path(f"{cfg.paths.checkpoint_dir}/{arch}_best.pt")
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_qwk": [],
        "val_macro_f1": [],
        "val_accuracy": [],
    }

    best_val_metrics: Dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_training=True,
        )
        scheduler.step()

        val_loss, y_true_val, y_pred_val = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            is_training=False,
        )

        val_metrics = calculate_metrics(y_true_val, y_pred_val)
        val_qwk = val_metrics["qwk"]
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_qwk"].append(round(val_qwk, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Checkpoint if best QWK
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0
            best_val_metrics = val_metrics

            checkpoint_data = {
                "epoch": epoch,
                "architecture": arch,
                "state_dict": model.state_dict(),
                "num_classes": cfg.model.num_classes,
                "class_labels": datasets_cfg.aptos.class_labels,
                "input_size": cfg.model.input_size,
                "norm_mean": cfg.model.norm_mean,
                "norm_std": cfg.model.norm_std,
                "seed": seed,
                "best_val_metrics": val_metrics,
                "history": history,
                "smoke_test": smoke_test,
                "timestamp": datetime.now().isoformat(),
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            # Also save a copy inside the run folder for auditability
            torch.save(checkpoint_data, run_dir / f"{arch}_checkpoint_best.pt")
            logger.info(f"--> Saved new best checkpoint (Val QWK: {val_qwk:.4f}) to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience and not smoke_test:
                logger.info(f"Early stopping triggered after {patience} epochs without QWK improvement.")
                break

    # Save training curves plot
    curves_plot_path = run_dir / "training_curves.png"
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss", color="#1F77B4")
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss", color="#FF7F0E")
    plt.title(f"Training & Validation Loss ({arch})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history["val_qwk"]) + 1), history["val_qwk"], label="Val QWK", color="#2CA02C")
    plt.plot(range(1, len(history["val_macro_f1"]) + 1), history["val_macro_f1"], label="Val Macro F1", color="#D62728")
    plt.title("Validation Performance Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(curves_plot_path, dpi=200)
    plt.close()

    # Save run summary JSON
    run_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "architecture": arch,
        "epochs_trained": len(history["train_loss"]),
        "smoke_test": smoke_test,
        "best_val_qwk": float(best_qwk),
        "best_val_metrics": best_val_metrics,
        "history": history,
        "best_checkpoint_path": str(best_checkpoint_path),
        "device": str(device),
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    logger.info(f"Training run completed! Artifacts saved in: {run_dir}")
    return run_summary

def main():
    parser = argparse.ArgumentParser(description="RetinaTrust Model Training CLI")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"], help="Architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--smoke-test", action="store_true", help="Run low-resource smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to default.yaml")
    args = parser.parse_args()

    try:
        train_model(
            config_path=args.config,
            smoke_test=args.smoke_test,
            override_epochs=args.epochs,
            override_batch_size=args.batch_size,
            override_arch=args.arch,
            override_device=args.device,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
