import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from retinatrust.config import load_config, load_datasets_config, resolve_path
from retinatrust.data.inspect_dataset import find_working_paths
from retinatrust.data.validation import resolve_image_path, validate_csv_schema
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.manifest")

def create_stratified_manifests(
    dataset_key: str = "aptos",
    output_dir: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generates strictly disjoint, stratified train/validation/test manifests for the primary dataset.

    Args:
        dataset_key: Dataset identifier (default 'aptos').
        output_dir: Directory where manifests will be saved (defaults to artifacts/manifests).
        train_ratio: Proportion of training data (0.70).
        val_ratio: Proportion of validation data (0.15).
        test_ratio: Proportion of test data (0.15).
        seed: Random seed for stratified splitting.

    Returns:
        Dictionary containing DataFrames for 'train', 'val', and 'test'.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    datasets_cfg = load_datasets_config()
    entry_cfg = getattr(datasets_cfg, dataset_key, None)
    if entry_cfg is None:
        raise ValueError(f"Dataset '{dataset_key}' not found in configuration.")

    working_paths = find_working_paths(entry_cfg)
    if working_paths is None:
        raise FileNotFoundError(f"Could not locate files for dataset '{dataset_key}'.")

    csv_path = working_paths["csv_path"]
    image_dir = working_paths["image_dir"]

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = entry_cfg.id_column
    label_col = entry_cfg.label_column

    # Validate schema
    is_valid, errors = validate_csv_schema(df, id_col, label_col, list(entry_cfg.class_labels.keys()))
    if not is_valid:
        raise ValueError(f"Schema errors: {errors}")

    # Resolve all image paths
    resolved_records = []
    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        label = int(row[label_col])
        resolved_img = resolve_image_path(image_dir, raw_id)
        if resolved_img is not None:
            resolved_records.append({
                "id_code": raw_id,
                "label": label,
                "image_path": str(resolved_img.resolve()),
                "dataset": dataset_key,
            })
        else:
            logger.warning(f"Excluding missing image: {raw_id}")

    clean_df = pd.DataFrame(resolved_records)
    if len(clean_df) < 10:
        raise ValueError(f"Dataset too small to split ({len(clean_df)} records).")

    # Step 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=temp_ratio,
        stratify=clean_df["label"],
        random_state=seed,
    )

    # Step 2: Split temp into val and test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )

    # Disjointness mathematical verification
    train_ids = set(train_df["id_code"])
    val_ids = set(val_df["id_code"])
    test_ids = set(test_df["id_code"])

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            f"Data leakage detected! Overlaps: train-val={len(overlap_train_val)}, "
            f"train-test={len(overlap_train_test)}, val-test={len(overlap_val_test)}"
        )

    logger.info("Split disjointness check: PASSED (zero overlap across train, val, test).")
    logger.info(f"Split sizes - Train: {len(train_df)} ({len(train_df)/len(clean_df):.1%}), "
                f"Val: {len(val_df)} ({len(val_df)/len(clean_df):.1%}), "
                f"Test: {len(test_df)} ({len(test_df)/len(clean_df):.1%})")

    # Save manifests
    app_cfg = load_config()
    out_dir = Path(output_dir or app_cfg.paths.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_manifest.csv"
    val_path = out_dir / "val_manifest.csv"
    test_path = out_dir / "test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train manifest: {train_path}")
    logger.info(f"Saved val manifest:   {val_path}")
    logger.info(f"Saved test manifest:  {test_path}")

    return {"train": train_df, "val": val_df, "test": test_df}

def create_idrid_manifest(output_dir: Optional[str] = None) -> pd.DataFrame:
    """Generates independent test manifest for the IDRiD external dataset."""
    datasets_cfg = load_datasets_config()
    idrid_cfg = getattr(datasets_cfg, "idrid", None)
    if idrid_cfg is None:
        raise ValueError("IDRiD dataset configuration not found.")

    working_paths = find_working_paths(idrid_cfg)
    if working_paths is None:
        raise FileNotFoundError("Could not find working paths for IDRiD.")

    test_csv = working_paths.get("test_csv_path")
    test_img_dir = working_paths.get("test_image_dir")

    if not test_csv or not test_csv.is_file():
        raise FileNotFoundError(f"IDRiD testing CSV not found at {test_csv}")
    if not test_img_dir or not test_img_dir.is_dir():
        raise FileNotFoundError(f"IDRiD testing image dir not found at {test_img_dir}")

    df = pd.read_csv(test_csv)
    df.columns = [str(c).strip() for c in df.columns]
    id_col = idrid_cfg.id_column
    label_col = idrid_cfg.label_column

    records = []
    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        label = int(row[label_col])
        resolved_img = resolve_image_path(test_img_dir, raw_id)
        if resolved_img is not None:
            records.append({
                "id_code": raw_id,
                "label": label,
                "image_path": str(resolved_img.resolve()),
                "dataset": "idrid_test",
            })
        else:
            logger.warning(f"IDRiD missing test image: {raw_id}")

    idrid_df = pd.DataFrame(records)
    app_cfg = load_config()
    out_dir = Path(output_dir or app_cfg.paths.manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = out_dir / "idrid_test_manifest.csv"
    idrid_df.to_csv(manifest_p, index=False)
    logger.info(f"Saved IDRiD test manifest ({len(idrid_df)} samples) to: {manifest_p}")
    return idrid_df

def main():
    parser = argparse.ArgumentParser(description="RetinaTrust Manifest Generator")
    parser.add_argument("--dataset", type=str, default="aptos", choices=["aptos", "idrid"], help="Dataset to split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        if args.dataset == "aptos":
            create_stratified_manifests(dataset_key="aptos", seed=args.seed)
        elif args.dataset == "idrid":
            create_idrid_manifest()
        logger.info("Manifest creation completed successfully.")
    except Exception as e:
        logger.error(f"Manifest creation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
