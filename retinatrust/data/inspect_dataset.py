import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from retinatrust.config import load_config, load_datasets_config, resolve_path
from retinatrust.data.validation import (
    check_image_readability,
    compute_file_md5,
    resolve_image_path,
    validate_csv_schema,
)
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.inspect_dataset")

def find_working_paths(entry_config) -> Optional[Dict[str, Path]]:
    """Iterates through candidate paths to find existing CSV and image directories."""
    for cand in entry_config.candidate_paths:
        csv_p = resolve_path(cand.csv_path)
        img_p = resolve_path(cand.image_dir)
        if csv_p.is_file() and img_p.is_dir():
            test_csv_p = resolve_path(cand.test_csv_path) if cand.test_csv_path else None
            test_img_p = resolve_path(cand.test_image_dir) if cand.test_image_dir else None
            return {
                "csv_path": csv_p,
                "image_dir": img_p,
                "test_csv_path": test_csv_p,
                "test_image_dir": test_img_p,
            }
    return None

def inspect_dataset(
    dataset_key: str = "aptos",
    datasets_cfg_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    check_duplicates_limit: int = 200,
) -> Dict[str, Any]:
    """Inspects a dataset, validates columns, checks readability and duplicates,

    and outputs reports and charts without modifying source files.
    """
    datasets_cfg = load_datasets_config(datasets_cfg_path)
    entry_cfg = getattr(datasets_cfg, dataset_key, None)
    if entry_cfg is None:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available: aptos, idrid")

    working_paths = find_working_paths(entry_cfg)
    if working_paths is None:
        cand_list = [f"CSV: {c.csv_path} | Images: {c.image_dir}" for c in entry_cfg.candidate_paths]
        raise FileNotFoundError(
            f"Could not locate dataset files for '{dataset_key}'.\n"
            f"Searched candidates:\n" + "\n".join(cand_list) + "\n"
            "Please check configs/datasets.yaml or verify files exist."
        )

    csv_path = working_paths["csv_path"]
    image_dir = working_paths["image_dir"]

    logger.info(f"Inspecting dataset: {entry_cfg.name}")
    logger.info(f"CSV Path: {csv_path}")
    logger.info(f"Image Directory: {image_dir}")

    # Read CSV safely (never modifying source)
    df = pd.read_csv(csv_path)
    # Strip any whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    # Drop any unnamed completely empty columns (common in some IDRiD exports)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    id_col = entry_cfg.id_column
    label_col = entry_cfg.label_column
    valid_grades = list(entry_cfg.class_labels.keys())

    is_valid, schema_errors = validate_csv_schema(
        df=df, id_col=id_col, label_col=label_col, valid_grades=valid_grades
    )

    if not is_valid:
        error_msg = f"Schema validation failed for {csv_path}:\n" + "\n".join(schema_errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    total_rows = len(df)
    logger.info(f"Total records in CSV: {total_rows}")
    logger.info(f"CSV Columns: {list(df.columns)}")

    # Analyze class distribution
    class_counts = df[label_col].value_counts().sort_index().to_dict()
    # Ensure all 5 classes are represented in summary
    grade_distribution = {}
    for grade in valid_grades:
        grade_distribution[int(grade)] = {
            "name": entry_cfg.class_labels[grade],
            "count": int(class_counts.get(grade, 0)),
            "percentage": round(float(class_counts.get(grade, 0) / total_rows * 100), 2)
            if total_rows > 0
            else 0.0,
        }

    # Verify image paths, readability, dimensions, and duplicates
    resolved_count = 0
    missing_images = []
    unreadable_images = []
    formats_detected: Dict[str, int] = {}
    sample_dimensions: List[Tuple[int, int]] = []
    
    seen_hashes: Dict[str, str] = {}
    duplicate_pairs: List[Dict[str, str]] = []

    logger.info(f"Verifying {total_rows} image files in {image_dir}...")
    for idx, row in df.iterrows():
        raw_id = str(row[id_col])
        resolved_img = resolve_image_path(image_dir, raw_id)
        if resolved_img is None:
            missing_images.append(raw_id)
            continue

        resolved_count += 1
        # Check readability on sample of images or all if small
        if idx < 500:
            readable, dims, img_fmt, err = check_image_readability(resolved_img)
            if not readable:
                unreadable_images.append({"id": raw_id, "path": str(resolved_img), "error": err})
            else:
                if dims:
                    sample_dimensions.append(dims)
                if img_fmt:
                    formats_detected[img_fmt] = formats_detected.get(img_fmt, 0) + 1

        # Check hash duplicates up to check_duplicates_limit
        if idx < check_duplicates_limit:
            file_hash = compute_file_md5(resolved_img)
            if file_hash in seen_hashes:
                duplicate_pairs.append({
                    "original_id": seen_hashes[file_hash],
                    "duplicate_id": raw_id,
                    "hash": file_hash,
                })
            else:
                seen_hashes[file_hash] = raw_id

    # Format summary stats
    summary: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_name": entry_cfg.name,
        "csv_path": str(csv_path),
        "image_dir": str(image_dir),
        "total_records": total_rows,
        "resolved_images": resolved_count,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:10],
        "unreadable_images_count": len(unreadable_images),
        "unreadable_images_sample": unreadable_images[:10],
        "duplicate_images_detected": len(duplicate_pairs),
        "duplicate_sample": duplicate_pairs[:5],
        "grade_distribution": grade_distribution,
        "formats_detected": formats_detected,
        "average_dimensions": [
            int(sum(d[0] for d in sample_dimensions) / len(sample_dimensions)),
            int(sum(d[1] for d in sample_dimensions) / len(sample_dimensions)),
        ]
        if sample_dimensions
        else None,
    }

    # Save summary outputs
    app_cfg = load_config()
    out_dir = Path(output_dir or app_cfg.paths.manifest_dir)
    fig_dir = resolve_path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{dataset_key}_inspection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved machine-readable inspection summary to: {json_path}")

    # Generate Class Distribution Chart
    chart_path = fig_dir / f"{dataset_key}_class_distribution.png"
    plt.figure(figsize=(8, 5))
    grades = [d["name"] for d in grade_distribution.values()]
    counts = [d["count"] for d in grade_distribution.values()]
    bars = plt.bar(grades, counts, color="#2E86AB", edgecolor="#1D566E", alpha=0.9)
    plt.title(f"Class Distribution: {entry_cfg.name}", fontsize=14, pad=15)
    plt.xlabel("Diabetic Retinopathy Grade", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (total_rows * 0.01), f"{yval}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()
    logger.info(f"Saved class distribution plot to: {chart_path}")

    # Generate Markdown Report
    md_path = fig_dir / f"{dataset_key}_inspection.md"
    md_content = f"""# Inspection Report: {entry_cfg.name}

- **Dataset Identifier:** `{dataset_key}`
- **Source CSV:** `{csv_path}`
- **Image Directory:** `{image_dir}`
- **Total Records:** `{total_rows}`
- **Resolved Images:** `{resolved_count}`
- **Missing Images:** `{len(missing_images)}`
- **Unreadable Images:** `{len(unreadable_images)}`
- **Duplicate Images Detected:** `{len(duplicate_pairs)}`

## Grade Distribution
| Grade | Severity Label | Image Count | Percentage |
|:---:|:---|:---:|:---:|
"""
    for grade, data in grade_distribution.items():
        md_content += f"| {grade} | {data['name']} | {data['count']} | {data['percentage']}% |\n"

    md_content += f"""
## Technical Image Properties
- **Detected Formats:** {formats_detected}
- **Average Dimensions (Sample):** {summary['average_dimensions']}
- **Distribution Chart:** ![{dataset_key} distribution]({chart_path})
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved human-readable inspection markdown to: {md_path}")

    return summary

def main():
    parser = argparse.ArgumentParser(description="RetinaTrust Dataset Inspection Utility")
    parser.add_argument("--dataset", type=str, default="aptos", choices=["aptos", "idrid"], help="Dataset key to inspect")
    parser.add_argument("--config", type=str, default=None, help="Path to custom datasets.yaml")
    args = parser.parse_args()

    try:
        inspect_dataset(dataset_key=args.dataset, datasets_cfg_path=args.config)
        logger.info(f"Dataset inspection for '{args.dataset}' completed successfully.")
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
