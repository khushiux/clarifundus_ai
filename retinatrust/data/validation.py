import hashlib
from pathlib import Path
from typing import List, Optional, Set, Tuple
import pandas as pd
from PIL import Image

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

def compute_file_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of a file for duplicate detection."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def validate_csv_schema(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    valid_grades: Optional[List[int]] = None
) -> Tuple[bool, List[str]]:
    """Validates dataframe columns and label values.

    Args:
        df: Input pandas DataFrame.
        id_col: Expected identifier column name.
        label_col: Expected severity grade column name.
        valid_grades: Expected list of valid integer grades (e.g., [0, 1, 2, 3, 4]).

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: List[str] = []
    
    if id_col not in df.columns:
        errors.append(f"Missing required identifier column '{id_col}'. Available columns: {list(df.columns)}")
    if label_col not in df.columns:
        errors.append(f"Missing required label column '{label_col}'. Available columns: {list(df.columns)}")
    
    if errors:
        return False, errors

    # Check for null identifiers or labels
    null_ids = df[id_col].isnull().sum()
    if null_ids > 0:
        errors.append(f"Found {null_ids} null values in identifier column '{id_col}'.")

    null_labels = df[label_col].isnull().sum()
    if null_labels > 0:
        errors.append(f"Found {null_labels} null values in label column '{label_col}'.")

    # Check valid grades
    if valid_grades is not None:
        try:
            numeric_labels = pd.to_numeric(df[label_col].dropna(), errors="coerce")
            if numeric_labels.isnull().any():
                errors.append(f"Non-numeric values found in label column '{label_col}'.")
            else:
                invalid_vals = set(numeric_labels.astype(int)) - set(valid_grades)
                if invalid_vals:
                    errors.append(f"Labels contain invalid grade values {invalid_vals}. Expected subset of {valid_grades}.")
        except Exception as e:
            errors.append(f"Error checking grade values: {str(e)}")

    return len(errors) == 0, errors

def resolve_image_path(
    image_dir: Path,
    image_id: str,
    extensions: Optional[List[str]] = None
) -> Optional[Path]:
    """Resolves an image identifier to an existing file in image_dir.

    Handles ids that already contain extensions as well as raw ids without extension.
    """
    exts = extensions or SUPPORTED_EXTENSIONS
    raw_str = str(image_id).strip()
    
    # 1. Exact match (if already has extension)
    direct_path = image_dir / raw_str
    if direct_path.is_file():
        return direct_path

    # 2. Try adding each extension
    for ext in exts:
        candidate = image_dir / f"{raw_str}{ext}"
        if candidate.is_file():
            return candidate
        # Also try case variations like .PNG, .JPG
        candidate_upper = image_dir / f"{raw_str}{ext.upper()}"
        if candidate_upper.is_file():
            return candidate_upper

    return None

def check_image_readability(
    image_path: Path
) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str], Optional[str]]:
    """Attempts to decode image and read format and dimensions safely.

    Returns:
        Tuple of (is_readable, (width, height), format, error_message).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open for size and mode because verify() can close file
        with Image.open(image_path) as img:
            width, height = img.size
            img_format = img.format
            return True, (width, height), img_format, None
    except Exception as e:
        return False, None, None, str(e)
