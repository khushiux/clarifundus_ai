import io
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from retinatrust.data.validation import (
    check_image_readability,
    compute_file_md5,
    resolve_image_path,
    validate_csv_schema,
)

def test_validate_csv_schema_valid():
    df = pd.DataFrame({
        "id_code": ["img_001", "img_002"],
        "diagnosis": [0, 4]
    })
    is_valid, errors = validate_csv_schema(df, "id_code", "diagnosis", [0, 1, 2, 3, 4])
    assert is_valid is True
    assert len(errors) == 0

def test_validate_csv_schema_invalid_column():
    df = pd.DataFrame({"wrong_col": [1, 2]})
    is_valid, errors = validate_csv_schema(df, "id_code", "diagnosis")
    assert is_valid is False
    assert any("id_code" in err for err in errors)

def test_validate_csv_schema_invalid_grades():
    df = pd.DataFrame({
        "id_code": ["img_001", "img_002"],
        "diagnosis": [0, 99]
    })
    is_valid, errors = validate_csv_schema(df, "id_code", "diagnosis", [0, 1, 2, 3, 4])
    assert is_valid is False
    assert any("invalid grade" in err.lower() for err in errors)

def test_resolve_image_path(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    sample_img = img_dir / "sample_001.png"
    sample_img.write_bytes(b"dummy image data")

    # Resolve with extension
    resolved = resolve_image_path(img_dir, "sample_001.png")
    assert resolved is not None
    assert resolved.name == "sample_001.png"

    # Resolve without extension
    resolved_no_ext = resolve_image_path(img_dir, "sample_001")
    assert resolved_no_ext is not None
    assert resolved_no_ext.name == "sample_001.png"

    # Resolve missing
    resolved_missing = resolve_image_path(img_dir, "non_existent")
    assert resolved_missing is None

def test_check_image_readability(tmp_path):
    img_dir = tmp_path / "img_test"
    img_dir.mkdir()

    # Valid image
    valid_file = img_dir / "valid.png"
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(valid_file)

    readable, dims, fmt, err = check_image_readability(valid_file)
    assert readable is True
    assert dims == (100, 100)
    assert fmt == "PNG"
    assert err is None

    # Corrupt image
    corrupt_file = img_dir / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_IMAGE_HEADER")

    c_readable, c_dims, c_fmt, c_err = check_image_readability(corrupt_file)
    assert c_readable is False
    assert c_err is not None
