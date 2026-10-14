import numpy as np
from PIL import Image
import pytest
from retinatrust.inference.image_quality import check_image_quality
from retinatrust.config import ImageQualityConfig

def test_quality_clean_image():
    # Sharp gradient image
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :150] = 50
    arr[:, 150:] = 200
    res = check_image_quality(arr)
    assert res.is_decodable is True
    assert res.is_acceptable is True
    assert len(res.errors) == 0

def test_quality_blurry_image():
    # Completely uniform image -> zero laplacian variance -> blur warning
    uniform = np.full((300, 300, 3), 128, dtype=np.uint8)
    res = check_image_quality(uniform)
    assert res.is_decodable is True
    assert any("blur" in w.lower() for w in res.warnings)

def test_quality_dark_image():
    # Extremely dark image
    dark = np.full((300, 300, 3), 10, dtype=np.uint8)
    res = check_image_quality(dark)
    assert res.is_decodable is True
    assert any("underexposed" in w.lower() or "dark" in w.lower() for w in res.warnings)

def test_quality_low_resolution():
    # Below min resolution (e.g. 50x50)
    small = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    res = check_image_quality(small)
    assert any("resolution" in w.lower() for w in res.warnings)

def test_quality_corrupt_bytes():
    corrupt = b"random_corrupted_data"
    res = check_image_quality(corrupt)
    assert res.is_decodable is False
    assert res.is_acceptable is False
    assert len(res.errors) > 0
