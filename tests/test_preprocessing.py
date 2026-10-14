import numpy as np
import pytest
from retinatrust.data.fundus_preprocessing import (
    apply_clahe_enhancement,
    ben_graham_preprocessing,
    crop_circular_fundus,
)

def test_crop_circular_fundus():
    # Synthetic image: 300x300 black background with a bright 100x100 circle in the center
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 100:200] = 180

    cropped = crop_circular_fundus(img, tolerance=10)
    assert cropped.shape[0] < 300
    assert cropped.shape[1] < 300
    assert cropped.shape[0] >= 98
    assert cropped.shape[1] >= 98

def test_ben_graham_preprocessing_output():
    # Random synthetic fundus photograph
    img = np.random.randint(20, 220, (250, 250, 3), dtype=np.uint8)
    processed = ben_graham_preprocessing(img, target_size=(224, 224), apply_clahe=True)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8
    # Center pixel should not be pure black or zeroed
    assert np.mean(processed) > 10.0

def test_apply_clahe_enhancement():
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    clahe_img = apply_clahe_enhancement(img)

    assert clahe_img.shape == (100, 100, 3)
    assert clahe_img.dtype == np.uint8
