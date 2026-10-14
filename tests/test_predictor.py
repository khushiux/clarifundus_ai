import numpy as np
import pytest
from PIL import Image
from retinatrust.config import resolve_path
from retinatrust.inference.predictor import RetinaPredictor, PredictionResult

def test_predictor_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        RetinaPredictor(checkpoint_path="non_existent_model_checkpoint.pt")

def test_predictor_valid_inference():
    ckpt_path = resolve_path("artifacts/models/resnet18_best.pt")
    if not ckpt_path.is_file():
        pytest.skip("Test requires trained model checkpoint.")

    predictor = RetinaPredictor(checkpoint_path=str(ckpt_path), device_name="cpu")
    
    # Create synthetic test fundus image
    test_img = np.random.randint(50, 200, (250, 250, 3), dtype=np.uint8)
    
    result = predictor.predict(image_input=test_img, generate_cam=True)

    assert isinstance(result, PredictionResult)
    assert 0 <= result.predicted_grade <= 4
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.class_probabilities) == 5
    assert result.architecture == "resnet18"
    assert result.gradcam_available is True
    assert result.gradcam_overlay is not None
    assert result.gradcam_overlay.shape[:2] == (250, 250)

def test_predictor_rejects_corrupted_input():
    ckpt_path = resolve_path("artifacts/models/resnet18_best.pt")
    if not ckpt_path.is_file():
        pytest.skip("Test requires trained model checkpoint.")

    predictor = RetinaPredictor(checkpoint_path=str(ckpt_path), device_name="cpu")

    with pytest.raises(ValueError):
        predictor.predict(image_input=b"INVALID_CORRUPT_BYTES")
