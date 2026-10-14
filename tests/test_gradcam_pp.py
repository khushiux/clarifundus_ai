import numpy as np
import pytest
import torch
from retinatrust.inference.explainability import GradCAMPlusPlus, generate_explanation
from retinatrust.models.build_model import build_model

def test_gradcam_pp_heatmap_generation():
    model, target_layer = build_model("resnet18", pretrained=False)
    cam_engine = GradCAMPlusPlus(model, target_layer)

    dummy_tensor = torch.randn(1, 3, 224, 224)
    heatmap = cam_engine.generate_heatmap(dummy_tensor, target_class=2, threshold=0.1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0
    cam_engine.remove_hooks()

def test_generate_explanation_end_to_end():
    model, target_layer = build_model("resnet18", pretrained=False)
    dummy_tensor = torch.randn(1, 3, 224, 224)
    dummy_orig = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    res = generate_explanation(
        model=model,
        target_layer=target_layer,
        input_tensor=dummy_tensor,
        original_image=dummy_orig,
        target_class=1,
        use_gradcam_pp=True,
    )

    assert res["success"] is True
    assert res["blended_image"] is not None
    assert res["blended_image"].shape == (224, 224, 3)
