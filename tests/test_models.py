import pytest
import torch
from retinatrust.models.build_model import build_model, RetinaEnsemble

def test_build_densenet121():
    model, target_layer = build_model(
        architecture="densenet121",
        num_classes=5,
        pretrained=False,
        use_coral=False,
    )
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)

    assert out.shape == (2, 5)
    assert target_layer is not None

def test_build_swin_t():
    model, target_layer = build_model(
        architecture="swin_t",
        num_classes=5,
        pretrained=False,
        use_coral=False,
    )
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)

    assert out.shape == (2, 5)
    assert target_layer is not None

def test_build_densenet_coral():
    model, target_layer = build_model(
        architecture="densenet121",
        num_classes=5,
        pretrained=False,
        use_coral=True,
    )
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)

    # CORAL produces K-1 = 4 rank logits
    assert out.shape == (2, 4)

def test_retina_ensemble():
    dense, _ = build_model("densenet121", pretrained=False)
    swin, _ = build_model("swin_t", pretrained=False)
    ensemble = RetinaEnsemble(dense, swin)

    dummy_x = torch.randn(2, 3, 224, 224)
    logits, probs = ensemble(dummy_x)

    assert logits.shape == (2, 5)
    assert probs.shape == (2, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)
