from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models

from retinatrust.models.coral import CoralLayer, coral_logits_to_probs
from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.models")

def get_device(requested_device: str = "auto") -> torch.device:
    """Selects execution device based on hardware availability and user request."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    elif requested_device == "cpu":
        return torch.device("cpu")
    else:  # 'auto'
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU accelerator: {device_name}")
            return torch.device("cuda")
        logger.info("GPU accelerator not found. Using CPU.")
        return torch.device("cpu")

def build_model(
    architecture: str = "densenet121",
    num_classes: int = 5,
    pretrained: bool = True,
    use_coral: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module]:
    """Constructs deep learning models (DenseNet-121, Swin-T, ResNet-18, EfficientNet-B0)

    with customized classification or ordinal ranking heads and returns the model and
    its Grad-CAM / Grad-CAM++ target feature layer.

    Args:
        architecture: 'densenet121', 'swin_t', 'resnet18', or 'efficientnet_b0'.
        num_classes: Number of severity grades (default 5).
        pretrained: Whether to load ImageNet pretrained weights.
        use_coral: If True, uses CORAL ordinal rank head (K-1 tasks) instead of standard linear.
        device: PyTorch device.

    Returns:
        Tuple of (model, target_cam_layer).
    """
    arch = architecture.lower().strip()

    if arch in ("densenet121", "densenet"):
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load official online weights for DenseNet-121 ({e}). Using random init.")
            model = models.densenet121(weights=None)

        in_features = model.classifier.in_features
        if use_coral:
            model.classifier = CoralLayer(in_features=in_features, num_classes=num_classes)
        else:
            model.classifier = nn.Linear(in_features, num_classes)

        # Final norm layer of DenseNet features is ideal target for Grad-CAM++
        target_layer = model.features[-1]

    elif arch in ("swin_t", "swin_transformer", "swin"):
        try:
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            model = models.swin_t(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load official online weights for Swin-T ({e}). Using random init.")
            model = models.swin_t(weights=None)

        in_features = model.head.in_features
        if use_coral:
            model.head = CoralLayer(in_features=in_features, num_classes=num_classes)
        else:
            model.head = nn.Linear(in_features, num_classes)

        # Last block in Swin-T features
        target_layer = model.features[-1]

    elif arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load official weights for ResNet-18 ({e}). Using random init.")
            model = models.resnet18(weights=None)

        in_features = model.fc.in_features
        if use_coral:
            model.fc = CoralLayer(in_features=in_features, num_classes=num_classes)
        else:
            model.fc = nn.Linear(in_features, num_classes)

        target_layer = model.layer4[-1]

    elif arch in ("efficientnet_b0", "efficientnet"):
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load official weights for EfficientNet-B0 ({e}). Using random init.")
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        if use_coral:
            model.classifier[1] = CoralLayer(in_features=in_features, num_classes=num_classes)
        else:
            model.classifier[1] = nn.Linear(in_features, num_classes)

        target_layer = model.features[-1]

    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Supported: ['densenet121', 'swin_t', 'resnet18', 'efficientnet_b0']"
        )

    if device is not None:
        model = model.to(device)

    return model, target_layer

class RetinaEnsemble(nn.Module):
    """Heterogeneous Soft-Voting Ensemble combining DenseNet-121 and Swin-Transformer-Tiny.

    Blends calibrated predictions across complementary CNN and Transformer paradigms.
    """

    def __init__(
        self,
        densenet_model: nn.Module,
        swin_model: nn.Module,
        weight_densenet: float = 0.55,
        weight_swin: float = 0.45,
    ):
        super().__init__()
        self.densenet = densenet_model
        self.swin = swin_model
        self.w_dense = weight_densenet
        self.w_swin = weight_swin

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_dense = self.densenet(x)
        logits_swin = self.swin(x)

        probs_dense = torch.softmax(logits_dense, dim=-1)
        probs_swin = torch.softmax(logits_swin, dim=-1)

        ensemble_probs = (self.w_dense * probs_dense) + (self.w_swin * probs_swin)
        # Approximate log-odds for downstream calibration
        ensemble_logits = torch.log(torch.clamp(ensemble_probs, min=1e-8))
        return ensemble_logits, ensemble_probs
