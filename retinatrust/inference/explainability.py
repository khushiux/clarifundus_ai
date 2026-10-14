from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from retinatrust.utils.logging_utils import setup_logger

logger = setup_logger("retinatrust.explainability")

class GradCAMPlusPlus:
    """Grad-CAM++ (Generalized Gradient-Weighted Class Activation Mapping).

    Computes pixel-wise weighted combinations of positive gradients using second-
    and third-order partial derivatives to accurately isolate multiple scattered
    microvascular lesions (microaneurysms, hemorrhages, cotton-wool spots).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Generates 2D normalized Grad-CAM++ heatmap array in range [0, 1]."""
        self.model.eval()
        self.model.zero_grad()

        input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        # Handle models returning tuples or dicts
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture feature maps or gradients in target layer.")

        # Gradients: [1, C, H, W], Activations: [1, C, H, W]
        grads = self.gradients
        acts = self.activations

        # Compute higher-order derivatives
        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)

        # Sum over spatial dimensions [1, C, 1, 1]
        sum_acts_grad3 = torch.sum(acts * grad_3, dim=[2, 3], keepdim=True)
        eps = 1e-8
        denominator = 2.0 * grad_2 + sum_acts_grad3 + eps
        
        # Alpha weights: [1, C, H, W]
        alpha = grad_2 / denominator
        # Zero out where gradient is zero
        alpha = torch.where(grads != 0.0, alpha, torch.zeros_like(alpha))

        # Positive gradients
        pos_grads = F.relu(grads)
        # Channel weights: [1, C, 1, 1]
        weights = torch.sum(alpha * pos_grads, dim=[2, 3], keepdim=True)

        # Weighted combination of feature activations
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)

        if max_val - min_val > 1e-8:
            norm_cam = (cam_np - min_val) / (max_val - min_val)
        else:
            norm_cam = np.zeros_like(cam_np)

        # Optional thresholding to isolate primary disease foci
        if threshold > 0.0:
            norm_cam = np.where(norm_cam >= threshold, norm_cam, 0.0)
            if np.max(norm_cam) > 0.0:
                norm_cam = norm_cam / np.max(norm_cam)

        return norm_cam

def overlay_gradcam(
    original_image: Union[np.ndarray, Image.Image, Path, str],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlays Grad-CAM++ heatmap onto the original fundus image."""
    if isinstance(original_image, (str, Path)):
        img = Image.open(original_image).convert("RGB")
        rgb_img = np.array(img)
    elif isinstance(original_image, Image.Image):
        rgb_img = np.array(original_image.convert("RGB"))
    elif isinstance(original_image, np.ndarray):
        rgb_img = original_image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(original_image)}")

    h, w = rgb_img.shape[:2]
    resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    resized_cam = np.clip(resized_cam, 0.0, 1.0)

    cam_uint8 = np.uint8(255 * resized_cam)
    colored_bgr = cv2.applyColorMap(cam_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, colored_rgb, alpha, 0)
    return blended, colored_rgb

def generate_explanation(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    original_image: Union[np.ndarray, Image.Image, Path, str],
    target_class: Optional[int] = None,
    alpha: float = 0.45,
    threshold: float = 0.0,
    use_gradcam_pp: bool = True,
) -> Dict[str, Any]:
    """Generates an end-to-end Grad-CAM++ visual explanation with threshold filtering."""
    engine = GradCAMPlusPlus(model, target_layer)
    try:
        heatmap = engine.generate_heatmap(input_tensor, target_class=target_class, threshold=threshold)
        blended, colored_map = overlay_gradcam(original_image, heatmap, alpha=alpha)
        return {
            "success": True,
            "heatmap": heatmap,
            "blended_image": blended,
            "colored_heatmap": colored_map,
            "method": "Grad-CAM++ (Multi-Lesion Higher-Order Weighting)",
            "disclaimer": (
                "Grad-CAM++ highlights regions influencing model activations. "
                "It does not establish causal medical reasoning or confirm pathology."
            ),
        }
    except Exception as e:
        logger.error(f"Grad-CAM++ generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "heatmap": None,
            "blended_image": None,
        }
    finally:
        engine.remove_hooks()
