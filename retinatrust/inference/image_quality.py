from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from retinatrust.config import ImageQualityConfig

class ImageQualityResult:
    """Encapsulates results of technical heuristic image quality checks."""

    def __init__(
        self,
        is_acceptable: bool,
        is_decodable: bool,
        dimensions: Tuple[int, int],
        channels: int,
        blur_score: float,
        mean_brightness: float,
        underexposed_ratio: float,
        overexposed_ratio: float,
        warnings: List[str],
        errors: List[str],
    ):
        self.is_acceptable = is_acceptable
        self.is_decodable = is_decodable
        self.dimensions = dimensions
        self.channels = channels
        self.blur_score = blur_score
        self.mean_brightness = mean_brightness
        self.underexposed_ratio = underexposed_ratio
        self.overexposed_ratio = overexposed_ratio
        self.warnings = warnings
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "is_decodable": self.is_decodable,
            "dimensions": {"width": self.dimensions[0], "height": self.dimensions[1]},
            "channels": self.channels,
            "blur_laplacian_variance": round(self.blur_score, 2),
            "mean_brightness": round(self.mean_brightness, 2),
            "underexposed_ratio": round(self.underexposed_ratio, 4),
            "overexposed_ratio": round(self.overexposed_ratio, 4),
            "warnings": self.warnings,
            "errors": self.errors,
            "disclaimer": (
                "Technical image quality heuristics only. "
                "Does not constitute medical adequacy or clinical certification."
            ),
        }

def check_image_quality(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    config: Optional[ImageQualityConfig] = None,
) -> ImageQualityResult:
    """Evaluates technical heuristics on a fundus image prior to inference.

    Args:
        image_input: File path, PIL Image, NumPy array (RGB/BGR), or raw byte buffer.
        config: Optional ImageQualityConfig instance with thresholds.

    Returns:
        ImageQualityResult with heuristic metrics and warning/error lists.
    """
    cfg = config or ImageQualityConfig()
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Decode / Convert input to NumPy RGB array
    np_img: Optional[np.ndarray] = None

    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, bytes):
            import io
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                np_img = image_input
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                errors.append(f"Unsupported array shape: {image_input.shape}")
        else:
            errors.append(f"Unsupported image input type: {type(image_input)}")
    except Exception as e:
        errors.append(f"Image decoding failed: {str(e)}")

    if np_img is None or len(errors) > 0:
        return ImageQualityResult(
            is_acceptable=False,
            is_decodable=False,
            dimensions=(0, 0),
            channels=0,
            blur_score=0.0,
            mean_brightness=0.0,
            underexposed_ratio=0.0,
            overexposed_ratio=0.0,
            warnings=warnings,
            errors=errors,
        )

    height, width = np_img.shape[:2]
    channels = np_img.shape[2] if np_img.ndim == 3 else 1

    # 2. Resolution Check
    if width < cfg.min_width or height < cfg.min_height:
        warnings.append(
            f"Low resolution: {width}x{height} pixels (recommended min: {cfg.min_width}x{cfg.min_height}). "
            "Fine microvascular lesions may be blurred or lost."
        )

    # 3. Blur Check via Laplacian Variance
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if laplacian_var < cfg.blur_laplacian_threshold:
        warnings.append(
            f"Potential blur detected (Laplacian variance {laplacian_var:.1f} < threshold {cfg.blur_laplacian_threshold:.1f}). "
            "Blurry fundus images may degrade diagnostic accuracy."
        )

    # 4. Exposure & Brightness Analysis
    mean_bright = float(np.mean(gray))
    total_pixels = width * height
    underexposed_count = int(np.sum(gray < cfg.brightness_low_threshold))
    overexposed_count = int(np.sum(gray > cfg.brightness_high_threshold))

    underexposed_ratio = float(underexposed_count / total_pixels)
    overexposed_ratio = float(overexposed_count / total_pixels)

    if mean_bright < 30.0 or underexposed_ratio > 0.60:
        warnings.append(
            f"Severely underexposed / dark image (mean brightness: {mean_bright:.1f}/255). "
            "Under-illuminated retinal fields may conceal dark hemorrhages."
        )
    elif mean_bright > 220.0 or overexposed_ratio > 0.40:
        warnings.append(
            f"Overexposed / flash artifact detected (mean brightness: {mean_bright:.1f}/255). "
            "Excessive glare may wash out retinal microstructures."
        )

    # Acceptable if decodable and no fatal errors
    is_acceptable = len(errors) == 0

    return ImageQualityResult(
        is_acceptable=is_acceptable,
        is_decodable=True,
        dimensions=(width, height),
        channels=channels,
        blur_score=laplacian_var,
        mean_brightness=mean_bright,
        underexposed_ratio=underexposed_ratio,
        overexposed_ratio=overexposed_ratio,
        warnings=warnings,
        errors=errors,
    )
