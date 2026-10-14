from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

def crop_circular_fundus(image: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """Crops the fundus photograph to the bounding box of the active retinal circle,

    removing redundant black borders.

    Args:
        image: RGB NumPy array.
        tolerance: Pixel intensity threshold to consider as non-background.

    Returns:
        Cropped RGB NumPy array.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance

    # If the mask is mostly empty, return original
    if not np.any(mask):
        return image

    # Find bounding box coordinates
    row_sum = np.sum(mask, axis=1)
    col_sum = np.sum(mask, axis=0)

    rows = np.where(row_sum > 0)[0]
    cols = np.where(col_sum > 0)[0]

    if len(rows) == 0 or len(cols) == 0:
        return image

    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]

    # Add small padding if possible
    h, w = image.shape[:2]
    y1 = max(0, y1 - 2)
    y2 = min(h, y2 + 2)
    x1 = max(0, x1 - 2)
    x2 = min(w, x2 + 2)

    return image[y1:y2, x1:x2]

def apply_clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB L-channel

    to enhance local microvascular contrast without distorting color balance.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def ben_graham_preprocessing(
    image: Union[np.ndarray, Image.Image, Path, str],
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = True,
    scale_radius: int = 300,
) -> np.ndarray:
    """Standardizes illumination and enhances microaneurysms using Ben Graham's method.

    1. Crops black borders to active retinal disk.
    2. Resizes to target dimension.
    3. Subtracts local Gaussian-blurred background mean illumination:
       Processed = 4 * Image - 4 * GaussianBlur(Image) + 128
    4. Applies circular mask to suppress peripheral edge artifacts.
    5. Optionally applies CLAHE for lesion contrast.

    Args:
        image: Filepath, PIL Image, or RGB NumPy array.
        target_size: Tuple of (height, width).
        apply_clahe: Whether to apply CLAHE enhancement.
        scale_radius: Base radius scaling parameter.

    Returns:
        Processed RGB NumPy array of shape [height, width, 3], uint8.
    """
    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    # 1. Circular Crop
    cropped = crop_circular_fundus(img_np)

    # 2. Resize to target size
    h_target, w_target = target_size
    resized = cv2.resize(cropped, (w_target, h_target), interpolation=cv2.INTER_AREA)

    # 3. Ben Graham Illumination Normalization
    sigma = max(1, int(w_target / 30.0))
    blurred = cv2.GaussianBlur(resized, (0, 0), sigma)
    # Weighted difference: 4 * img - 4 * blur + 128
    high_pass = cv2.addWeighted(resized, 4.0, blurred, -4.0, 128)

    # 4. Circular Masking (90% radius)
    center = (int(w_target / 2), int(h_target / 2))
    radius = int(min(w_target, h_target) * 0.48)
    mask = np.zeros((h_target, w_target), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)

    # Apply mask with neutral gray 128 background
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    normalized = high_pass * mask_3d + 128 * (1 - mask_3d)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # 5. Optional CLAHE
    if apply_clahe:
        normalized = apply_clahe_enhancement(normalized)

    return normalized
