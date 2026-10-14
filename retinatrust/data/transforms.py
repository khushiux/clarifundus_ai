from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torchvision.transforms as T

from retinatrust.data.fundus_preprocessing import ben_graham_preprocessing

class BenGrahamTransform:
    """Torchvision-compatible transform applying circular crop & Ben Graham normalization."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), apply_clahe: bool = True):
        self.target_size = target_size
        self.apply_clahe = apply_clahe

    def __call__(self, img: Image.Image) -> Image.Image:
        processed_np = ben_graham_preprocessing(
            img,
            target_size=self.target_size,
            apply_clahe=self.apply_clahe,
        )
        return Image.fromarray(processed_np)

def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    use_ben_graham: bool = False,
) -> T.Compose:
    """Builds conservative, domain-appropriate data augmentations for retinal fundus images."""
    mean = norm_mean or [0.485, 0.456, 0.406]
    std = norm_std or [0.229, 0.224, 0.225]

    transforms_list = []
    if use_ben_graham:
        transforms_list.append(BenGrahamTransform(target_size=input_size, apply_clahe=True))
    else:
        transforms_list.append(T.Resize(input_size))

    transforms_list.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)

def get_eval_transforms(
    input_size: Tuple[int, int] = (224, 224),
    norm_mean: Optional[List[float]] = None,
    norm_std: Optional[List[float]] = None,
    use_ben_graham: bool = False,
) -> T.Compose:
    """Builds deterministic preprocessing pipeline for validation, testing, and inference."""
    mean = norm_mean or [0.485, 0.456, 0.406]
    std = norm_std or [0.229, 0.224, 0.225]

    transforms_list = []
    if use_ben_graham:
        transforms_list.append(BenGrahamTransform(target_size=input_size, apply_clahe=True))
    else:
        transforms_list.append(T.Resize(input_size))

    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return T.Compose(transforms_list)
