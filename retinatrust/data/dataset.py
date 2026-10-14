from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# Allow loading of truncated images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RetinaDataset(Dataset):
    """PyTorch Dataset for retinal fundus photographs.

    Loads images lazily from pre-computed manifests, converts them to RGB,
    and applies domain-specific transforms.
    """

    def __init__(
        self,
        manifest: Union[str, Path, pd.DataFrame],
        transform: Optional[Callable] = None,
    ):
        """Args:

            manifest: Path to manifest CSV or pre-loaded pandas DataFrame.
            transform: Optional torchvision transform pipeline.
        """
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

        required_cols = {"id_code", "label", "image_path"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Manifest missing required columns: {required_cols - set(self.df.columns)}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label = int(row["label"])
        id_code = str(row["id_code"])

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "id_code": id_code,
            "image_path": str(img_path),
        }

def get_dataloaders(
    train_manifest: Union[str, Path],
    val_manifest: Union[str, Path],
    test_manifest: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Constructs standard PyTorch DataLoaders for training, validation, and testing."""
    train_dataset = RetinaDataset(train_manifest, transform=train_transform)
    val_dataset = RetinaDataset(val_manifest, transform=eval_transform)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        ),
    }

    if test_manifest:
        test_dataset = RetinaDataset(test_manifest, transform=eval_transform)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    return loaders
