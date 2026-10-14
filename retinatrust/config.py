import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """Resolves a path string relative to project root if not absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    base = base_dir or PROJECT_ROOT
    return (base / path).resolve()

class ModelConfig(BaseModel):
    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 5
    input_size: Tuple[int, int] = (224, 224)
    norm_mean: List[float] = [0.485, 0.456, 0.406]
    norm_std: List[float] = [0.229, 0.224, 0.225]

class TrainingConfig(BaseModel):
    epochs: int = 15
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    class_weighting: str = "balanced"  # 'balanced' or 'none'
    early_stopping_patience: int = 4
    seed: int = 42
    target_metric: str = "val_qwk"
    device: str = "auto"  # 'auto', 'cuda', or 'cpu'

class SmokeTestConfig(BaseModel):
    enabled: bool = False
    max_samples: int = 60
    epochs: int = 2
    batch_size: int = 4

class DataSplitConfig(BaseModel):
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratified: bool = True
    seed: int = 42

class UncertaintyConfig(BaseModel):
    min_confidence_threshold: float = 0.55
    max_entropy_threshold: float = 1.60
    temperature: float = 1.0

class ImageQualityConfig(BaseModel):
    min_width: int = 200
    min_height: int = 200
    blur_laplacian_threshold: float = 60.0
    brightness_low_threshold: float = 20.0
    brightness_high_threshold: float = 240.0

class ExplainabilityConfig(BaseModel):
    target_layer: str = "auto"
    colormap: str = "jet"
    alpha: float = 0.45

class PathsConfig(BaseModel):
    checkpoint_dir: str = "artifacts/models"
    manifest_dir: str = "artifacts/manifests"
    report_dir: str = "reports/runs"

class PreprocessingConfig(BaseModel):
    use_ben_graham: bool = True
    apply_clahe: bool = True
    clip_limit: float = 2.0

class ConformalConfig(BaseModel):
    coverage_level_90: float = 0.90
    coverage_level_95: float = 0.95
    q_hat_90: float = 0.88
    q_hat_95: float = 0.95

class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    smoke_test: SmokeTestConfig = Field(default_factory=SmokeTestConfig)
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    image_quality: ImageQualityConfig = Field(default_factory=ImageQualityConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

class DatasetCandidatePath(BaseModel):
    csv_path: str
    image_dir: str
    test_csv_path: Optional[str] = None
    test_image_dir: Optional[str] = None

class DatasetEntryConfig(BaseModel):
    name: str
    description: str
    candidate_paths: List[DatasetCandidatePath]
    id_column: str
    label_column: str
    image_extension_fallback: str = ".png"
    num_classes: int = 5
    class_labels: Dict[int, str]

class DatasetsConfig(BaseModel):
    aptos: DatasetEntryConfig
    idrid: Optional[DatasetEntryConfig] = None

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads and validates the main application config from YAML file."""
    path = resolve_path(config_path or "configs/default.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return AppConfig(**data)

def load_datasets_config(datasets_config_path: Optional[str] = None) -> DatasetsConfig:
    """Loads and validates the dataset locations and metadata schema config."""
    path = resolve_path(datasets_config_path or "configs/datasets.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Datasets configuration file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    return DatasetsConfig(**data)
