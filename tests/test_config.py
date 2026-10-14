import pytest
from retinatrust.config import load_config, load_datasets_config, AppConfig, DatasetsConfig

def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.architecture in ["resnet18", "densenet121"]
    assert cfg.model.num_classes == 5
    assert cfg.training.early_stopping_patience > 0
    assert cfg.data_split.train_ratio + cfg.data_split.val_ratio + cfg.data_split.test_ratio == pytest.approx(1.0)
    assert cfg.uncertainty.min_confidence_threshold > 0.0
    assert cfg.preprocessing.use_ben_graham is True
    assert cfg.conformal.coverage_level_95 == 0.95

def test_load_datasets_config():
    datasets_cfg = load_datasets_config()
    assert isinstance(datasets_cfg, DatasetsConfig)
    assert datasets_cfg.aptos is not None
    assert datasets_cfg.aptos.num_classes == 5
    assert len(datasets_cfg.aptos.class_labels) == 5
    assert 0 in datasets_cfg.aptos.class_labels
    assert 4 in datasets_cfg.aptos.class_labels
