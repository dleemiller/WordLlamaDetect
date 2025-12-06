"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from wldetect.config.loader import load_model_config, save_model_config
from wldetect.config.models import (
    DatasetConfig,
    InferenceConfig,
    ModelConfig,
    SingleModelConfig,
    TrainingConfig,
)


def test_single_model_config():
    """Test single model configuration."""
    config = ModelConfig(
        model=SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
        ),
        languages={"en": 0, "es": 1},
    )

    assert config.hidden_dim == 128
    assert config.n_languages == 2
    assert not config.is_multi_model


def test_multi_model_config():
    """Test multi-model configuration."""
    config = ModelConfig(
        models=[
            SingleModelConfig(name="test/model1", type="test", hidden_dim=128),
            SingleModelConfig(name="test/model2", type="test", hidden_dim=256),
        ],
        languages={"en": 0, "es": 1},
    )

    assert config.hidden_dim == 128 + 256
    assert config.n_languages == 2
    assert config.is_multi_model


def test_invalid_model_config():
    """Test that invalid config raises error."""
    with pytest.raises(ValueError):
        # No model or models specified
        ModelConfig(languages={"en": 0})


def test_invalid_language_indices():
    """Test that non-sequential language indices raise error."""
    with pytest.raises(ValueError):
        ModelConfig(
            model=SingleModelConfig(name="test/model", type="test", hidden_dim=128),
            languages={"en": 0, "es": 2},  # Missing index 1
        )


def test_load_model_config_yaml():
    """Test loading model config from YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_content = {
            "model": {
                "name": "test/model",
                "type": "test",
                "hidden_dim": 128,
            },
            "languages": {"en": 0, "es": 1},
        }
        yaml.dump(yaml_content, f)
        temp_path = f.name

    try:
        config = load_model_config(temp_path)
        assert config.hidden_dim == 128
        assert config.n_languages == 2
    finally:
        Path(temp_path).unlink()


def test_save_and_load_model_config():
    """Test saving and loading model config."""
    config = ModelConfig(
        model=SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
        ),
        languages={"en": 0, "es": 1, "fr": 2},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        save_model_config(config, config_path)

        loaded_config = load_model_config(config_path)
        assert loaded_config.hidden_dim == config.hidden_dim
        assert loaded_config.n_languages == config.n_languages
        assert loaded_config.languages == config.languages


def test_training_config():
    """Test training configuration."""
    config = TrainingConfig(
        model_config_path="path/to/model.yaml",
    )

    assert config.dataset.name == "laurievb/OpenLID-v2"
    assert config.dataset.shuffle_seed == 42
    assert config.training.batch_size == 32
    assert config.training.learning_rate == 1e-3
    assert config.training.loss == "cross_entropy"
    assert config.training.focal_gamma == 2.0
    assert config.training.focal_alpha is None
    assert config.output.checkpoint_every_steps is None
    assert config.evaluation.flores_eval_every_steps is None
    assert config.evaluation.flores_split == "dev"
    assert config.evaluation.flores_batch_size is None
    assert config.evaluation.flores_hf_dataset == "openlanguagedata/flores_plus"
    assert config.evaluation.flores_cache_dir is None
    assert config.evaluation.metrics == ["accuracy", "f1_macro", "f1_weighted", "confusion_matrix"]


def test_inference_config():
    """Test inference configuration."""
    config = InferenceConfig(
        max_sequence_length=256,
        pooling="average",
    )

    assert config.max_sequence_length == 256
    assert config.pooling == "average"


def test_inference_config_invalid_pooling():
    """Test that invalid pooling raises error."""
    with pytest.raises(ValueError):
        InferenceConfig(pooling="invalid")


def test_dataset_config():
    """Test dataset configuration."""
    config = DatasetConfig(
        filter_languages=True,
        max_samples_per_language=5000,
    )

    assert config.filter_languages is True
    assert config.max_samples_per_language == 5000
