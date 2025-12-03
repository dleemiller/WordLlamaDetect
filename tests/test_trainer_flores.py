"""Trainer FLORES evaluation tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from langtoken.config.models import (
    EvaluationConfig,
    OutputConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
from langtoken.training.trainer import Trainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, token_ids):
        features = token_ids.float().mean(dim=1, keepdim=True)
        return self.linear(features)


def test_get_flores_loader_without_tokenizer(tmp_path: Path):
    """FLORES loader should return None when tokenizer is not provided."""
    output_config = OutputConfig(
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=str(tmp_path / "runs"),
    )
    eval_config = EvaluationConfig(
        flores_split="dev",
        flores_hf_dataset="openlanguagedata/flores_plus",
    )
    training_params = TrainingHyperparameters(batch_size=32, learning_rate=0.001, epochs=1)
    config = TrainingConfig(
        model_config_path="path/to/model.yaml",
        output=output_config,
        evaluation=eval_config,
        training=training_params,
    )
    model = _TinyModel()

    trainer = Trainer(model, config, device=torch.device("cpu"))
    # Without tokenizer, should return None
    result = trainer._get_flores_loader()
    assert result is None
    trainer.close()


def test_get_flores_loader_uses_hf_dataset(tmp_path: Path):
    """FLORES loader should use HuggingFace dataset parameters."""
    output_config = OutputConfig(
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=str(tmp_path / "runs"),
    )
    eval_config = EvaluationConfig(
        flores_split="dev",
        flores_hf_dataset="openlanguagedata/flores_plus",
        flores_cache_dir=str(tmp_path / "cache"),
    )
    training_params = TrainingHyperparameters(batch_size=32, learning_rate=0.001, epochs=1)
    config = TrainingConfig(
        model_config_path="path/to/model.yaml",
        output=output_config,
        evaluation=eval_config,
        training=training_params,
    )
    model = _TinyModel()

    # Create mock tokenizer and model_config
    mock_tokenizer = MagicMock()
    mock_model_config = MagicMock()
    mock_model_config.languages = {"eng_Latn": 0, "spa_Latn": 1}
    mock_model_config.inference.max_sequence_length = 512

    trainer = Trainer(
        model,
        config,
        device=torch.device("cpu"),
        tokenizer=mock_tokenizer,
        model_config=mock_model_config,
    )

    # Mock the FLORES dataset functions
    with (
        patch("langtoken.training.trainer.create_flores_dataset") as mock_create,
        patch("langtoken.training.trainer.get_flores_language_distribution") as mock_get_dist,
    ):
        # Setup mocks to return minimal valid data
        mock_create.return_value = ([], {}, set())
        mock_get_dist.return_value = {}

        # Call should not raise AttributeError for flores_dir or flores_source
        _ = trainer._get_flores_loader()

        # Verify the functions were called with correct parameters (HF-only)
        mock_create.assert_called_once_with(
            mock_model_config.languages,
            "dev",
            hf_dataset="openlanguagedata/flores_plus",
            cache_dir=str(tmp_path / "cache"),
        )
        mock_get_dist.assert_called_once_with(
            mock_model_config.languages,
            "dev",
            hf_dataset="openlanguagedata/flores_plus",
            cache_dir=str(tmp_path / "cache"),
        )

    trainer.close()


def test_evaluation_config_no_legacy_attributes():
    """EvaluationConfig should not have flores_dir or flores_source attributes."""
    eval_config = EvaluationConfig(
        flores_split="dev",
        flores_hf_dataset="openlanguagedata/flores_plus",
    )

    # These attributes should not exist
    assert not hasattr(eval_config, "flores_dir")
    assert not hasattr(eval_config, "flores_source")

    # These should exist
    assert hasattr(eval_config, "flores_split")
    assert hasattr(eval_config, "flores_hf_dataset")
    assert hasattr(eval_config, "flores_cache_dir")
