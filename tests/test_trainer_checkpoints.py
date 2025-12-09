"""Trainer checkpointing tests."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from wldetect.config import OutputConfig, TrainingConfig
from wldetect.training.trainer import Trainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, token_ids):
        # token_ids shape: (batch, seq_len); use mean as a simple feature
        features = token_ids.float().mean(dim=1, keepdim=True)
        return self.linear(features)


def test_save_checkpoint_includes_step(tmp_path: Path):
    """Checkpoint files should include the step in the filename and payload."""
    checkpoint_dir = tmp_path / "checkpoints"
    tensorboard_dir = tmp_path / "runs"

    output_config = OutputConfig(
        artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_dir=str(checkpoint_dir),
        tensorboard_dir=str(tensorboard_dir),
        checkpoint_every_steps=2,
    )
    config = TrainingConfig(model_config_path="path/to/model.yaml", output=output_config)
    model = _TinyModel()

    trainer = Trainer(model, config, device=torch.device("cpu"))
    trainer.global_step = 2
    trainer.save_checkpoint(epoch=0, step=trainer.global_step)

    expected_checkpoint = checkpoint_dir / "checkpoint_step_2.pt"
    assert expected_checkpoint.exists()

    state = torch.load(expected_checkpoint, map_location="cpu")
    assert state["step"] == 2
    assert state["epoch"] == 0

    trainer.close()
