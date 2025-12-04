"""YAML configuration loaders."""

from pathlib import Path

import yaml

from wldetect.config.models import ModelConfig, TrainingConfig


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")

    return data


def load_model_config(path: str | Path) -> ModelConfig:
    """Load model configuration from YAML file.

    Args:
        path: Path to model config YAML

    Returns:
        Validated ModelConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If config is invalid
    """
    data = load_yaml(path)
    return ModelConfig(**data)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load training configuration from YAML file.

    Args:
        path: Path to training config YAML

    Returns:
        Validated TrainingConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If config is invalid
    """
    data = load_yaml(path)
    return TrainingConfig(**data)


def save_model_config(config: ModelConfig, path: str | Path) -> None:
    """Save model configuration to YAML file.

    Args:
        config: ModelConfig instance
        path: Output path for YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding None values
    data = config.model_dump(exclude_none=True)

    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
