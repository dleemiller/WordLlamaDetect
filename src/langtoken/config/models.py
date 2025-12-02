"""Pydantic models for configuration schemas."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class InferenceConfig(BaseModel):
    """Inference settings."""

    max_sequence_length: int = Field(default=512, gt=0)
    pooling: Literal["max", "average", "geometric", "harmonic", "logsumexp"] = "max"


class SingleModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str = Field(..., description="HuggingFace model name")
    type: str = Field(..., description="Model type (e.g., gemma3, qwen3)")
    hidden_dim: int = Field(..., gt=0, description="Hidden dimension size")
    shard_pattern: str = Field(
        default="model-*.safetensors", description="Pattern to find embedding shard"
    )
    embedding_layer_name: str = Field(
        default="model.embed_tokens.weight", description="Name of embedding layer in state dict"
    )


class MultiModelConfig(BaseModel):
    """Configuration for multiple models (concatenation)."""

    models: list[SingleModelConfig] = Field(..., min_length=1)

    @field_validator("models")
    @classmethod
    def validate_models(cls, v: list[SingleModelConfig]) -> list[SingleModelConfig]:
        """Ensure at least one model is provided."""
        if not v:
            raise ValueError("At least one model must be provided")
        return v

    @property
    def combined_hidden_dim(self) -> int:
        """Calculate total hidden dimension from all models."""
        return sum(model.hidden_dim for model in self.models)


class ModelConfig(BaseModel):
    """Top-level model configuration."""

    model: SingleModelConfig | None = None
    models: list[SingleModelConfig] | None = None
    languages: dict[str, int] = Field(..., description="Language code to index mapping")
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: dict[str, int]) -> dict[str, int]:
        """Ensure languages is not empty and indices are sequential."""
        if not v:
            raise ValueError("At least one language must be specified")
        indices = sorted(v.values())
        if indices != list(range(len(indices))):
            raise ValueError("Language indices must be sequential starting from 0")
        return v

    def model_post_init(self, __context) -> None:
        """Validate that either model or models is provided, but not both."""
        if self.model is None and self.models is None:
            raise ValueError("Either 'model' or 'models' must be provided")
        if self.model is not None and self.models is not None:
            raise ValueError("Cannot specify both 'model' and 'models'")

    @property
    def is_multi_model(self) -> bool:
        """Check if this is a multi-model configuration."""
        return self.models is not None

    @property
    def hidden_dim(self) -> int:
        """Get the total hidden dimension."""
        if self.model:
            return self.model.hidden_dim
        elif self.models:
            return sum(m.hidden_dim for m in self.models)
        else:
            raise ValueError("No model configuration found")

    @property
    def all_models(self) -> list[SingleModelConfig]:
        """Get all models as a list (single or multiple)."""
        if self.model:
            return [self.model]
        elif self.models:
            return self.models
        else:
            raise ValueError("No model configuration found")

    @property
    def n_languages(self) -> int:
        """Get the number of languages."""
        return len(self.languages)


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(default="laurievb/OpenLID-v2")
    train_split: str = Field(default="train")
    val_split: str = Field(default="validation")
    test_split: str = Field(default="test")
    filter_languages: bool = Field(
        default=True, description="Filter to only model's supported languages"
    )
    max_samples_per_language: int | None = Field(
        default=None, gt=0, description="Max samples per language (for balancing)"
    )
    shuffle_seed: int | None = Field(
        default=42, description="Seed to shuffle training split before loading (set null to skip)"
    )


class ProjectionConfig(BaseModel):
    """Projection layer configuration."""

    dropout: float = Field(default=0.1, ge=0.0, le=1.0)


class TrainingHyperparameters(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    epochs: int = Field(default=10, gt=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    loss: Literal["cross_entropy", "focal"] = Field(
        default="cross_entropy", description="Loss function to use during training"
    )
    focal_gamma: float = Field(default=2.0, ge=0.0, description="Focusing parameter for focal loss")
    focal_alpha: float | list[float] | None = Field(
        default=None,
        description="Optional class weighting for focal loss (float or list aligned to classes)",
    )
    weight_decay: float = Field(default=1e-5, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    momentum: float = Field(default=0.9, ge=0, le=1, description="Momentum for SGD optimizer")
    num_workers: int = Field(default=4, ge=0, description="Number of DataLoader worker processes")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)

    # Learning rate scheduler
    scheduler: Literal["none", "cosine", "cosine_warmup"] | None = Field(
        default="cosine_warmup", description="Learning rate scheduler type"
    )
    warmup_steps: int = Field(
        default=500, ge=0, description="Number of warmup steps (only for cosine_warmup)"
    )
    min_lr_ratio: float = Field(
        default=0.1, gt=0, le=1, description="Minimum LR as ratio of max LR (for cosine schedulers)"
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[Literal["accuracy", "f1_macro", "f1_weighted", "confusion_matrix"]] = Field(
        default=["accuracy", "f1_macro", "f1_weighted", "confusion_matrix"]
    )
    flores_eval_every_steps: int | None = Field(
        default=None,
        gt=0,
        description="Run FLORES-200 evaluation every N training steps (null to disable)",
    )
    flores_split: Literal["dev", "devtest"] = Field(
        default="dev", description="FLORES-200 split to use for periodic evaluation"
    )
    flores_batch_size: int | None = Field(
        default=None,
        gt=0,
        description="Override FLORES-200 evaluation batch size (defaults to training batch size)",
    )
    flores_hf_dataset: str = Field(
        default="openlanguagedata/flores_plus",
        description="HuggingFace dataset name for FLORES-200",
    )
    flores_cache_dir: str | None = Field(
        default=None, description="Optional cache dir for HuggingFace FLORES dataset"
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    artifacts_dir: str = Field(default="artifacts/")
    checkpoint_dir: str = Field(default="artifacts/checkpoints/")
    tensorboard_dir: str = Field(default="runs/")
    projection_matrix_name: str = Field(default="projection.safetensors")
    config_name: str = Field(default="model_config.yaml")
    checkpoint_every_steps: int | None = Field(
        default=None,
        gt=0,
        description="Save checkpoints every N steps (set to null to disable step checkpoints)",
    )


class TrainingConfig(BaseModel):
    """Top-level training configuration."""

    model_config_path: str = Field(..., description="Path to model config YAML")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingHyperparameters = Field(default_factory=TrainingHyperparameters)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
