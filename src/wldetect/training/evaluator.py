"""Evaluation metrics for language detection."""

import logging
from pathlib import Path

from wldetect.training.model import LanguageDetectionModel

logger = logging.getLogger("wldetect")


def save_projection_matrix(
    model: LanguageDetectionModel,
    output_path: str,
) -> None:
    """Save projection matrix and token weights to safetensors.

    Args:
        model: Trained model
        output_path: Output path for safetensors file
    """
    from pathlib import Path

    from safetensors.numpy import save_file

    # Get projection matrix (n_languages, hidden_dim)
    weight = model.get_projection_matrix().cpu().numpy()
    bias = model.get_projection_bias().cpu().numpy()
    token_weights = model.get_token_weights().cpu().numpy()

    # Save to safetensors
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "weight": weight,
            "bias": bias,
            "token_weights": token_weights,
        },
        str(output_path),
    )

    logger.info(f"Saved projection matrix and token weights to {output_path}")
    logger.info(
        f"  Shape: weight={weight.shape}, bias={bias.shape}, token_weights={token_weights.shape}"
    )


def save_lookup_table(
    model: LanguageDetectionModel,
    model_config,
    output_dir: str | Path,
) -> Path:
    """Generate and save fp8_e4m3fn lookup table.

    Args:
        model: Trained model
        model_config: Model configuration
        output_dir: Output directory for lookup table

    Returns:
        Path to saved lookup table file
    """

    from wldetect.embeddings import EmbeddingsManager
    from wldetect.training.lookup_table import (
        compute_lookup_table,
    )
    from wldetect.training.lookup_table import (
        save_lookup_table as save_lookup_table_file,
    )

    logger.info("Generating fp8_e4m3fn lookup table...")

    # Load embeddings from cache
    embeddings_manager = EmbeddingsManager(model_config, cache_dir="artifacts/embeddings")
    embeddings = embeddings_manager.load_cached_embeddings()  # (vocab_size, hidden_dim)
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Get trained parameters (convert to numpy)
    projection_weight = model.get_projection_matrix().cpu().numpy()  # (n_langs, hidden_dim)
    projection_bias = model.get_projection_bias().cpu().numpy()  # (n_langs,)
    token_weights = model.get_token_weights().cpu().numpy()  # (vocab_size, 1)

    # Compute lookup table
    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Save as fp8_e4m3fn
    saved_path = save_lookup_table_file(
        lookup_table_fp32=lookup_table,
        output_dir=output_dir,
        base_name="lookup_table",
    )

    return saved_path
