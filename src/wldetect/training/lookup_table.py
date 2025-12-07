"""Lookup table generation for pre-computed language detection."""

import logging
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

logger = logging.getLogger("wldetect")


def compute_lookup_table(
    embeddings: np.ndarray,
    token_weights: np.ndarray,
    projection_weight: np.ndarray,
    projection_bias: np.ndarray,
) -> np.ndarray:
    """Compute pre-computed lookup table.

    Computes: (embeddings * token_weights) @ projection.T + bias

    Args:
        embeddings: (vocab_size, hidden_dim)
        token_weights: (vocab_size, 1)
        projection_weight: (n_langs, hidden_dim)
        projection_bias: (n_langs,)

    Returns:
        lookup_table: (vocab_size, n_langs)
    """
    logger.info(
        f"Computing lookup table: embeddings {embeddings.shape} * token_weights {token_weights.shape}"
    )

    # Apply token weights: (vocab_size, hidden_dim) * (vocab_size, 1) -> (vocab_size, hidden_dim)
    weighted_embeddings = embeddings * token_weights

    logger.info(f"Applying projection: {weighted_embeddings.shape} @ {projection_weight.T.shape}")

    # Project: (vocab_size, hidden_dim) @ (hidden_dim, n_langs) -> (vocab_size, n_langs)
    lookup_table = weighted_embeddings @ projection_weight.T

    # Add bias: (vocab_size, n_langs) + (n_langs,) -> (vocab_size, n_langs)
    lookup_table = lookup_table + projection_bias

    logger.info(f"Lookup table shape: {lookup_table.shape}, dtype: {lookup_table.dtype}")

    return lookup_table


def save_lookup_table_e3m4(
    lookup_table_fp32: np.ndarray,
    output_dir: str | Path,
    base_name: str = "lookup_table",
    scale_factor: float | None = None,
    metadata: dict | None = None,
    token_mask: np.ndarray | None = None,
) -> Path:
    """Save lookup table as fp8_e3m4 with scaling.

    Args:
        lookup_table_fp32: Pre-computed lookup table (fp32)
        output_dir: Output directory
        base_name: Base filename (without extension)
        scale_factor: Optional manual scale factor (auto-computed if None)
        metadata: Optional metadata dict (zero_weight_mask, min_count_threshold, etc.)
        token_mask: Optional boolean mask (vocab_size,) where False = masked token

    Returns:
        Path to saved file
    """
    import ml_dtypes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-compute scale factor if not provided (before applying mask)
    if scale_factor is None:
        max_val = np.abs(lookup_table_fp32).max()
        e3m4_max = 15.5  # E3M4 range (~±15.5)
        scale_factor = float(np.ceil(max_val / (e3m4_max * 0.9)))

    # Apply mask before quantization using scaled min FP8 E3M4 value
    if token_mask is not None:
        lookup_table_fp32 = lookup_table_fp32.copy()
        # E3M4 min value is approximately -15.5, scaled by scale_factor
        e3m4_min_scaled = -15.5 * scale_factor
        lookup_table_fp32[~token_mask] = e3m4_min_scaled

    # Scale and quantize
    if scale_factor > 1.0:
        scaled_values = lookup_table_fp32 / scale_factor
        lookup_e3m4_bytes = scaled_values.astype(ml_dtypes.float8_e3m4)
        logger.info(f"Using {scale_factor:.0f}x scaling for E3M4 quantization")
    else:
        lookup_e3m4_bytes = lookup_table_fp32.astype(ml_dtypes.float8_e3m4)

    # Save with required tensors only
    tensors = {
        "lookup_table": lookup_e3m4_bytes.view(np.uint8),
        "dtype": np.array([26], dtype=np.uint8),  # 26 = fp8_e3m4
        "shape": np.array(lookup_e3m4_bytes.shape, dtype=np.int64),
        "scale": np.array([scale_factor], dtype=np.float32),
    }

    fp8_path = output_dir / f"{base_name}_fp8_e3m4.safetensors"
    save_file(tensors, str(fp8_path))

    file_size_mb = fp8_path.stat().st_size / (1024**2)
    logger.info(f"Saved fp8_e3m4 lookup table: {fp8_path} ({file_size_mb:.1f} MB)")

    return fp8_path


def compute_lookup_table_from_model(
    model,
    model_config,
    cache_dir: str | Path = "artifacts/embeddings",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute lookup table directly from a trained model and cached embeddings.

    Returns:
        tuple: (lookup_table, token_mask) where token_mask is None if no masking
    """
    from wldetect.embeddings import EmbeddingsManager

    embeddings_manager = EmbeddingsManager(model_config, cache_dir=cache_dir)
    embeddings = embeddings_manager.load_cached_embeddings()
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    projection_weight = model.get_projection_matrix().cpu().numpy()
    projection_bias = model.get_projection_bias().cpu().numpy()
    token_weights = model.get_token_weights().cpu().numpy()

    # Extract token mask if present
    token_mask = getattr(model, "token_mask", None)
    mask_np: np.ndarray | None = None
    if token_mask is not None:
        mask_np = token_mask.detach().cpu().numpy().reshape(-1).astype(bool)

    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Return lookup table and mask separately
    # Mask will be applied in save functions using appropriate FP8 min values
    return lookup_table, mask_np


def save_lookup_table_e3m4_from_model(
    model,
    model_config,
    output_dir: str | Path,
    cache_dir: str | Path = "artifacts/embeddings",
    scale_factor: float | None = None,
    metadata: dict | None = None,
    base_name: str = "lookup_table",
) -> Path:
    """Generate and save fp8_e3m4 lookup table (with optional scaling/metadata)."""
    lookup_table, token_mask = compute_lookup_table_from_model(
        model=model,
        model_config=model_config,
        cache_dir=cache_dir,
    )

    if token_mask is not None:
        n_masked = (~token_mask).sum()
        logger.info(f"Applying token mask: {n_masked:,} tokens will be set to FP8 E3M4 min value")

    return save_lookup_table_e3m4(
        lookup_table_fp32=lookup_table,
        output_dir=output_dir,
        base_name=base_name,
        scale_factor=scale_factor,
        metadata=metadata,
        token_mask=token_mask,
    )


def save_projection_matrix(
    model,
    output_path: str | Path,
) -> None:
    """Save projection matrix, bias, and token weights for inspection/compatibility."""
    from safetensors.numpy import save_file

    weight = model.get_projection_matrix().cpu().numpy()
    bias = model.get_projection_bias().cpu().numpy()
    token_weights = model.get_token_weights().cpu().numpy()

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
