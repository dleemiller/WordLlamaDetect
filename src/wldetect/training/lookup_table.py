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


def save_lookup_table(
    lookup_table_fp32: np.ndarray,
    output_dir: str | Path,
    base_name: str = "lookup_table",
) -> Path:
    """Save lookup table as fp8_e4m3fn.

    Args:
        lookup_table_fp32: Pre-computed lookup table (fp32)
        output_dir: Output directory
        base_name: Base filename (without extension)

    Returns:
        Path to saved file
    """
    from wldetect.inference.quantization import quantize_fp8_e4m3fn

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quantize to fp8_e4m3fn
    lookup_fp8 = quantize_fp8_e4m3fn(lookup_table_fp32)

    # Save as uint8 view (safetensors doesn't support ml_dtypes)
    fp8_path = output_dir / f"{base_name}_fp8_e4m3fn.safetensors"
    save_file(
        {
            "lookup_table": lookup_fp8.view(np.uint8),
            "dtype": np.array([0], dtype=np.uint8),  # 0 = fp8_e4m3fn
            "shape": np.array(lookup_fp8.shape, dtype=np.int64),
        },
        str(fp8_path),
    )

    file_size_mb = fp8_path.stat().st_size / (1024**2)
    logger.info(f"Saved fp8_e4m3fn lookup table: {fp8_path} ({file_size_mb:.1f} MB)")

    return fp8_path


def save_lookup_table_e3m4(
    lookup_table_fp32: np.ndarray,
    output_dir: str | Path,
    base_name: str = "lookup_table",
    scale_factor: float | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save lookup table as fp8_e3m4 with scaling.

    Args:
        lookup_table_fp32: Pre-computed lookup table (fp32)
        output_dir: Output directory
        base_name: Base filename (without extension)
        scale_factor: Optional manual scale factor (auto-computed if None)
        metadata: Optional metadata dict (zero_weight_mask, min_count_threshold, etc.)

    Returns:
        Path to saved file
    """
    import ml_dtypes

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-compute scale factor if not provided
    if scale_factor is None:
        max_val = np.abs(lookup_table_fp32).max()
        e3m4_max = 15.5  # E3M4 range (~±15.5)
        scale_factor = float(np.ceil(max_val / (e3m4_max * 0.9)))

    # Scale and quantize
    if scale_factor > 1.0:
        scaled_values = lookup_table_fp32 / scale_factor
        lookup_e3m4_bytes = scaled_values.astype(ml_dtypes.float8_e3m4)
        logger.info(f"Using {scale_factor:.0f}x scaling for E3M4 quantization")
    else:
        lookup_e3m4_bytes = lookup_table_fp32.astype(ml_dtypes.float8_e3m4)

    # Save with metadata
    tensors = {
        "lookup_table": lookup_e3m4_bytes.view(np.uint8),
        "dtype": np.array([26], dtype=np.uint8),  # 26 = fp8_e3m4
        "shape": np.array(lookup_e3m4_bytes.shape, dtype=np.int64),
        "scale": np.array([scale_factor], dtype=np.float32),
    }

    # Add optional metadata
    if metadata:
        if "zero_weight_mask" in metadata:
            tensors["zero_weight_mask"] = metadata["zero_weight_mask"].astype(np.uint8)
        if "min_count_threshold" in metadata:
            tensors["min_count_threshold"] = np.array(
                [metadata["min_count_threshold"]], dtype=np.int32
            )

    fp8_path = output_dir / f"{base_name}_fp8_e3m4.safetensors"
    save_file(tensors, str(fp8_path))

    file_size_mb = fp8_path.stat().st_size / (1024**2)
    logger.info(f"Saved fp8_e3m4 lookup table: {fp8_path} ({file_size_mb:.1f} MB)")

    return fp8_path
