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
    """Compute pre-computed lookup table (raw logits).

    Computes: (embeddings * token_weights) @ projection.T + bias

    Args:
        embeddings: (vocab_size, hidden_dim)
        token_weights: (vocab_size, 1)
        projection_weight: (n_langs, hidden_dim)
        projection_bias: (n_langs,)

    Returns:
        lookup_table: (vocab_size, n_langs) - raw logits in fp32
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


def compute_lookup_table_exp(
    lookup_table_fp32: np.ndarray,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute exponentiated lookup table for logsumexp pooling.

    Takes the raw logits and applies exp(), then zeros out masked tokens.
    This pre-computes the expensive exp() operation.

    During inference with logsumexp pooling:
    - Original: log(sum(exp(logits)))
    - With pre-exp: log(sum(pre_exp_values))

    Args:
        lookup_table_fp32: Raw logits (vocab_size, n_langs) in fp32
        token_mask: Optional boolean mask (vocab_size,) where False = masked token

    Returns:
        lookup_table_exp: Exponentiated values (vocab_size, n_langs) in fp32,
                         with masked tokens set to 0.0
    """
    logger.info(f"Computing exp() of lookup table: {lookup_table_fp32.shape}")

    # Apply exp to all values
    lookup_exp = np.exp(lookup_table_fp32.astype(np.float32))

    # Set masked values to zero (these won't contribute to the sum)
    if token_mask is not None:
        n_masked = (~token_mask).sum()
        logger.info(f"Setting {n_masked:,} masked tokens to 0.0")
        # Broadcast mask over languages
        mask_broadcasted = token_mask.reshape(-1, 1)
        lookup_exp = np.where(mask_broadcasted, lookup_exp, 0.0)

    logger.info(f"Exp lookup table: shape={lookup_exp.shape}, dtype={lookup_exp.dtype}")
    logger.info(f"  Non-zero values: {np.count_nonzero(lookup_exp):,} / {lookup_exp.size:,}")
    logger.info(f"  Min: {lookup_exp.min():.6e}, Max: {lookup_exp.max():.6e}")

    return lookup_exp


def save_lookup_table_exp(
    lookup_table_exp: np.ndarray,
    output_dir: str | Path,
    base_name: str = "lookup_table",
    threshold: float = 10.0,
    sparse: bool = True,
) -> Path:
    """Save exponentiated lookup table in sparse or dense format.

    Args:
        lookup_table_exp: Pre-exponentiated lookup table (vocab_size, n_langs) in fp32
        output_dir: Output directory
        base_name: Base filename (without extension)
        threshold: Sparsification threshold - values < threshold are set to 0 (default: 10.0)
        sparse: If True, save in sparse COO format; if False, save dense (default: True)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sparse:
        return _save_lookup_table_sparse(
            lookup_table_exp=lookup_table_exp,
            output_dir=output_dir,
            base_name=base_name,
            threshold=threshold,
        )
    else:
        return _save_lookup_table_dense(
            lookup_table_exp=lookup_table_exp,
            output_dir=output_dir,
            base_name=base_name,
        )


def _save_lookup_table_dense(
    lookup_table_exp: np.ndarray,
    output_dir: Path,
    base_name: str,
) -> Path:
    """Save exponentiated lookup table as dense fp32.

    Args:
        lookup_table_exp: Pre-exponentiated lookup table (vocab_size, n_langs) in fp32
        output_dir: Output directory
        base_name: Base filename (without extension)

    Returns:
        Path to saved file
    """
    tensors = {
        "lookup_table": lookup_table_exp.astype(np.float32),
        "dtype": np.array([32], dtype=np.uint8),  # 32 = dense exp format
        "shape": np.array(lookup_table_exp.shape, dtype=np.int64),
    }

    output_path = output_dir / f"{base_name}_exp.safetensors"
    save_file(tensors, str(output_path))

    file_size_mb = output_path.stat().st_size / (1024**2)
    logger.info(f"Saved dense exp lookup table: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def _save_lookup_table_sparse(
    lookup_table_exp: np.ndarray,
    output_dir: Path,
    base_name: str,
    threshold: float,
) -> Path:
    """Save exponentiated lookup table in sparse COO format.

    Only stores values >= threshold. During inference, values below threshold
    are treated as 0.0.

    Args:
        lookup_table_exp: Pre-exponentiated lookup table (vocab_size, n_langs) in fp32
        output_dir: Output directory
        base_name: Base filename (without extension)
        threshold: Values < threshold are set to 0 and not stored

    Returns:
        Path to saved file
    """
    logger.info(f"Applying sparsification threshold: {threshold}")

    # Find non-zero values above threshold
    mask = lookup_table_exp >= threshold
    nnz_before = np.count_nonzero(lookup_table_exp)
    nnz_after = np.sum(mask)

    sparsity = 100.0 * (1 - nnz_after / lookup_table_exp.size)
    logger.info(f"  Values >= {threshold}: {nnz_after:,} / {lookup_table_exp.size:,}")
    logger.info(f"  Sparsity: {sparsity:.2f}%")
    logger.info(f"  Removed: {nnz_before - nnz_after:,} non-zero values below threshold")

    # Get COO format: row indices, column indices, and values
    row_indices, col_indices = np.nonzero(mask)
    values = lookup_table_exp[row_indices, col_indices]

    logger.info(f"  Sparse representation: {len(values):,} values stored")

    # Save in COO format
    tensors = {
        "data": values.astype(np.float32),
        "row": row_indices.astype(np.int32),
        "col": col_indices.astype(np.int32),
        "shape": np.array(lookup_table_exp.shape, dtype=np.int64),
        "threshold": np.array([threshold], dtype=np.float32),
        "dtype": np.array([33], dtype=np.uint8),  # 33 = sparse COO format
    }

    output_path = output_dir / f"{base_name}_exp.safetensors"
    save_file(tensors, str(output_path))

    file_size_mb = output_path.stat().st_size / (1024**2)
    dense_size_mb = (lookup_table_exp.size * 4) / (1024**2)
    reduction = 100.0 * (1 - file_size_mb / dense_size_mb)

    logger.info(f"Saved sparse exp lookup table: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(
        f"  Size reduction: {dense_size_mb:.1f} MB → {file_size_mb:.1f} MB ({reduction:.1f}%)"
    )

    return output_path


def compute_lookup_table_from_model(
    model,
    model_config,
    cache_dir: str | Path = "artifacts/embeddings",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute lookup table directly from a trained model and cached embeddings.

    Returns:
        tuple: (lookup_table, token_mask) where token_mask is None if no masking
    """
    from wldetect.training.embeddings import EmbeddingsManager

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

    return lookup_table, mask_np


def save_lookup_table_exp_from_model(
    model,
    model_config,
    output_dir: str | Path,
    cache_dir: str | Path = "artifacts/embeddings",
    base_name: str = "lookup_table",
    threshold: float = 10.0,
    sparse: bool = True,
) -> Path:
    """Generate and save exponentiated lookup table from model.

    Args:
        model: Trained model
        model_config: Model configuration
        output_dir: Output directory
        cache_dir: Embeddings cache directory
        base_name: Base filename (without extension)
        threshold: Sparsification threshold (default: 10.0)
        sparse: If True, save in sparse format; if False, save dense (default: True)

    Returns:
        Path to saved file
    """
    lookup_table, token_mask = compute_lookup_table_from_model(
        model=model,
        model_config=model_config,
        cache_dir=cache_dir,
    )

    if token_mask is not None:
        n_masked = (~token_mask).sum()
        logger.info(f"Applying token mask: {n_masked:,} tokens will be set to 0.0")

    # Apply exp and mask
    lookup_table_exp = compute_lookup_table_exp(
        lookup_table_fp32=lookup_table,
        token_mask=token_mask,
    )

    return save_lookup_table_exp(
        lookup_table_exp=lookup_table_exp,
        output_dir=output_dir,
        base_name=base_name,
        threshold=threshold,
        sparse=sparse,
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
