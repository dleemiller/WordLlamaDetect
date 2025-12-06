"""Create E3M4 lookup table with filtered tokens (zero-weight under-represented tokens).

This script:
1. Loads token counts from training data
2. Identifies tokens below threshold
3. Creates lookup table with filtered token_weights (zeroed for low-count tokens)
4. Quantizes to E3M4 with scaling
5. Saves with metadata for reproducibility
"""

import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file


def create_filtered_e3m4_lookup_table(
    checkpoint_path: str | Path,
    embeddings_path: str | Path,
    token_counts_path: str | Path,
    output_path: str | Path,
    min_count_threshold: int = 10,
):
    """Create filtered E3M4 lookup table with zero-weighted under-represented tokens.

    Args:
        checkpoint_path: Path to model checkpoint with projection weights
        embeddings_path: Path to embeddings safetensors file
        token_counts_path: Path to token_counts.npy
        output_path: Path to save filtered E3M4 lookup table
        min_count_threshold: Minimum token count threshold (tokens below are zeroed)
    """
    checkpoint_path = Path(checkpoint_path)
    embeddings_path = Path(embeddings_path)
    token_counts_path = Path(token_counts_path)
    output_path = Path(output_path)

    print("=" * 80)
    print("CREATING FILTERED E3M4 LOOKUP TABLE")
    print("=" * 80)
    print()
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Embeddings:         {embeddings_path}")
    print(f"Token counts:       {token_counts_path}")
    print(f"Output:             {output_path}")
    print(f"Min count threshold: {min_count_threshold}")
    print()

    # Load token counts
    print(f"Loading token counts from {token_counts_path}...")
    token_counts = np.load(token_counts_path).flatten()
    vocab_size = len(token_counts)
    print(f"✅ Loaded token counts: {vocab_size:,} tokens")
    print(f"   Total training tokens: {token_counts.sum():,}")
    print()

    # Identify tokens to zero-weight
    zero_weight_mask = token_counts < min_count_threshold
    n_zero = zero_weight_mask.sum()
    n_keep = vocab_size - n_zero

    print("Token filtering:")
    print(f"  Tokens below threshold: {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)")
    print(f"  Tokens to keep:         {n_keep:,} ({n_keep / vocab_size * 100:.2f}%)")
    print()

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    with safe_open(embeddings_path, framework="numpy") as f:
        embeddings = f.get_tensor("embeddings")

    emb_vocab_size, hidden_dim = embeddings.shape
    print(f"✅ Loaded embeddings: {embeddings.shape}")

    if emb_vocab_size != vocab_size:
        print("⚠️  WARNING: Vocab size mismatch!")
        print(f"   Embeddings: {emb_vocab_size:,}")
        print(f"   Token counts: {vocab_size:,}")
        sys.exit(1)
    print()

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]

    # Extract model parameters
    projection_weight = state_dict["projection.weight"].numpy()  # (n_languages, hidden_dim)
    projection_bias = state_dict["projection.bias"].numpy()  # (n_languages,)
    token_weights = state_dict["token_weights"].numpy()  # (vocab_size, 1)

    n_languages = projection_weight.shape[0]

    print("✅ Loaded checkpoint:")
    print(f"   Projection weight: {projection_weight.shape}")
    print(f"   Projection bias:   {projection_bias.shape}")
    print(f"   Token weights:     {token_weights.shape}")
    print(f"   Number of languages: {n_languages}")
    print()

    # Apply filtering to token_weights
    print("Applying token filtering...")
    filtered_token_weights = token_weights.copy()
    filtered_token_weights[zero_weight_mask] = 0.0

    n_nonzero_original = (token_weights != 0).sum()
    n_nonzero_filtered = (filtered_token_weights != 0).sum()

    print(f"  Non-zero weights before: {n_nonzero_original:,}")
    print(f"  Non-zero weights after:  {n_nonzero_filtered:,}")
    print(f"  Weights zeroed:          {n_nonzero_original - n_nonzero_filtered:,}")
    print()

    # Create lookup table with filtered weights
    print("Computing lookup table...")
    print("  Formula: (embeddings * filtered_token_weights) @ projection.T + bias")

    weighted_embeddings = embeddings * filtered_token_weights
    lookup_table_fp32 = weighted_embeddings @ projection_weight.T  # (vocab_size, n_languages)
    lookup_table_fp32 = lookup_table_fp32 + projection_bias  # Add bias

    print(f"✅ Lookup table shape: {lookup_table_fp32.shape}")
    print()

    # Analyze values
    print("FP32 Statistics:")
    print(f"  Min:     {lookup_table_fp32.min():.6f}")
    print(f"  Max:     {lookup_table_fp32.max():.6f}")
    print(f"  Max abs: {np.abs(lookup_table_fp32).max():.6f}")
    print(f"  Mean:    {lookup_table_fp32.mean():.6f}")
    print(f"  Std:     {lookup_table_fp32.std():.6f}")
    print()

    # Check E3M4 range
    e3m4_max_test = (
        np.array([15.5], dtype=np.float32).astype(ml_dtypes.float8_e3m4).astype(np.float32)[0]
    )
    print(f"E3M4 max representable value: ±{e3m4_max_test:.6f}")
    print()

    max_abs_val = np.abs(lookup_table_fp32).max()

    # Determine scale factor
    if max_abs_val > e3m4_max_test:
        # Values exceed E3M4 range - use scaling
        target_max = e3m4_max_test * 0.9
        scale_factor = np.ceil(max_abs_val / target_max)

        print(f"⚠️  Max value ({max_abs_val:.2f}) exceeds E3M4 range (~±{e3m4_max_test:.1f})")
        print(f"   Using {scale_factor:.0f}x scaling")
        print(f"   Scaled range: ±{max_abs_val / scale_factor:.2f}")
        utilization = (max_abs_val / scale_factor / e3m4_max_test) * 100
        print(f"   Utilization after scaling: {utilization:.2f}%")
    else:
        # Values fit - no scaling needed
        scale_factor = 1.0
        utilization = (max_abs_val / e3m4_max_test) * 100
        print("✅ Values fit in E3M4 range")
        print(f"   Max value: {max_abs_val:.2f}")
        print(f"   E3M4 max: ~{e3m4_max_test:.1f}")
        print(f"   Utilization: {utilization:.2f}%")
    print()

    # Convert to E3M4
    if scale_factor > 1.0:
        print(f"Converting to E3M4 with {scale_factor:.0f}x scaling...")
        scaled_values = lookup_table_fp32 / scale_factor
        lookup_table_e3m4_bytes = scaled_values.astype(ml_dtypes.float8_e3m4)
    else:
        print("Converting to E3M4...")
        lookup_table_e3m4_bytes = lookup_table_fp32.astype(ml_dtypes.float8_e3m4)

    lookup_table_e3m4_uint8 = lookup_table_e3m4_bytes.view(np.uint8)

    # Test dequantization
    print("Testing dequantization...")
    dequantized = lookup_table_e3m4_uint8.view(ml_dtypes.float8_e3m4).astype(np.float32)
    if scale_factor > 1.0:
        dequantized = dequantized * scale_factor

    abs_error = np.abs(lookup_table_fp32 - dequantized)
    rel_error = abs_error / (np.abs(lookup_table_fp32) + 1e-10)

    print("Quantization Error Metrics:")
    print(f"  Mean absolute error: {abs_error.mean():.8f}")
    print(f"  Max absolute error:  {abs_error.max():.8f}")
    print(f"  Mean relative error: {rel_error.mean():.8f}")
    print(f"  RMS error:           {np.sqrt(np.mean(abs_error**2)):.8f}")
    print()

    # Save to safetensors with metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...")

    tensors = {
        "lookup_table": lookup_table_e3m4_uint8.reshape(lookup_table_fp32.shape),
        "dtype": np.array([26], dtype=np.uint8),  # E3M4 dtype ID
        "shape": np.array(lookup_table_fp32.shape, dtype=np.int64),
        "scale": np.array([scale_factor], dtype=np.float32),
        "zero_weight_mask": zero_weight_mask.astype(np.uint8),  # Save mask
        "min_count_threshold": np.array([min_count_threshold], dtype=np.int32),
    }

    save_file(tensors, str(output_path))

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Saved: {file_size_mb:.2f} MB")
    print()

    print("=" * 80)
    print("✅ FILTERED E3M4 CREATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Lookup table: (vocab_size={vocab_size:,}, n_languages={n_languages})")
    print("Format: float8_e3m4")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Scale factor: {scale_factor:.0f}x")
    print(f"Tokens filtered: {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)")
    print(f"Mean quantization error: {abs_error.mean():.8f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create filtered E3M4 lookup table")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="artifacts/embeddings/embeddings_a26b8f6b3226_150langs.safetensors",
        help="Path to embeddings safetensors",
    )
    parser.add_argument(
        "--token-counts",
        type=str,
        default="artifacts/token_counts.npy",
        help="Path to token counts array",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/gemma3-27b/lookup_table_fp8_e3m4_filtered.safetensors",
        help="Path to save filtered E3M4 lookup table",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum token count threshold",
    )

    args = parser.parse_args()

    create_filtered_e3m4_lookup_table(
        args.checkpoint,
        args.embeddings,
        args.token_counts,
        args.output,
        args.threshold,
    )
