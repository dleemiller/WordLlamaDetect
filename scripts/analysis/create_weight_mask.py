#!/usr/bin/env python3
"""Create a boolean weight mask from token counts.

This script creates a mask to zero-weight tokens with insufficient training representation.
The mask is applied during training initialization.

Usage:
    uv run python scripts/analysis/create_weight_mask.py \
        --token-counts artifacts/token_counts.npy \
        --threshold 10 \
        --output artifacts/token_mask.npy
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse_args():
    parser = ArgumentParser(description="Create weight mask from token counts")
    parser.add_argument(
        "--token-counts",
        type=str,
        required=True,
        help="Path to token_counts.npy (n_vocab x 1 or n_vocab,)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum token count threshold (tokens below this are masked)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save mask (n_vocab,) boolean array",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("CREATING WEIGHT MASK FROM TOKEN COUNTS")
    print("=" * 80)
    print()

    # Load token counts
    print(f"Loading token counts from {args.token_counts}...")
    token_counts = np.load(args.token_counts)  # (vocab_size, 1) or (vocab_size,)

    # Flatten if needed
    if token_counts.ndim == 2:
        token_counts = token_counts.flatten()

    vocab_size = len(token_counts)
    total_tokens = int(token_counts.sum())

    print(f"  Vocab size:    {vocab_size:,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print()

    # Create mask: True for tokens to KEEP, False for tokens to ZERO
    mask = token_counts >= args.threshold

    n_kept = mask.sum()
    n_masked = (~mask).sum()

    print(f"Threshold: {args.threshold} occurrences")
    print(
        f"  Tokens kept (count >= {args.threshold}):  {n_kept:,} ({n_kept / vocab_size * 100:.2f}%)"
    )
    print(
        f"  Tokens masked (count < {args.threshold}): {n_masked:,} ({n_masked / vocab_size * 100:.2f}%)"
    )
    print()

    # Statistics on masked tokens
    masked_counts = token_counts[~mask]
    if len(masked_counts) > 0:
        print("Masked token statistics:")
        print(f"  Total occurrences: {masked_counts.sum():,}")
        print(f"  % of training data: {masked_counts.sum() / total_tokens * 100:.4f}%")
        print(f"  Mean count: {masked_counts.mean():.2f}")
        print(f"  Max count:  {masked_counts.max()}")
        print()

    # Statistics on kept tokens
    kept_counts = token_counts[mask]
    if len(kept_counts) > 0:
        print("Kept token statistics:")
        print(f"  Total occurrences: {kept_counts.sum():,}")
        print(f"  % of training data: {kept_counts.sum() / total_tokens * 100:.4f}%")
        print(f"  Mean count: {kept_counts.mean():.1f}")
        print(f"  Min count:  {kept_counts.min()}")
        print()

    # Save mask
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving mask to {output_path}...")
    np.save(output_path, mask)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {file_size_kb:.1f} KB")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Mask shape:     {mask.shape}")
    print(f"Mask dtype:     {mask.dtype}")
    print(f"Tokens kept:    {n_kept:,} ({n_kept / vocab_size * 100:.2f}%)")
    print(f"Tokens masked:  {n_masked:,} ({n_masked / vocab_size * 100:.2f}%)")
    print()
    print("Apply this mask during training with:")
    print(f"  uv run wldetect train --config <config>.yaml --token-mask {output_path}")


if __name__ == "__main__":
    main()
