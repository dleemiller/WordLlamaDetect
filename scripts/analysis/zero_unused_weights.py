#!/usr/bin/env python3
"""Zero out token weights for tokens with 0 training occurrences.

This script:
1. Loads a checkpoint with token_weights
2. Loads token_counts.npy
3. Identifies tokens with count=0
4. Sets their weights to 0
5. Saves a new checkpoint with zeroed weights
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = ArgumentParser(description="Zero out weights for unused tokens")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint with token_weights",
    )
    parser.add_argument(
        "--token-counts",
        type=str,
        required=True,
        help="Path to token_counts.npy",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save modified checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ZEROING WEIGHTS FOR UNUSED TOKENS")
    print("=" * 80)
    print()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    token_weights = state_dict["token_weights"]  # (vocab_size, 1)

    print(f"  Token weights shape: {token_weights.shape}")
    print(f"  Token weights dtype: {token_weights.dtype}")
    print()

    # Load token counts
    print(f"Loading token counts from {args.token_counts}...")
    token_counts = np.load(args.token_counts)  # (vocab_size, 1)
    print(f"  Token counts shape: {token_counts.shape}")
    print(f"  Total tokens in training: {token_counts.sum():,}")
    print()

    # Check shapes match
    if token_weights.shape[0] != token_counts.shape[0]:
        raise ValueError(
            f"Shape mismatch: token_weights={token_weights.shape[0]}, "
            f"token_counts={token_counts.shape[0]}"
        )

    # Find tokens with 0 counts
    zero_count_mask = (token_counts == 0).flatten()
    n_zero = zero_count_mask.sum()
    vocab_size = len(zero_count_mask)

    print(f"Tokens with 0 counts: {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)")
    print()

    # Get statistics before zeroing
    weights_numpy = token_weights.numpy()
    zero_count_weights = weights_numpy[zero_count_mask]

    print("Weights for tokens with 0 counts:")
    print(f"  Mean:   {zero_count_weights.mean():.6f}")
    print(f"  Median: {np.median(zero_count_weights):.6f}")
    print(f"  Std:    {zero_count_weights.std():.6f}")
    print(f"  Min:    {zero_count_weights.min():.6f}")
    print(f"  Max:    {zero_count_weights.max():.6f}")
    print()

    # Zero out weights for tokens with 0 counts
    print("Zeroing weights for unused tokens...")
    token_weights_modified = token_weights.clone()
    token_weights_modified[zero_count_mask] = 0.0

    # Update checkpoint
    state_dict["token_weights"] = token_weights_modified
    checkpoint["model_state_dict"] = state_dict

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving modified checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)

    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Saved: {file_size_mb:.2f} MB")
    print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tokens with 0 counts:   {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)")
    print(f"Weights zeroed:         {n_zero:,}")
    print(
        f"Non-zero weights:       {(vocab_size - n_zero):,} ({(1 - n_zero / vocab_size) * 100:.2f}%)"
    )
    print()
    print(f"Modified checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
