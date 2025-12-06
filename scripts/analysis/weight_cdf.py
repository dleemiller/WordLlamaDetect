"""Visualize cumulative distribution function of token weights.

This script analyzes the learned token weights from a trained model checkpoint
and generates a CDF plot to understand the distribution of weights across the vocabulary.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_token_weights_cdf(checkpoint_path: str | Path, output_path: str | Path):
    """Generate CDF plot of token weights from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        output_path: Path to save the CDF plot (.png file)
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract token weights
    state_dict = checkpoint["model_state_dict"]
    token_weights = state_dict["token_weights"].numpy()

    print(f"Loaded token weights: {token_weights.shape}")
    print()

    # Compute statistics
    weights_flat = token_weights.flatten()
    print("Token Weight Statistics:")
    print(f"  Mean:   {weights_flat.mean():.6f}")
    print(f"  Median: {np.median(weights_flat):.6f}")
    print(f"  Std:    {weights_flat.std():.6f}")
    print(f"  Min:    {weights_flat.min():.6f}")
    print(f"  Max:    {weights_flat.max():.6f}")
    print()

    # Compute CDF
    sorted_weights = np.sort(weights_flat)
    cdf = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full CDF
    ax1.plot(sorted_weights, cdf, linewidth=2)
    ax1.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Median")
    ax1.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95th percentile")
    ax1.set_xlabel("Token Weight", fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.set_title("CDF of Token Weights (Full Range)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed CDF (exclude extremes)
    p5, p95 = np.percentile(weights_flat, [5, 95])
    mask = (sorted_weights >= p5) & (sorted_weights <= p95)
    ax2.plot(sorted_weights[mask], cdf[mask], linewidth=2, color="darkblue")
    ax2.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Median")
    ax2.set_xlabel("Token Weight", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_title("CDF of Token Weights (5th-95th Percentile)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved CDF plot to {output_path}")

    # Print percentiles
    print()
    print("Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(weights_flat, p)
        print(f"  {p:2d}th: {val:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CDF plot of token weights")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/token_weights_cdf.png",
        help="Path to save output plot",
    )

    args = parser.parse_args()

    plot_token_weights_cdf(args.checkpoint, args.output)
