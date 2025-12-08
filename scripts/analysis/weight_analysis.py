"""Comprehensive analysis of token weights vs token frequency.

This script analyzes the relationship between:
- Token frequencies (how often tokens appear in training data)
- Token weights (learned weights from the model)

It helps identify tokens that should be zero-weighted due to insufficient representation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_token_weights_vs_frequency(
    checkpoint_path: str | Path,
    token_counts_path: str | Path,
    output_dir: str | Path,
    min_count_threshold: int = 10,
):
    """Analyze token weights against token frequencies.

    Args:
        checkpoint_path: Path to model checkpoint with token_weights
        token_counts_path: Path to token_counts.npy (n_vocab x 1 or n_vocab,)
        output_dir: Directory to save analysis plots
        min_count_threshold: Suggested minimum count threshold for zeroing
    """
    checkpoint_path = Path(checkpoint_path)
    token_counts_path = Path(token_counts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TOKEN WEIGHT VS FREQUENCY ANALYSIS")
    print("=" * 80)
    print()

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    token_weights = state_dict["token_weights"].numpy().flatten()

    # Load token counts
    print(f"Loading token counts from {token_counts_path}...")
    token_counts = np.load(token_counts_path).flatten()

    vocab_size = len(token_weights)
    print(f"Vocabulary size: {vocab_size:,}")
    print()

    # Verify shapes match
    if len(token_counts) != vocab_size:
        raise ValueError(
            f"Shape mismatch: token_weights={vocab_size}, token_counts={len(token_counts)}"
        )

    # Statistics
    print("Token Count Statistics:")
    print(f"  Total tokens in training: {token_counts.sum():,}")
    print(
        f"  Tokens with 0 counts:     {(token_counts == 0).sum():,} ({(token_counts == 0).sum() / vocab_size * 100:.2f}%)"
    )
    print(
        f"  Tokens with < 10 counts:  {(token_counts < 10).sum():,} ({(token_counts < 10).sum() / vocab_size * 100:.2f}%)"
    )
    print(
        f"  Tokens with < 100 counts: {(token_counts < 100).sum():,} ({(token_counts < 100).sum() / vocab_size * 100:.2f}%)"
    )
    print(f"  Median count:             {np.median(token_counts):.1f}")
    print(f"  Mean count:               {token_counts.mean():.1f}")
    print()

    print("Token Weight Statistics:")
    print(f"  Mean:   {token_weights.mean():.6f}")
    print(f"  Median: {np.median(token_weights):.6f}")
    print(f"  Std:    {token_weights.std():.6f}")
    print(f"  Min:    {token_weights.min():.6f}")
    print(f"  Max:    {token_weights.max():.6f}")
    print()

    # Identify tokens to zero-weight
    zero_weight_mask = token_counts < min_count_threshold
    n_zero = zero_weight_mask.sum()

    print(f"Tokens below threshold ({min_count_threshold} counts):")
    print(f"  Count: {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)")
    print(f"  Combined token weight: {token_weights[zero_weight_mask].sum():.6f}")
    print(f"  Average weight: {token_weights[zero_weight_mask].mean():.6f}")
    print()

    # Plot 1: Scatter plot (log-log scale)
    ax1 = plt.subplot(2, 3, 1)
    # Use log1p to handle zero counts
    scatter = ax1.scatter(
        np.log1p(token_counts),
        token_weights,
        alpha=0.3,
        s=10,
        c=zero_weight_mask,
        cmap="coolwarm",
    )
    ax1.axvline(
        x=np.log1p(min_count_threshold),
        color="r",
        linestyle="--",
        label=f"Threshold ({min_count_threshold})",
    )
    ax1.set_xlabel("log(Token Count + 1)", fontsize=11)
    ax1.set_ylabel("Token Weight", fontsize=11)
    ax1.set_title("Token Weight vs Frequency (Log Scale)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label="Below Threshold")

    # Plot 2: Histogram of counts (log scale)
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(np.log10(token_counts + 1), bins=100, edgecolor="black", alpha=0.7)
    ax2.axvline(
        x=np.log10(min_count_threshold + 1),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({min_count_threshold})",
    )
    ax2.set_xlabel("log10(Token Count + 1)", fontsize=11)
    ax2.set_ylabel("Number of Tokens", fontsize=11)
    ax2.set_title("Distribution of Token Counts", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Histogram of weights
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(token_weights, bins=100, edgecolor="black", alpha=0.7, color="green")
    ax3.set_xlabel("Token Weight", fontsize=11)
    ax3.set_ylabel("Number of Tokens", fontsize=11)
    ax3.set_title("Distribution of Token Weights", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Coverage curve (Pareto)
    ax4 = plt.subplot(2, 3, 4)
    sorted_indices = np.argsort(token_counts)[::-1]  # Descending
    cumsum = np.cumsum(token_counts[sorted_indices])
    coverage = cumsum / cumsum[-1]
    ax4.plot(coverage, linewidth=2)
    ax4.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95% coverage")
    ax4.axhline(y=0.99, color="orange", linestyle="--", alpha=0.7, label="99% coverage")
    # Find tokens covering 95%
    tokens_for_95 = np.searchsorted(coverage, 0.95)
    ax4.axvline(x=tokens_for_95, color="r", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Token Rank (by frequency)", fontsize=11)
    ax4.set_ylabel("Cumulative Token Coverage", fontsize=11)
    ax4.set_title(
        f"Token Coverage Curve (95% at rank {tokens_for_95:,})", fontsize=12, fontweight="bold"
    )
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Weight distribution by count bins
    ax5 = plt.subplot(2, 3, 5)
    bins = [0, 1, 10, 100, 1000, 10000, np.inf]
    bin_labels = ["0", "1-9", "10-99", "100-999", "1K-10K", "10K+"]
    bin_indices = np.digitize(token_counts, bins)

    bin_means = []
    bin_stds = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.any():
            bin_means.append(token_weights[mask].mean())
            bin_stds.append(token_weights[mask].std())
        else:
            bin_means.append(0)
            bin_stds.append(0)

    ax5.bar(
        range(len(bin_labels)), bin_means, yerr=bin_stds, capsize=5, alpha=0.7, edgecolor="black"
    )
    ax5.set_xticks(range(len(bin_labels)))
    ax5.set_xticklabels(bin_labels)
    ax5.set_xlabel("Token Count Bin", fontsize=11)
    ax5.set_ylabel("Mean Token Weight", fontsize=11)
    ax5.set_title("Average Weight by Frequency Bin", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="y")

    # Plot 6: Outlier analysis
    ax6 = plt.subplot(2, 3, 6)
    # Identify high-weight but low-frequency tokens
    high_weight_threshold = np.percentile(token_weights, 75)
    low_freq_mask = token_counts < min_count_threshold
    high_weight_mask = token_weights > high_weight_threshold
    outliers = low_freq_mask & high_weight_mask

    ax6.scatter(
        np.log1p(token_counts[~outliers]),
        token_weights[~outliers],
        alpha=0.2,
        s=10,
        label="Normal",
        color="blue",
    )
    ax6.scatter(
        np.log1p(token_counts[outliers]),
        token_weights[outliers],
        alpha=0.7,
        s=30,
        label=f"Outliers ({outliers.sum()})",
        color="red",
        marker="x",
    )
    ax6.set_xlabel("log(Token Count + 1)", fontsize=11)
    ax6.set_ylabel("Token Weight", fontsize=11)
    ax6.set_title("Outlier Detection (High Weight + Low Frequency)", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "token_weight_frequency_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved analysis plot to {output_path}")
    print()

    # Save threshold recommendations
    recommendations_path = output_dir / "token_filtering_recommendations.txt"
    with open(recommendations_path, "w") as f:
        f.write("TOKEN FILTERING RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n")
        f.write(f"Total training tokens: {token_counts.sum():,}\n\n")

        f.write("Suggested Thresholds:\n")
        for threshold in [1, 5, 10, 50, 100]:
            n_below = (token_counts < threshold).sum()
            pct = n_below / vocab_size * 100
            f.write(f"  {threshold:3d} counts: {n_below:6,} tokens ({pct:5.2f}%)\n")

        f.write(f"\nRecommended threshold: {min_count_threshold} counts\n")
        f.write(f"  Tokens to zero-weight: {n_zero:,} ({n_zero / vocab_size * 100:.2f}%)\n")
        f.write(
            f"  Tokens remaining:      {vocab_size - n_zero:,} ({(1 - n_zero / vocab_size) * 100:.2f}%)\n"
        )

        f.write(f"\nOutliers (high weight + low frequency): {outliers.sum()}\n")
        f.write("  These tokens have high learned weights but appear rarely.\n")
        f.write("  Consider manual review before zero-weighting.\n")

    print(f"Saved recommendations to {recommendations_path}")
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze token weights vs frequency")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/gemma3-27b/checkpoints/checkpoint_epoch_1.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--token-counts",
        type=str,
        default="artifacts/token_counts.npy",
        help="Path to token counts array",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/",
        help="Directory to save analysis outputs",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Minimum count threshold for zeroing tokens",
    )

    args = parser.parse_args()

    analyze_token_weights_vs_frequency(
        args.checkpoint,
        args.token_counts,
        args.output_dir,
        args.threshold,
    )
