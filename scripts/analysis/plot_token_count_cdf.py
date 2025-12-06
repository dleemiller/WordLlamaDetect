#!/usr/bin/env python3
"""Plot the token-count CDF from a saved count tensor (n_vocab x 1)."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def parse_args() -> tuple[str, str]:
    parser = ArgumentParser(description="Plot cumulative distribution of token counts.")
    parser.add_argument(
        "--counts",
        default="artifacts/token_counts.npy",
        help="Path to the numpy file with counts shaped (n_vocab, 1).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/token_counts_cdf.png",
        help="Where to save the CDF plot.",
    )
    args = parser.parse_args()
    return args.counts, args.output


def main() -> None:
    counts_path, output_path = parse_args()

    counts = np.load(counts_path).squeeze()
    if counts.ndim != 1:
        raise ValueError(f"Expected 1D counts after squeeze, got shape {counts.shape}")

    # Sort ascending so the lowest-count tokens start at x=0
    sorted_counts = np.sort(counts)
    cumulative_tokens = np.cumsum(sorted_counts)
    total_tokens = cumulative_tokens[-1]
    x = np.arange(len(sorted_counts))

    y_for_plot = np.maximum(cumulative_tokens, 1)  # Avoid log(0) when counts are zero

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_for_plot, linewidth=1.5)
    plt.xlabel("Token rank (ascending by count)")
    plt.ylabel("Cumulative tokens")
    plt.title("Token Count CDF (low-frequency tokens at origin)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved CDF plot to {output_path} (total_tokens={int(total_tokens)})")


if __name__ == "__main__":
    main()
