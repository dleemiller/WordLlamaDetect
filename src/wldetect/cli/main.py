"""CLI entry point for WLDetect."""

import argparse
import sys

from wldetect.cli.commands import create_lookup, curate, detect, eval, train


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WLDetect - Fast, accurate language detection using static LLM embeddings"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a language detection model",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a model on FLORES (PyTorch checkpoint or fp8 inference model)",
    )
    mode_group = eval_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file (PyTorch mode)",
    )
    mode_group.add_argument(
        "--model-path",
        type=str,
        help="Path to model directory with fp8 lookup table (inference mode)",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path to load (PyTorch mode only, default: best_model.pt in checkpoint_dir)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        help="Output path for evaluation metrics JSON (default: flores_{split}_results.json)",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "devtest"],
        help="FLORES split to evaluate on (default: dev)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (default: 512 for fp8, training config for PyTorch)",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on (PyTorch mode only, default: auto)",
    )
    eval_parser.add_argument(
        "--embedding-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to load embeddings with (PyTorch mode only, default: float32)",
    )

    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect language of text",
    )
    detect_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    group = detect_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--text",
        type=str,
        help="Text to detect language of",
    )
    group.add_argument(
        "--file",
        type=str,
        help="File containing text to detect language of",
    )

    # Create-lookup command
    create_lookup_parser = subparsers.add_parser(
        "create-lookup",
        help="Generate lookup table from a checkpoint",
    )
    create_lookup_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoint_step_100000.pt)",
    )
    create_lookup_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    create_lookup_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for lookup table",
    )

    # Curate command
    curate_parser = subparsers.add_parser(
        "curate",
        help="Evaluate model, threshold languages, and write a pruned model config",
    )
    curate_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    curate_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path to load (default: best_model.pt in checkpoint_dir)",
    )
    curate_parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.3,
        help="Minimum accuracy to keep a language (default: 0.3)",
    )
    curate_parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.0,
        help="Minimum F1 to keep a language (default: 0.0)",
    )
    curate_parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum evaluation samples to consider a language (default: 1)",
    )
    curate_parser.add_argument(
        "--output-config",
        type=str,
        required=True,
        help="Output path for pruned model configuration YAML",
    )
    curate_parser.add_argument(
        "--output-metrics",
        type=str,
        help="Optional path to save evaluation metrics JSON",
    )
    curate_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (default: training batch size)",
    )
    curate_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on (default: auto)",
    )
    curate_parser.add_argument(
        "--embedding-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to load embeddings with (default: float32)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command
    command_map = {
        "train": train.run,
        "eval": eval.run,
        "detect": detect.run,
        "create-lookup": create_lookup.run,
        "curate": curate.run,
    }

    return command_map[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
