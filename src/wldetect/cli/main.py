"""CLI entry point for WLDetect."""

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("wldetect")


def train_command(args):
    """Execute training command."""
    logger = setup_logging()
    try:
        import torch
        from torch.utils.data import DataLoader

        from wldetect.config.loader import load_model_config, load_training_config
        from wldetect.data.dataset import prepare_dataset
        from wldetect.data.tokenizer import load_tokenizer
        from wldetect.embeddings.extractor import extract_embeddings
        from wldetect.training.evaluator import save_lookup_table, save_projection_matrix
        from wldetect.training.model import LanguageDetectionModel
        from wldetect.training.trainer import (
            LanguageDetectionDataset,
            Trainer,
            collate_fn,
        )
    except ImportError as e:
        logger.error("Error: Training dependencies not installed. Run: uv sync --extra training")
        logger.error(f"Details: {e}")
        return 1

    logger.info("Loading configuration...")
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)

    logger.info("\n" + "=" * 60)
    logger.info("LANGTOKEN TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model: {model_config.all_models[0].name}")
    logger.info(f"Languages: {len(model_config.languages)}")
    logger.info(f"Hidden dim: {model_config.hidden_dim}")
    logger.info("=" * 60 + "\n")

    # Extract embeddings
    logger.info("Step 1: Extract embeddings")
    from wldetect.embeddings.extractor import get_cache_path, load_embeddings_as_memmap

    cache_path = get_cache_path(model_config)
    if not cache_path.exists():
        # Extract if not cached
        embeddings = extract_embeddings(model_config)
        # Now load as memmap for fast reading
        embeddings = load_embeddings_as_memmap(cache_path)
    else:
        # Load directly as memmap (fast disk read)
        logger.info(f"Loading embeddings from {cache_path}")
        embeddings = load_embeddings_as_memmap(cache_path)

    logger.info(f"  Embeddings: {embeddings.shape}, dtype={embeddings.dtype}")
    logger.info(f"  Memory: {embeddings.nbytes / 1024**3:.2f} GB (will be loaded to GPU)")

    # Load tokenizer
    logger.info("\nStep 2: Load tokenizer")
    tokenizer = load_tokenizer(model_config)

    # Prepare dataset
    logger.info("\nStep 3: Prepare dataset")
    language_codes = list(model_config.languages.keys())
    dataset_dict = prepare_dataset(config.dataset, language_codes)

    # Create PyTorch datasets with lazy tokenization
    logger.info("\nStep 4: Create training data loader")
    logger.info(f"  Train split: {len(dataset_dict['train'])} examples")

    train_dataset = LanguageDetectionDataset(
        dataset_dict["train"],
        tokenizer,
        model_config.languages,
        max_length=model_config.inference.max_sequence_length,
        logger=logger,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=4
        if config.training.num_workers > 0
        else None,  # Prefetch 4 batches per worker
        persistent_workers=True if config.training.num_workers > 0 else False,  # Keep workers alive
    )

    # Create model
    logger.info("\nStep 5: Initialize model")
    vocab_size = embeddings.shape[0]  # First dimension of embeddings

    # Convert embeddings to torch tensor (will move to GPU with model)
    logger.info("  Converting embeddings to torch tensor...")
    embeddings_tensor = torch.from_numpy(embeddings).float()
    logger.info(
        f"  Embeddings tensor: {embeddings_tensor.shape}, "
        f"{embeddings_tensor.element_size() * embeddings_tensor.nelement() / 1024**3:.2f} GB"
    )

    model = LanguageDetectionModel(
        hidden_dim=model_config.hidden_dim,
        n_languages=model_config.n_languages,
        vocab_size=vocab_size,
        embeddings=embeddings_tensor,
        dropout=config.training.projection.dropout,
        pooling=model_config.inference.pooling,
    )
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    logger.info("\nStep 6: Train model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Moving model to {device}...")
    trainer = Trainer(
        model,
        config,
        device=device,
        tokenizer=tokenizer,
        model_config=model_config,
        logger=logger,
    )
    logger.info(f"  Embeddings are now on {device} (fast GPU indexing!)")
    trainer.train(train_loader, val_loader=None)  # No validation during training

    # Final evaluation on FLORES-200
    logger.info("\nStep 7: Final evaluation on FLORES")
    from wldetect.training.flores_eval import evaluate_on_flores, save_flores_evaluation

    flores_results = evaluate_on_flores(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        split=config.evaluation.flores_split,
        batch_size=config.evaluation.flores_batch_size or config.training.batch_size,
        num_workers=config.training.num_workers,
        device=device,
        hf_dataset=config.evaluation.flores_hf_dataset,
    )

    # Save FLORES-200 evaluation metrics
    flores_metrics_path = (
        Path(config.output.artifacts_dir) / f"flores_{config.evaluation.flores_split}_results.json"
    )
    save_flores_evaluation(flores_results, flores_metrics_path)

    # Log FLORES-200 results to TensorBoard
    trainer.writer.add_scalar("flores/accuracy", flores_results["overall"]["accuracy"], 0)
    trainer.writer.add_scalar("flores/f1_macro", flores_results["overall"]["f1_macro"], 0)
    trainer.writer.add_scalar("flores/f1_weighted", flores_results["overall"]["f1_weighted"], 0)

    # Log top 10 and bottom 10 language accuracies as histograms
    per_lang_metrics = flores_results["per_language"]
    lang_accuracies = {lang: metrics["accuracy"] for lang, metrics in per_lang_metrics.items()}
    sorted_langs = sorted(lang_accuracies.items(), key=lambda x: x[1], reverse=True)

    # Add per-language accuracies as text
    lang_accuracy_text = "Top 10 Languages:\n"
    for lang, acc in sorted_langs[:10]:
        lang_accuracy_text += f"  {lang}: {acc:.4f}\n"
    lang_accuracy_text += "\nBottom 10 Languages:\n"
    for lang, acc in sorted_langs[-10:]:
        lang_accuracy_text += f"  {lang}: {acc:.4f}\n"
    trainer.writer.add_text("flores/language_accuracies", lang_accuracy_text, 0)

    # Save artifacts
    logger.info("\nStep 8: Save artifacts")
    trainer.save_final_model()

    # Save projection matrix
    projection_path = Path(config.output.artifacts_dir) / config.output.projection_matrix_name
    save_projection_matrix(model, str(projection_path))

    # Generate and save fp8_e4m3fn lookup table
    logger.info("\nStep 8b: Generate fp8_e4m3fn lookup table")
    lookup_table_path = save_lookup_table(
        model=model,
        model_config=model_config,
        output_dir=config.output.artifacts_dir,
    )

    # Log saved file
    size_mb = lookup_table_path.stat().st_size / (1024**2)
    logger.info(f"Lookup table saved: {lookup_table_path.name} ({size_mb:.1f} MB)")

    # Save model config
    from wldetect.config.loader import save_model_config

    config_path = Path(config.output.artifacts_dir) / config.output.config_name
    save_model_config(model_config, config_path)

    # Close TensorBoard writer
    trainer.close()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Artifacts saved to: {config.output.artifacts_dir}")
    logger.info(f"  - Projection matrix: {projection_path}")
    logger.info(f"  - Model config: {config_path}")
    logger.info("=" * 60 + "\n")

    return 0


def eval_command(args):
    """Execute evaluation command."""
    logger = setup_logging()
    import torch
    from transformers import AutoTokenizer

    from wldetect.config.loader import load_model_config, load_training_config
    from wldetect.embeddings.extractor import (
        extract_embeddings,
        get_cache_path,
        load_embeddings_as_memmap,
    )
    from wldetect.training.flores_eval import evaluate_on_flores, save_flores_evaluation
    from wldetect.training.model import LanguageDetectionModel

    logger.info("=" * 60)
    logger.info("LANGTOKEN EVALUATION")
    logger.info("=" * 60)

    # Load configs
    logger.info("\nStep 1: Load configurations")
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)
    logger.info(f"Model: {model_config.model.name if model_config.model else 'Multi-model'}")
    logger.info(f"Languages: {model_config.n_languages}")

    # Load embeddings (use memmap for multi-worker efficiency)
    logger.info("\nStep 2: Load embeddings")
    cache_path = get_cache_path(model_config, cache_dir="artifacts/embeddings")
    if not cache_path.exists():
        logger.info(f"Embeddings not found at {cache_path}, extracting...")
        extract_embeddings(
            model_config,
            cache_dir="artifacts/embeddings",
            use_cache=True,
        )
    logger.info(f"Loading embeddings as memory-mapped array from {cache_path}")
    embeddings = load_embeddings_as_memmap(cache_path)
    logger.info(f"  Memory-mapped embeddings: {embeddings.shape}, dtype={embeddings.dtype}")

    # Load tokenizer
    logger.info("\nStep 3: Load tokenizer")
    first_model = model_config.model if model_config.model else model_config.models[0]
    tokenizer = AutoTokenizer.from_pretrained(first_model.name)

    # Load model
    logger.info("\nStep 4: Load trained model")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error("Error: CUDA requested but not available")
            return 1
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = embeddings.shape[0]  # First dimension of embeddings

    # Convert embeddings to torch tensor for GPU/CPU
    logger.info(f"  Loading embeddings to {device}...")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(getattr(args, "embedding_dtype", "float32"), torch.float32)
    embeddings_tensor = torch.from_numpy(embeddings).clone().to(device=device, dtype=target_dtype)

    model = LanguageDetectionModel(
        hidden_dim=model_config.hidden_dim,
        n_languages=model_config.n_languages,
        vocab_size=vocab_size,
        embeddings=embeddings_tensor,
        dropout=config.training.projection.dropout,
        pooling=model_config.inference.pooling,
    )

    # Load checkpoint
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config.output.checkpoint_dir) / "best_model.pt"
    )
    if not checkpoint_path.exists():
        logger.error(f"Error: Model checkpoint not found at {checkpoint_path}")
        return 1

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        logger.warning(
            f"  Warning: weights_only load failed ({exc}); retrying without weights_only"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    # Evaluate on FLORES
    split = config.evaluation.flores_split
    batch_size = (
        args.batch_size or config.evaluation.flores_batch_size or config.training.batch_size
    )
    logger.info(f"\nStep 5: Evaluate on FLORES '{split}' split")
    results = evaluate_on_flores(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        split=split,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        device=device,
        hf_dataset=config.evaluation.flores_hf_dataset,
        cache_dir=config.evaluation.flores_cache_dir,
    )

    # Save metrics to JSON
    output_path = args.output or (
        Path(config.output.artifacts_dir) / f"flores_{split}_results.json"
    )
    save_flores_evaluation(results, output_path)
    # Save confusion heatmap
    from wldetect.training.flores_eval import save_confusion_heatmap

    heatmap_path = output_path.with_suffix(".png")
    save_confusion_heatmap(results, list(model_config.languages.keys()), heatmap_path)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Metrics saved to: {output_path}")
    logger.info("=" * 60 + "\n")

    return 0


def detect_command(args):
    """Execute detection command."""
    from wldetect import WLDetect

    logger = setup_logging()

    logger.info(f"Loading model from {args.model_path}...")
    detector = WLDetect.load(args.model_path)

    if args.text:
        # Detect single text
        top_lang, confidence = detector.predict(args.text)

        logger.info(f"\nText: {args.text}")
        logger.info(f"Detected language: {top_lang} (confidence: {confidence:.2%})")

    elif args.file:
        # Detect from file
        with open(args.file) as f:
            text = f.read()

        top_lang, confidence = detector.predict(text)

        logger.info(f"\nFile: {args.file}")
        logger.info(f"Detected language: {top_lang} (confidence: {confidence:.2%})")

    else:
        logger.error("Error: Either --text or --file must be provided")
        return 1

    return 0


def create_lookup_command(args):
    """Generate lookup table from a checkpoint."""
    logger = setup_logging()

    try:
        import torch

        from wldetect.config.loader import load_model_config, load_training_config
        from wldetect.embeddings.extractor import get_cache_path, load_embeddings
        from wldetect.training.lookup_table import compute_lookup_table, save_lookup_table
    except ImportError as e:
        logger.error("Error: Training dependencies not installed. Run: uv sync --extra training")
        logger.error(f"Details: {e}")
        return 1

    logger.info("=" * 60)
    logger.info("LOOKUP TABLE GENERATION")
    logger.info("=" * 60)

    # Load configs
    logger.info(f"\nLoading training config from {args.config}")
    training_config = load_training_config(args.config)

    logger.info(f"Loading model config from {training_config.model_config_path}")
    model_config = load_model_config(training_config.model_config_path)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    logger.info(f"\nLoading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning(f"Warning: weights_only load failed ({exc}); retrying without weights_only")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    logger.info(f"Checkpoint info: step={checkpoint.get('step')}, epoch={checkpoint.get('epoch')}")

    # Extract projection weights from state dict
    state_dict = checkpoint["model_state_dict"]
    projection_weight = state_dict["projection.weight"].cpu().numpy()  # (n_langs, hidden_dim)
    projection_bias = state_dict["projection.bias"].cpu().numpy()  # (n_langs,)
    token_weights = state_dict["token_weights"].cpu().numpy()  # (vocab_size, 1)

    logger.info(
        f"Extracted weights: projection={projection_weight.shape}, bias={projection_bias.shape}, token_weights={token_weights.shape}"
    )

    # Load embeddings
    logger.info("\nLoading embeddings from cache")
    cache_path = get_cache_path(model_config, cache_dir="artifacts/embeddings")
    if not cache_path.exists():
        logger.error(f"Error: Embeddings cache not found at {cache_path}")
        logger.error("Run training first to generate embeddings cache")
        return 1

    embeddings = load_embeddings(cache_path)
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Compute lookup table
    logger.info("\nComputing lookup table")
    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Save as fp8_e4m3fn
    output_dir = Path(args.output_dir)
    logger.info(f"\nSaving fp8_e4m3fn lookup table to {output_dir}")
    saved_path = save_lookup_table(
        lookup_table_fp32=lookup_table,
        output_dir=output_dir,
        base_name="lookup_table",
    )

    logger.info("\n" + "=" * 60)
    logger.info("LOOKUP TABLE GENERATED")
    logger.info("=" * 60)
    size_mb = saved_path.stat().st_size / (1024**2)
    logger.info(f"  {saved_path.name} ({size_mb:.1f} MB)")
    logger.info(f"  Location: {saved_path}")
    logger.info("=" * 60 + "\n")

    return 0


def curate_command(args):
    """Evaluate model, threshold languages, and write a pruned model config."""

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from transformers import AutoTokenizer

    from wldetect.config.loader import load_model_config, load_training_config, save_model_config
    from wldetect.embeddings.extractor import (
        extract_embeddings,
        get_cache_path,
        load_embeddings_as_memmap,
    )
    from wldetect.training.flores_eval import evaluate_on_flores, save_flores_evaluation
    from wldetect.training.model import LanguageDetectionModel

    logger = setup_logging()
    console = Console()

    # Load configs
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)

    # Load embeddings (memmap)
    cache_path = get_cache_path(model_config, cache_dir="artifacts/embeddings")
    if not cache_path.exists():
        logger.info(f"Embeddings not found at {cache_path}, extracting...")
        extract_embeddings(
            model_config,
            cache_dir="artifacts/embeddings",
            use_cache=True,
        )
    embeddings = load_embeddings_as_memmap(cache_path)

    # Tokenizer
    first_model = model_config.model if model_config.model else model_config.models[0]
    tokenizer = AutoTokenizer.from_pretrained(first_model.name)

    # Device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error("Error: CUDA requested but not available")
            return 1
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    vocab_size = embeddings.shape[0]
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(args.embedding_dtype, torch.float32)
    embeddings_tensor = torch.from_numpy(embeddings).clone().to(device=device, dtype=target_dtype)

    model = LanguageDetectionModel(
        hidden_dim=model_config.hidden_dim,
        n_languages=model_config.n_languages,
        vocab_size=vocab_size,
        embeddings=embeddings_tensor,
        dropout=config.training.projection.dropout,
        pooling=model_config.inference.pooling,
    )

    # Load checkpoint
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config.output.checkpoint_dir) / "best_model.pt"
    )
    if not checkpoint_path.exists():
        logger.error(f"Error: checkpoint not found at {checkpoint_path}")
        return 1
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        logger.warning(f"Warning: weights_only load failed ({exc}); retrying without weights_only")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Evaluate
    split = config.evaluation.flores_split
    batch_size = (
        args.batch_size or config.evaluation.flores_batch_size or config.training.batch_size
    )
    results = evaluate_on_flores(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        split=split,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        device=device,
        hf_dataset=config.evaluation.flores_hf_dataset,
        cache_dir=config.evaluation.flores_cache_dir,
    )

    # Save metrics if requested
    if args.output_metrics:
        save_flores_evaluation(results, args.output_metrics)

    # Threshold languages
    per_lang = results.get("per_language", {})
    good_codes = []
    for lang, metrics in per_lang.items():
        if (
            metrics.get("n_samples", 0) >= args.min_samples
            and metrics.get("accuracy", 0.0) >= args.accuracy_threshold
            and metrics.get("f1", 0.0) >= args.f1_threshold
        ):
            good_codes.append(lang)

    if not good_codes:
        logger.warning("No languages met the thresholds; not writing pruned config.")
        return 1

    # Preserve original order from model_config.languages
    pruned_languages = {}
    for lang_code in model_config.languages:
        if lang_code in good_codes:
            pruned_languages[lang_code] = len(pruned_languages)

    pruned_config = model_config.model_copy(update={"languages": pruned_languages})
    save_model_config(pruned_config, Path(args.output_config))

    # Display curation summary with rich formatting
    console.print("\n")
    console.print(Panel("[bold green]CURATION SUMMARY[/bold green]", expand=False))

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green")
    summary_table.add_row(
        "Languages Kept", f"{len(pruned_languages)} / {len(model_config.languages)}"
    )
    summary_table.add_row(
        "Thresholds",
        f"acc >= {args.accuracy_threshold}, f1 >= {args.f1_threshold}, n >= {args.min_samples}",
    )
    summary_table.add_row("Config Output", str(args.output_config))
    if args.output_metrics:
        summary_table.add_row("Metrics Output", str(args.output_metrics))

    console.print(summary_table)

    return 0


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
        help="Evaluate a trained model on FLORES (using current config settings)",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path to load (default: best_model.pt in checkpoint_dir)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        help="Output path for evaluation metrics JSON (default: artifacts/flores_{split}.json)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (default: training batch size or evaluation override)",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on (default: auto)",
    )
    eval_parser.add_argument(
        "--embedding-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type to load embeddings with (default: float32)",
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

    # Curate command (evaluate + prune)
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

    # Execute command
    command_map = {
        "train": train_command,
        "eval": eval_command,
        "detect": detect_command,
        "create-lookup": create_lookup_command,
        "curate": curate_command,
    }

    return command_map[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
