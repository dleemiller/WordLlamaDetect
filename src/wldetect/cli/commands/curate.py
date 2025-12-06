"""Curate command - evaluate and prune languages based on thresholds."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from wldetect.cli.utils import ensure_training_deps, setup_logging


def run(args) -> int:
    """Evaluate model, threshold languages, and write pruned config.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging()

    if not ensure_training_deps(logger):
        return 1

    import torch
    from transformers import AutoTokenizer

    from wldetect.config.loader import load_model_config, load_training_config, save_model_config
    from wldetect.embeddings import EmbeddingsManager
    from wldetect.training.flores_eval import evaluate_on_flores, save_flores_evaluation
    from wldetect.training.model import LanguageDetectionModel

    console = Console()

    # Load configs
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)

    # Load embeddings
    embeddings_manager = EmbeddingsManager(model_config, cache_dir="artifacts/embeddings")
    embeddings_manager.extract_embeddings()
    embeddings = embeddings_manager.load_as_memmap()

    # Load tokenizer
    first_model = model_config.model if model_config.model else model_config.models[0]
    tokenizer = AutoTokenizer.from_pretrained(first_model.name)

    # Determine device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error("Error: CUDA requested but not available")
            return 1
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
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

    # Display curation summary
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
