"""Eval command - evaluate model on FLORES dataset."""

from pathlib import Path

from wldetect.cli.utils import ensure_training_deps, print_header, setup_logging
from wldetect.tokenization import disable_chat_template


def _eval_inference_mode(args, logger) -> int:
    """Evaluate inference model (exp lookup table).

    Args:
        args: Command arguments
        logger: Logger instance

    Returns:
        Exit code
    """
    from wldetect import WLDetect
    from wldetect.training.flores_eval import (
        evaluate_on_flores_inference,
        save_confusion_heatmap,
        save_flores_evaluation,
    )

    print_header(logger, "WLDETECT INFERENCE EVALUATION")

    # Load model
    logger.info("\nStep 1: Load exp inference model")
    detector = WLDetect.load(args.model_path)
    logger.info(f"Model: {detector.config.model.name if detector.config.model else 'Multi-model'}")
    logger.info(f"Languages: {detector.config.n_languages}")

    # Evaluate on FLORES
    split = args.split
    batch_size = args.batch_size or 512
    logger.info(f"\nStep 2: Evaluate on FLORES '{split}' split")
    results = evaluate_on_flores_inference(
        detector=detector,
        model_config=detector.config,
        split=split,
        batch_size=batch_size,
        hf_dataset="openlanguagedata/flores_plus",
        cache_dir=None,
    )

    # Save results
    output_path = Path(args.output) if args.output else Path(f"flores_{split}_results.json")
    save_flores_evaluation(results, output_path)

    heatmap_path = output_path.with_suffix(".png")
    save_confusion_heatmap(results, list(detector.config.languages.keys()), heatmap_path)

    print_header(logger, "EVALUATION COMPLETE")
    logger.info(f"Metrics saved to: {output_path}")
    logger.info(f"Heatmap saved to: {heatmap_path}")
    logger.info("=" * 60 + "\n")

    return 0


def _eval_pytorch_mode(args, logger) -> int:
    """Evaluate PyTorch model.

    Args:
        args: Command arguments
        logger: Logger instance

    Returns:
        Exit code
    """
    if not ensure_training_deps(logger):
        return 1

    import torch
    from transformers import AutoTokenizer

    from wldetect.config.loader import load_model_config, load_training_config
    from wldetect.training.embeddings import EmbeddingsManager
    from wldetect.training.flores_eval import (
        evaluate_on_flores,
        save_confusion_heatmap,
        save_flores_evaluation,
    )
    from wldetect.training.model import LanguageDetectionModel

    print_header(logger, "WLDETECT PYTORCH EVALUATION")

    # Load configs
    logger.info("\nStep 1: Load configurations")
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)
    logger.info(f"Model: {model_config.model.name if model_config.model else 'Multi-model'}")
    logger.info(f"Languages: {model_config.n_languages}")

    # Load embeddings
    logger.info("\nStep 2: Load embeddings")
    embeddings_manager = EmbeddingsManager(model_config, cache_dir="artifacts/embeddings")
    embeddings_manager.extract_embeddings()
    embeddings = embeddings_manager.load_as_memmap()
    logger.info(f"  Memory-mapped embeddings: {embeddings.shape}, dtype={embeddings.dtype}")

    # Load tokenizer
    logger.info("\nStep 3: Load tokenizer")
    first_model = model_config.model if model_config.model else model_config.models[0]
    tokenizer = AutoTokenizer.from_pretrained(first_model.name)
    disable_chat_template(tokenizer)

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

    # Load model
    logger.info("\nStep 4: Load trained model")
    logger.info(f"  Loading embeddings to {device}...")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map.get(getattr(args, "embedding_dtype", "float32"), torch.float32)
    embeddings_tensor = torch.from_numpy(embeddings).clone().to(device=device, dtype=target_dtype)

    vocab_size = embeddings.shape[0]

    # Load checkpoint first to check for token_mask
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

    # Extract token_mask from checkpoint if present
    token_mask = None
    if "model_state_dict" in checkpoint and "token_mask" in checkpoint["model_state_dict"]:
        token_mask = checkpoint["model_state_dict"]["token_mask"]
        logger.info(f"  Found token_mask in checkpoint: {token_mask.sum().item():,} tokens enabled")

    # Create model with token_mask from checkpoint
    model = LanguageDetectionModel(
        hidden_dim=model_config.hidden_dim,
        n_languages=model_config.n_languages,
        vocab_size=vocab_size,
        embeddings=embeddings_tensor,
        dropout=config.training.projection_dropout,
        pooling=model_config.inference.pooling,
        token_mask=token_mask,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    # Evaluate on FLORES
    split = args.split
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

    # Save results
    output_path = (
        Path(args.output)
        if args.output
        else (Path(config.output.artifacts_dir) / f"flores_{split}_results.json")
    )
    save_flores_evaluation(results, output_path)

    heatmap_path = output_path.with_suffix(".png")
    save_confusion_heatmap(results, list(model_config.languages.keys()), heatmap_path)

    print_header(logger, "EVALUATION COMPLETE")
    logger.info(f"Metrics saved to: {output_path}")
    logger.info(f"Heatmap saved to: {heatmap_path}")
    logger.info("=" * 60 + "\n")

    return 0


def run(args) -> int:
    """Execute evaluation command.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging()

    if args.model_path:
        return _eval_inference_mode(args, logger)
    else:
        return _eval_pytorch_mode(args, logger)
