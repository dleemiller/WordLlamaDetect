"""Create-lookup command - generate lookup table from checkpoint."""

from pathlib import Path

from wldetect.cli.utils import ensure_training_deps, print_header, setup_logging


def run(args) -> int:
    """Generate lookup table from a checkpoint.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging()

    if not ensure_training_deps(logger):
        return 1

    import torch

    from wldetect.config.loader import load_model_config, load_training_config
    from wldetect.training.embeddings import EmbeddingsManager
    from wldetect.training.lookup_table import (
        compute_lookup_table,
        compute_lookup_table_exp,
        save_lookup_table_exp,
    )

    print_header(logger, "LOOKUP TABLE GENERATION")

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
    token_mask = None
    if "token_mask" in state_dict:
        token_mask = state_dict["token_mask"].cpu().numpy().reshape(-1).astype(bool)

    logger.info(
        f"Extracted weights: projection={projection_weight.shape}, bias={projection_bias.shape}, token_weights={token_weights.shape}"
    )

    # Load embeddings
    logger.info("\nLoading embeddings from cache")
    embeddings_manager = EmbeddingsManager(model_config, cache_dir="artifacts/embeddings")

    try:
        embeddings = embeddings_manager.load_cached_embeddings()
        logger.info(f"Loaded embeddings: {embeddings.shape}")
    except FileNotFoundError:
        logger.error("Error: Embeddings cache not found")
        logger.error("Run training first to generate embeddings cache")
        return 1

    # Compute lookup table (raw logits)
    logger.info("\nComputing lookup table (raw logits)")
    lookup_table = compute_lookup_table(
        embeddings=embeddings,
        token_weights=token_weights,
        projection_weight=projection_weight,
        projection_bias=projection_bias,
    )

    # Apply exp and mask
    logger.info("\nApplying exp() and masking")
    lookup_table_exp = compute_lookup_table_exp(
        lookup_table_fp32=lookup_table,
        token_mask=token_mask,
    )

    # Save exp lookup table
    output_dir = Path(args.output_dir)
    threshold = getattr(args, "threshold", 10.0)
    sparse = not getattr(args, "dense", False)  # Default to sparse

    logger.info(f"\nSaving exp lookup table to {output_dir}")
    if sparse:
        logger.info(f"Format: sparse (threshold={threshold})")
    else:
        logger.info("Format: dense")

    exp_path = save_lookup_table_exp(
        lookup_table_exp=lookup_table_exp,
        output_dir=output_dir,
        base_name="lookup_table",
        threshold=threshold,
        sparse=sparse,
    )

    print_header(logger, "LOOKUP TABLE GENERATED")
    size_mb = exp_path.stat().st_size / (1024**2)
    logger.info(f"  {exp_path.name} ({size_mb:.1f} MB)")
    logger.info(f"  Location: {exp_path}")
    logger.info("=" * 60 + "\n")

    return 0
