"""Train command - train language detection model."""

from pathlib import Path

from wldetect.cli.utils import ensure_training_deps, print_header, setup_logging
from wldetect.tokenization import disable_chat_template


def run(args) -> int:
    """Execute training command.

    Args:
        args: Command arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = setup_logging()

    if not ensure_training_deps(logger):
        return 1

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from wldetect.config.loader import load_model_config, load_training_config, save_model_config
    from wldetect.data.dataset import prepare_dataset
    from wldetect.training.embeddings import EmbeddingsManager
    from wldetect.training.flores_eval import evaluate_on_flores, save_flores_evaluation
    from wldetect.training.lookup_table import (
        save_lookup_table_exp_from_model,
        save_projection_matrix,
    )
    from wldetect.training.model import LanguageDetectionModel
    from wldetect.training.trainer import (
        LanguageDetectionDataset,
        Trainer,
        collate_fn,
    )

    # Load configuration
    logger.info("Loading configuration...")
    config = load_training_config(args.config)
    model_config = load_model_config(config.model_config_path)

    print_header(logger, "WLDETECT TRAINING")
    logger.info(f"Model: {model_config.all_models[0].name}")
    logger.info(f"Languages: {len(model_config.languages)}")
    logger.info(f"Hidden dim: {model_config.hidden_dim}")
    logger.info("=" * 60 + "\n")

    # Extract embeddings
    logger.info("Step 1: Extract embeddings")
    embeddings_manager = EmbeddingsManager(model_config)
    embeddings_manager.extract_embeddings()
    embeddings = embeddings_manager.load_as_memmap()

    logger.info(f"  Embeddings: {embeddings.shape}, dtype={embeddings.dtype}")
    logger.info(f"  Memory: {embeddings.nbytes / 1024**3:.2f} GB (will be loaded to GPU)")

    # Load tokenizer
    logger.info("\nStep 2: Load tokenizer")
    first_model = model_config.all_models[0]
    tokenizer = AutoTokenizer.from_pretrained(first_model.name)
    disable_chat_template(tokenizer)

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
        prefetch_factor=4 if config.training.num_workers > 0 else None,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )

    # Create model
    logger.info("\nStep 5: Initialize model")
    vocab_size = embeddings.shape[0]

    logger.info("  Converting embeddings to torch float32 tensor...")
    embeddings_tensor = torch.from_numpy(embeddings).float()
    logger.info(
        f"  Embeddings tensor: {embeddings_tensor.shape}, "
        f"{embeddings_tensor.element_size() * embeddings_tensor.nelement() / 1024**3:.2f} GB"
    )

    # Load token mask if provided
    token_mask = None
    if config.training.token_mask_path is not None:
        import numpy as np

        logger.info(f"  Loading token mask from {config.training.token_mask_path}...")
        token_mask_np = np.load(config.training.token_mask_path)
        token_mask = torch.from_numpy(token_mask_np).bool()

        n_masked = (~token_mask).sum().item()
        logger.info(f"    Mask shape: {token_mask.shape}")
        logger.info(f"    Tokens to zero: {n_masked:,} ({n_masked / vocab_size * 100:.2f}%)")
        logger.info(f"    Tokens to train: {token_mask.sum().item():,}")

    model = LanguageDetectionModel(
        hidden_dim=model_config.hidden_dim,
        n_languages=model_config.n_languages,
        vocab_size=vocab_size,
        embeddings=embeddings_tensor,
        dropout=config.training.projection_dropout,
        pooling=model_config.inference.pooling,
        token_mask=token_mask,
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
    trainer.train(train_loader, val_loader=None)

    # Final evaluation on FLORES
    logger.info("\nStep 7: Final evaluation on FLORES")
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

    # Save FLORES evaluation metrics
    flores_metrics_path = (
        Path(config.output.artifacts_dir) / f"flores_{config.evaluation.flores_split}_results.json"
    )
    save_flores_evaluation(flores_results, flores_metrics_path)

    # Log FLORES results to TensorBoard
    trainer.writer.add_scalar("flores/accuracy", flores_results["overall"]["accuracy"], 0)
    trainer.writer.add_scalar("flores/f1_macro", flores_results["overall"]["f1_macro"], 0)
    trainer.writer.add_scalar("flores/f1_weighted", flores_results["overall"]["f1_weighted"], 0)

    # Log top/bottom language accuracies
    per_lang_metrics = flores_results["per_language"]
    lang_accuracies = {lang: metrics["accuracy"] for lang, metrics in per_lang_metrics.items()}
    sorted_langs = sorted(lang_accuracies.items(), key=lambda x: x[1], reverse=True)

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

    # Generate and save exp lookup table
    logger.info("\nStep 8b: Generate exp lookup table")
    lookup_table_path = save_lookup_table_exp_from_model(
        model=model,
        model_config=model_config,
        output_dir=config.output.artifacts_dir,
    )
    size_mb = lookup_table_path.stat().st_size / (1024**2)
    logger.info(f"  Saved: {lookup_table_path.name} ({size_mb:.1f} MB)")

    # Save model config
    config_path = Path(config.output.artifacts_dir) / config.output.config_name
    save_model_config(model_config, config_path)

    # Close TensorBoard writer
    trainer.close()

    print_header(logger, "TRAINING COMPLETE")
    logger.info(f"Artifacts saved to: {config.output.artifacts_dir}")
    logger.info(f"  - Projection matrix: {projection_path}")
    logger.info(f"  - Lookup table: {lookup_table_path.name}")
    logger.info(f"  - Model config: {config_path}")
    logger.info("=" * 60 + "\n")

    return 0
