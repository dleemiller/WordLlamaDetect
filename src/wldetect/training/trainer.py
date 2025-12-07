"""Training loop for language detection model."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wldetect.config.models import TrainingConfig
from wldetect.data.flores import create_flores_dataset, get_flores_language_distribution
from wldetect.training.losses import FocalLoss
from wldetect.training.model import LanguageDetectionModel


class LanguageDetectionDataset(Dataset):
    """PyTorch dataset for language detection with lazy tokenization.

    IMPORTANT: Does NOT load embeddings - only tokenizes.
    Embeddings are looked up in collate_fn to prevent worker memory accumulation.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        language_to_idx: dict[str, int],
        max_length: int = 512,
        logger: logging.Logger | None = None,
    ):
        """Initialize dataset.

        Args:
            dataset: HuggingFace dataset with 'text' and 'language' columns
            tokenizer: Tokenizer instance
            language_to_idx: Mapping from language code to index
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.language_to_idx = language_to_idx
        self.max_length = max_length
        self.logger = logger or logging.getLogger("wldetect")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        text = example.get("text", "")
        if not isinstance(text, str) or text == "":
            if not hasattr(self, "_warned_missing_text"):
                self.logger.warning(
                    f"Missing or empty text at index {idx}; replacing with empty string"
                )
                self._warned_missing_text = True
            text = ""
        language = example["language"]

        # Tokenize on-the-fly
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Pad in collate_fn
            return_tensors=None,  # Get lists
            add_special_tokens=False,  # Keep raw tokens (no BOS/EOS or chat template)
        )

        # CRITICAL: Only return token IDs and labels (no embeddings!)
        # Workers don't accumulate large embedding tensors
        return {
            "token_ids": encoded["input_ids"],  # List of ints
            "labels": self.language_to_idx[language],  # Single int
        }


def collate_fn(batch):
    """Collate function for variable-length sequences.

    Simple padding - no embedding lookup needed (model does it on GPU!)

    Args:
        batch: List of examples with 'token_ids' and 'labels'

    Returns:
        Batched data with padded token_ids and labels
    """
    token_ids_list = [item["token_ids"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    # Find max length in batch
    max_len = max(len(ids) for ids in token_ids_list)

    # Pre-allocate padded token IDs
    batch_size = len(batch)
    padded_token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    # Fill in token IDs
    for i, token_ids in enumerate(token_ids_list):
        seq_len = len(token_ids)
        padded_token_ids[i, :seq_len] = torch.tensor(token_ids, dtype=torch.long)

    return {
        "token_ids": padded_token_ids,
        "labels": labels,
    }


class Trainer:
    """Trainer for language detection model."""

    def __init__(
        self,
        model: LanguageDetectionModel,
        config: TrainingConfig,
        device: torch.device | None = None,
        tokenizer=None,
        model_config=None,
        logger: logging.Logger | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Language detection model
            config: Training configuration
            device: Device to train on (default: cuda if available)
            tokenizer: Tokenizer for FLORES evaluation
            model_config: Model configuration (for FLORES languages)
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.logger = logger or logging.getLogger("wldetect")

        self.model.to(self.device)

        # Setup optimizer
        optimizer_class = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }[config.training.optimizer]

        optimizer_kwargs = {
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
        }

        # Add momentum for SGD
        if config.training.optimizer == "sgd":
            optimizer_kwargs["momentum"] = config.training.momentum

        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Calculate class weights if configured
        class_weights = None
        if config.training.class_weights and config.training.class_weights != "none":
            from wldetect.training.class_weights import get_class_weights_for_training

            class_weights_np = get_class_weights_for_training(
                languages=model_config.languages,
                config=config,
            )
            if class_weights_np is not None:
                class_weights = torch.from_numpy(class_weights_np).to(device)
                logger.info(f"Using class weights: {config.training.class_weights}")

        # Loss function
        if config.training.loss == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif config.training.loss == "focal":
            # For focal loss, prefer focal_alpha if specified, otherwise use class_weights
            alpha = config.training.focal_alpha
            if alpha is None and class_weights is not None:
                alpha = class_weights.cpu().numpy().tolist()
            self.criterion = FocalLoss(gamma=config.training.focal_gamma, alpha=alpha)
        else:
            raise ValueError(f"Unsupported loss type: {config.training.loss}")

        # Scheduler (will be created in train() when we know total steps)
        self.scheduler = None
        self.total_steps = None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self._flores_loader = None
        self._flores_lang_distribution = None
        self._flores_last_mapping_info = None

        # TensorBoard writer with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = Path(config.output.tensorboard_dir) / timestamp
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        self.logger.info(f"TensorBoard logging to: {tensorboard_dir}")

    def _get_flores_loader(self):
        """Lazily build FLORES DataLoader for periodic evaluation."""
        if self._flores_loader is not None:
            return self._flores_loader

        if self.tokenizer is None or self.model_config is None:
            self.logger.warning("FLORES evaluation skipped: tokenizer/model_config not provided")
            return None

        flores_split = self.config.evaluation.flores_split

        self.logger.info(f"\nPreparing FLORES dataset for periodic evaluation ({flores_split})...")
        flores_dataset, mapped_languages, skipped_languages = create_flores_dataset(
            self.model_config.languages,
            flores_split,
            hf_dataset=self.config.evaluation.flores_hf_dataset,
            cache_dir=self.config.evaluation.flores_cache_dir,
        )
        self._flores_lang_distribution = get_flores_language_distribution(
            self.model_config.languages,
            flores_split,
            hf_dataset=self.config.evaluation.flores_hf_dataset,
            cache_dir=self.config.evaluation.flores_cache_dir,
        )
        if skipped_languages:
            info = (
                f"Skipped FLORES languages (unmapped to model): "
                f"{sorted(skipped_languages)[:10]} (total {len(skipped_languages)})"
            )
            self.logger.warning(info)
            self._flores_last_mapping_info = info
        else:
            self._flores_last_mapping_info = "All FLORES languages mapped to model languages."

        eval_dataset = LanguageDetectionDataset(
            flores_dataset,
            self.tokenizer,
            self.model_config.languages,
            max_length=self.model_config.inference.max_sequence_length,
        )

        batch_size = self.config.evaluation.flores_batch_size or self.config.training.batch_size
        self._flores_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.config.training.num_workers > 0 else None,
            persistent_workers=True if self.config.training.num_workers > 0 else False,
        )
        self.logger.info(
            f"FLORES loader ready: {len(eval_dataset)} samples, "
            f"batch_size={batch_size}, split={flores_split}"
        )
        return self._flores_loader

    def _evaluate_flores_periodic(self):
        """Run FLORES-200 evaluation and log metrics."""
        loader = self._get_flores_loader()
        if loader is None:
            return

        was_training = self.model.training
        self.model.eval()

        predictions = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                token_ids = batch["token_ids"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                logits = self.model(token_ids)
                preds = torch.argmax(logits, dim=1)

                predictions.append(preds.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())

        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        preds_np = np.concatenate(predictions)
        labels_np = np.concatenate(labels)

        accuracy = accuracy_score(labels_np, preds_np)
        f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        f1_weighted = f1_score(labels_np, preds_np, average="weighted", zero_division=0)

        step_tag = self.global_step
        self.writer.add_scalar("flores/accuracy_step", accuracy, step_tag)
        self.writer.add_scalar("flores/f1_macro_step", f1_macro, step_tag)
        self.writer.add_scalar("flores/f1_weighted_step", f1_weighted, step_tag)

        # Per-language summary for quick inspection
        language_codes = sorted(
            self.model_config.languages.keys(), key=self.model_config.languages.get
        )
        per_lang = []
        for idx, lang in enumerate(language_codes):
            mask = labels_np == idx
            if mask.sum() == 0:
                continue
            lang_acc = accuracy_score(labels_np[mask], preds_np[mask])
            lang_f1 = f1_score(
                labels_np[mask],
                preds_np[mask],
                labels=[idx],
                average="macro",
                zero_division=0,
            )
            per_lang.append((lang, lang_acc, lang_f1, int(mask.sum())))

        if per_lang:
            per_lang_sorted = sorted(per_lang, key=lambda x: x[1], reverse=True)
            top = per_lang_sorted[:10]
            bottom = per_lang_sorted[-10:]

            def _fmt(rows):
                return "\n".join(
                    f"{lang}: acc={acc:.4f}, f1={f1:.4f}, n={n}" for lang, acc, f1, n in rows
                )

            summary_text = (
                "Top 10 Languages:\n" + _fmt(top) + "\n\nBottom 10 Languages:\n" + _fmt(bottom)
            )
            self.writer.add_text("flores/per_language_step", summary_text, step_tag)
            if self._flores_last_mapping_info:
                self.writer.add_text(
                    "flores/mapping_info", self._flores_last_mapping_info, step_tag
                )

        self.logger.info(
            f"[FLORES eval @ step {step_tag}] "
            f"accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}"
        )

        if was_training:
            self.model.train()

    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler.

        Args:
            total_steps: Total number of training steps

        Returns:
            Learning rate scheduler or None
        """
        if self.config.training.scheduler in [None, "none"]:
            return None

        max_lr = self.config.training.learning_rate
        min_lr = max_lr * self.config.training.min_lr_ratio

        if self.config.training.scheduler == "cosine":
            # Simple cosine annealing from max_lr to min_lr
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=min_lr,
            )
            self.logger.info(f"  Using cosine annealing: {max_lr:.2e} → {min_lr:.2e}")
            return scheduler

        elif self.config.training.scheduler == "cosine_warmup":
            # Cosine with linear warmup
            from torch.optim.lr_scheduler import LambdaLR

            warmup_steps = self.config.training.warmup_steps

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(min_lr / max_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

            scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.logger.info(
                f"  Using cosine with warmup: {warmup_steps} steps warmup, {max_lr:.2e} → {min_lr:.2e}"
            )
            return scheduler

        return None

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    token_ids = batch["token_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Forward pass (model does GPU embedding lookup internally)
                    logits = self.model(token_ids)
                    loss = self.criterion(logits, labels)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if self.config.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.gradient_clip,
                        )

                    self.optimizer.step()

                    # Step learning rate scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Track metrics - CRITICAL: Use .item() and no_grad to prevent memory leak
                    total_loss += loss.item()  # .item() converts to Python scalar
                    with torch.no_grad():  # Don't build computation graph for metrics
                        predictions = torch.argmax(logits, dim=1)
                        correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    self.global_step += 1

                    # Update progress bar with Python scalars only
                    postfix = {
                        "loss": loss.item(),
                        "acc": correct / total,
                    }
                    # Add learning rate if scheduler is used
                    if self.scheduler is not None:
                        postfix["lr"] = self.optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix(postfix)

                    # TensorBoard logging
                    self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                    self.writer.add_scalar("train/accuracy_step", correct / total, self.global_step)
                    if self.scheduler is not None:
                        self.writer.add_scalar(
                            "train/learning_rate",
                            self.optimizer.param_groups[0]["lr"],
                            self.global_step,
                        )

                    # Step-level checkpointing
                    checkpoint_every = self.config.output.checkpoint_every_steps
                    if checkpoint_every and self.global_step % checkpoint_every == 0:
                        self.save_checkpoint(epoch=self.current_epoch, step=self.global_step)

                    flores_every = self.config.evaluation.flores_eval_every_steps
                    if flores_every and self.global_step % flores_every == 0:
                        self._evaluate_flores_periodic()

                    # CRITICAL: Explicitly free tensors to prevent accumulation
                    del token_ids, labels, logits, loss, predictions

                    # Periodic memory cleanup every 50 batches
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc

                        gc.collect()

                except Exception as e:
                    self.logger.error(f"\n✗ Error in batch {batch_idx}: {e}", exc_info=True)
                    import traceback

                    traceback.print_exc()
                    raise

        except Exception as e:
            self.logger.error(f"\n✗ Fatal error in training epoch: {e}", exc_info=True)
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Final cleanup after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

        return {
            "loss": total_loss / len(train_loader),
            "accuracy": correct / total,
        }

    def evaluate(
        self,
        eval_loader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate model.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
                token_ids = batch["token_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass (model does GPU embedding lookup internally)
                logits = self.model(token_ids)
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # CRITICAL: Free tensors
                del token_ids, labels, logits, loss, predictions

                # Periodic cleanup
                if batch_idx % 50 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc

                    gc.collect()

        return {
            "loss": total_loss / len(eval_loader),
            "accuracy": correct / total,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        # Log hyperparameters to TensorBoard
        hparams = {
            "batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "epochs": self.config.training.epochs,
            "optimizer": self.config.training.optimizer,
            "weight_decay": self.config.training.weight_decay,
            "gradient_clip": self.config.training.gradient_clip,
            "scheduler": self.config.training.scheduler or "none",
            "warmup_steps": self.config.training.warmup_steps,
            "dropout": self.config.training.projection_dropout,
            "num_workers": self.config.training.num_workers,
        }
        self.writer.add_hparams(hparams, {})

        # Create learning rate scheduler (needs total steps)
        self.total_steps = len(train_loader) * self.config.training.epochs
        self.scheduler = self._create_scheduler(self.total_steps)

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.logger.info(
                f"Epoch {epoch} - Train loss: {train_metrics['loss']:.4f}, "
                f"accuracy: {train_metrics['accuracy']:.4f}"
            )

            # Log epoch metrics to TensorBoard
            self.writer.add_scalar("train/loss_epoch", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/accuracy_epoch", train_metrics["accuracy"], epoch)

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.logger.info(
                    f"Epoch {epoch} - Val loss: {val_metrics['loss']:.4f}, "
                    f"accuracy: {val_metrics['accuracy']:.4f}"
                )

                # Log validation metrics to TensorBoard
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)

                # Save best model
                self.save_best_model(val_metrics["loss"], val_metrics["accuracy"])

            # Save checkpoint
            self.save_checkpoint(epoch=epoch)

    def save_checkpoint(self, epoch: int | None = None, step: int | None = None) -> None:
        """Save model checkpoint."""
        if epoch is None and step is None:
            raise ValueError("Either epoch or step must be provided to save a checkpoint")

        checkpoint_dir = Path(self.config.output.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"epoch_{epoch}" if step is None else f"step_{step}"
        checkpoint_path = checkpoint_dir / f"checkpoint_{suffix}.pt"
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            checkpoint_path,
        )

        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def save_best_model(self, val_loss: float, val_accuracy: float) -> None:
        """Save best model based on validation metrics.

        Args:
            val_loss: Current validation loss
            val_accuracy: Current validation accuracy
        """
        checkpoint_dir = Path(self.config.output.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check if this is the best model (based on validation loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_accuracy = val_accuracy

            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": self.current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "global_step": self.global_step,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                best_model_path,
            )

            self.logger.info(
                f"  → New best model! Val loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}"
            )

    def save_final_model(self) -> None:
        """Save final trained model."""
        artifacts_dir = Path(self.config.output.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Saved final model to {model_path}")

    def close(self) -> None:
        """Close TensorBoard writer and flush remaining logs."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.logger.info("TensorBoard writer closed")
