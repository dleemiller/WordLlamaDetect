"""Evaluation metrics for language detection."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from langtoken.training.model import LanguageDetectionModel

logger = logging.getLogger("langtoken")
console = Console()


class Evaluator:
    """Evaluator for language detection model."""

    def __init__(
        self,
        model: LanguageDetectionModel,
        language_codes: list[str],
        device: torch.device,
    ):
        """Initialize evaluator.

        Args:
            model: Language detection model
            language_codes: List of language codes (in order of indices)
            device: Device to run evaluation on
        """
        self.model = model
        self.language_codes = language_codes
        self.device = device

    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions for a dataset.

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (predictions, labels) as numpy arrays
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                embeddings = batch["embeddings"].to(self.device)
                token_ids = batch["token_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                logits = self.model(embeddings, token_ids)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        return predictions, labels

    def evaluate(self, data_loader: DataLoader) -> dict[str, float]:
        """Evaluate model and compute metrics.

        Args:
            data_loader: Data loader

        Returns:
            Dictionary of evaluation metrics
        """
        predictions, labels = self.predict(data_loader)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        }

        return metrics

    def get_confusion_matrix(self, data_loader: DataLoader) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            data_loader: Data loader

        Returns:
            Confusion matrix (n_languages, n_languages)
        """
        predictions, labels = self.predict(data_loader)
        cm = confusion_matrix(labels, predictions)
        return cm

    def print_evaluation_report(self, data_loader: DataLoader) -> None:
        """Print detailed evaluation report.

        Args:
            data_loader: Data loader
        """
        console.print(Panel("[bold cyan]EVALUATION REPORT[/bold cyan]", expand=False))

        # Compute metrics
        metrics = self.evaluate(data_loader)
        predictions, labels = self.predict(data_loader)

        # Overall metrics table
        overall_table = Table(title="Overall Metrics", show_header=False, box=None)
        overall_table.add_column("Metric", style="cyan", width=15)
        overall_table.add_column("Value", style="green", justify="right")
        overall_table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        overall_table.add_row("F1 (macro)", f"{metrics['f1_macro']:.4f}")
        overall_table.add_row("F1 (weighted)", f"{metrics['f1_weighted']:.4f}")
        console.print(overall_table)

        # Per-language metrics table
        lang_table = Table(title="Per-Language Metrics", show_header=True)
        lang_table.add_column("Language", style="cyan", width=10)
        lang_table.add_column("Accuracy", justify="right", style="green")
        lang_table.add_column("F1", justify="right", style="blue")
        lang_table.add_column("Samples", justify="right", style="yellow")

        for i, lang_code in enumerate(self.language_codes):
            lang_mask = labels == i
            if lang_mask.sum() == 0:
                continue

            lang_predictions = predictions[lang_mask]
            lang_labels = labels[lang_mask]

            lang_accuracy = accuracy_score(lang_labels, lang_predictions)
            lang_f1 = f1_score(
                lang_labels,
                lang_predictions,
                labels=[i],
                average="macro",
                zero_division=0,
            )

            lang_table.add_row(
                lang_code,
                f"{lang_accuracy:.4f}",
                f"{lang_f1:.4f}",
                str(int(lang_mask.sum())),
            )

        console.print(lang_table)

        # Confusion matrix (only show if not too large)
        if len(self.language_codes) <= 20:
            logger.info("Confusion matrix computed (see JSON output for full matrix)")
        else:
            logger.info(
                f"Confusion matrix too large ({len(self.language_codes)} languages) - skipping display"
            )

    def save_metrics_to_json(self, data_loader: DataLoader, output_path: str) -> None:
        """Save per-language evaluation metrics to JSON file.

        Args:
            data_loader: Data loader
            output_path: Output path for JSON file
        """
        # Compute metrics
        metrics = self.evaluate(data_loader)
        predictions, labels = self.predict(data_loader)

        # Build results dictionary
        results = {
            "overall": {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "f1_weighted": float(metrics["f1_weighted"]),
                "total_samples": int(len(labels)),
            },
            "per_language": {},
        }

        # Per-language metrics
        for i, lang_code in enumerate(self.language_codes):
            lang_mask = labels == i
            n_samples = int(lang_mask.sum())

            if n_samples == 0:
                # No samples for this language
                results["per_language"][lang_code] = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "n_samples": 0,
                    "support": 0.0,
                }
                continue

            lang_predictions = predictions[lang_mask]
            lang_labels = labels[lang_mask]

            lang_accuracy = accuracy_score(lang_labels, lang_predictions)
            lang_f1 = f1_score(
                lang_labels,
                lang_predictions,
                labels=[i],
                average="macro",
                zero_division=0,
            )

            results["per_language"][lang_code] = {
                "accuracy": float(lang_accuracy),
                "f1": float(lang_f1),
                "n_samples": n_samples,
                "support": float(n_samples / len(labels)),
            }

        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation metrics to {output_path}")


def save_projection_matrix(
    model: LanguageDetectionModel,
    output_path: str,
) -> None:
    """Save projection matrix and token weights to safetensors.

    Args:
        model: Trained model
        output_path: Output path for safetensors file
    """
    from pathlib import Path

    from safetensors.numpy import save_file

    # Get projection matrix (n_languages, hidden_dim)
    weight = model.get_projection_matrix().cpu().numpy()
    bias = model.get_projection_bias().cpu().numpy()
    token_weights = model.get_token_weights().cpu().numpy()

    # Save to safetensors
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_file(
        {
            "weight": weight,
            "bias": bias,
            "token_weights": token_weights,
        },
        str(output_path),
    )

    logger.info(f"Saved projection matrix and token weights to {output_path}")
    logger.info(
        f"  Shape: weight={weight.shape}, bias={bias.shape}, token_weights={token_weights.shape}"
    )
