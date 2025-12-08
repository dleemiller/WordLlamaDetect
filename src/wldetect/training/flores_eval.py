"""FLORES-200 evaluation for language detection models."""

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
from tqdm import tqdm

from wldetect.data.flores import create_flores_dataset, get_flores_language_distribution
from wldetect.training.datasets import LanguageDetectionDataset, collate_fn

logger = logging.getLogger("wldetect")
console = Console()


def _prepare_flores_dataset(
    model_config,
    split: str,
    hf_dataset: str | None,
    cache_dir: str | None,
    show_summary: bool = True,
):
    """Load FLORES dataset and language distribution."""
    hf_name = hf_dataset or "openlanguagedata/flores_plus"

    if show_summary:
        console.print(
            Panel(
                f"[bold cyan]FLORES EVALUATION[/bold cyan]\nSplit: {split} | Dataset: {hf_name}",
                expand=False,
            )
        )

    logger.info("Loading FLORES dataset...")
    flores_dataset, mapped_languages, skipped_languages = create_flores_dataset(
        model_config.languages,
        split,
        hf_dataset=hf_dataset,
        cache_dir=cache_dir,
    )
    logger.info(f"Total examples: {len(flores_dataset)}")

    distribution = get_flores_language_distribution(
        model_config.languages,
        split,
        hf_dataset=hf_dataset,
        cache_dir=cache_dir,
    )

    if show_summary:
        dist_table = Table(title="Language Distribution (Top 10)", show_header=True)
        dist_table.add_column("Language", style="cyan")
        dist_table.add_column("Sentences", justify="right", style="green")

        for lang, count in sorted(distribution.items(), key=lambda x: -x[1])[:10]:
            dist_table.add_row(lang, str(count))

        if len(distribution) > 10:
            dist_table.add_row(f"... and {len(distribution) - 10} more", "", style="dim")

        console.print(dist_table)

    if skipped_languages:
        logger.warning(
            f"Skipped {len(skipped_languages)} FLORES languages not mapped to model: "
            f"{sorted(skipped_languages)[:10]}{'...' if len(skipped_languages) > 10 else ''}"
        )

    return flores_dataset, skipped_languages, distribution


def _compute_metrics(labels: np.ndarray, predictions: np.ndarray, model_config) -> dict:
    """Compute overall and per-language metrics."""
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    language_codes = sorted(model_config.languages.keys(), key=lambda k: model_config.languages[k])
    per_language_metrics = {}

    for i, lang_code in enumerate(language_codes):
        lang_mask = labels == i
        n_samples = int(lang_mask.sum())
        if n_samples == 0:
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

        per_language_metrics[lang_code] = {
            "accuracy": float(lang_accuracy),
            "f1": float(lang_f1),
            "n_samples": n_samples,
            "support": float(n_samples / len(labels)),
        }

    cm = confusion_matrix(labels, predictions)

    return {
        "overall": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "total_samples": int(len(labels)),
        },
        "per_language": per_language_metrics,
        "confusion_matrix": cm.tolist(),
    }


def _print_metrics(metrics: dict) -> None:
    """Pretty-print overall and per-language metrics."""
    overall = metrics["overall"]
    per_language_metrics = metrics["per_language"]

    console.print("\n")
    console.print(Panel("[bold green]EVALUATION RESULTS[/bold green]", expand=False))

    overall_table = Table(title="Overall Metrics", show_header=False, box=None)
    overall_table.add_column("Metric", style="cyan", width=15)
    overall_table.add_column("Value", style="green", justify="right")
    overall_table.add_row("Accuracy", f"{overall['accuracy']:.4f}")
    overall_table.add_row("F1 (macro)", f"{overall['f1_macro']:.4f}")
    overall_table.add_row("F1 (weighted)", f"{overall['f1_weighted']:.4f}")
    overall_table.add_row("Total samples", f"{overall['total_samples']:,}")
    console.print(overall_table)

    sorted_langs = sorted(
        per_language_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    if sorted_langs:
        top_table = Table(title="Top 10 Languages", show_header=True)
        top_table.add_column("Language", style="cyan", width=10)
        top_table.add_column("Accuracy", justify="right", style="green")
        top_table.add_column("F1", justify="right", style="blue")
        top_table.add_column("Samples", justify="right", style="yellow")

        for lang, metrics_lang in sorted_langs[:10]:
            top_table.add_row(
                lang,
                f"{metrics_lang['accuracy']:.4f}",
                f"{metrics_lang['f1']:.4f}",
                f"{metrics_lang['n_samples']:,}",
            )
        console.print(top_table)

        bottom_table = Table(title="Bottom 10 Languages", show_header=True)
        bottom_table.add_column("Language", style="cyan", width=10)
        bottom_table.add_column("Accuracy", justify="right", style="red")
        bottom_table.add_column("F1", justify="right", style="blue")
        bottom_table.add_column("Samples", justify="right", style="yellow")

        for lang, metrics_lang in sorted_langs[-10:]:
            bottom_table.add_row(
                lang,
                f"{metrics_lang['accuracy']:.4f}",
                f"{metrics_lang['f1']:.4f}",
                f"{metrics_lang['n_samples']:,}",
            )
        console.print(bottom_table)


def compute_flores_metrics(labels: np.ndarray, predictions: np.ndarray, model_config) -> dict:
    """Public wrapper to compute FLORES metrics."""
    return _compute_metrics(labels, predictions, model_config)


def create_flores_eval_loader(
    model_config,
    tokenizer,
    split: str = "dev",
    batch_size: int = 32,
    num_workers: int = 0,
    hf_dataset: str | None = None,
    cache_dir: str | None = None,
    show_summary: bool = True,
):
    """Build a FLORES DataLoader for model evaluation (shared by trainer/CLI)."""
    flores_dataset, skipped_languages, distribution = _prepare_flores_dataset(
        model_config=model_config,
        split=split,
        hf_dataset=hf_dataset,
        cache_dir=cache_dir,
        show_summary=show_summary,
    )

    eval_dataset = LanguageDetectionDataset(
        flores_dataset,
        tokenizer,
        model_config.languages,
        max_length=model_config.inference.max_sequence_length,
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    if skipped_languages:
        mapping_info = (
            f"Skipped FLORES languages (unmapped to model): "
            f"{sorted(skipped_languages)[:10]} (total {len(skipped_languages)})"
        )
    else:
        mapping_info = "All FLORES languages mapped to model languages."

    return loader, skipped_languages, mapping_info, distribution


def evaluate_on_flores_inference(
    detector,
    model_config,
    split: str = "dev",
    batch_size: int = 512,
    hf_dataset: str | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Evaluate exp inference model on FLORES-200 dataset from HuggingFace."""
    flores_dataset, skipped_languages, _ = _prepare_flores_dataset(
        model_config=model_config,
        split=split,
        hf_dataset=hf_dataset,
        cache_dir=cache_dir,
    )

    all_predictions = []
    all_labels = []

    for i in tqdm(range(0, len(flores_dataset), batch_size), desc="Evaluating"):
        batch = flores_dataset[i : i + batch_size]
        texts = [sample["text"] for sample in batch]
        true_langs = [sample["language"] for sample in batch]

        results = detector.predict(texts)
        for (pred_lang, _), true_lang in zip(results, true_langs, strict=True):
            all_predictions.append(model_config.languages[pred_lang])
            all_labels.append(model_config.languages[true_lang])

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    metrics = _compute_metrics(labels, predictions, model_config)
    metrics["skipped_languages"] = skipped_languages
    _print_metrics(metrics)
    return metrics


def evaluate_on_flores(
    model,
    tokenizer,
    model_config,
    split: str = "dev",
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
    hf_dataset: str | None = None,
    cache_dir: str | None = None,
) -> dict:
    """Evaluate PyTorch model on FLORES-200 using a shared metrics pipeline."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_loader, skipped_languages, _, _ = create_flores_eval_loader(
        model_config=model_config,
        tokenizer=tokenizer,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        hf_dataset=hf_dataset,
        cache_dir=cache_dir,
        show_summary=True,
    )

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            token_ids = batch["token_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(token_ids)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    labels = np.concatenate(all_labels)

    metrics = _compute_metrics(labels, predictions, model_config)
    metrics["skipped_languages"] = skipped_languages
    _print_metrics(metrics)
    return metrics


def save_flores_evaluation(results: dict, output_path: str | Path) -> None:
    """Save FLORES-200 evaluation results to JSON.

    Args:
        results: Evaluation results dictionary
        output_path: Output path for JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved FLORES evaluation results to {output_path}")


def save_confusion_heatmap(results: dict, labels: list[str], output_path: str | Path) -> None:
    """Save confusion matrix as a heatmap image."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(results.get("confusion_matrix", []))
    if cm.size == 0:
        logger.warning("No confusion matrix in results; skipping heatmap.")
        return

    # Use log scale for better visualization (add 1 to avoid log(0))
    cm_log = np.log10(cm + 1)

    # Scale figure size based on number of languages
    n_langs = len(labels)
    fig_size = max(20, n_langs * 0.15)  # At least 20 inches, scale with languages

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_log,
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0,
        cbar_kws={"label": "log10(count + 1)"},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("FLORES Confusion Matrix (Log Scale)", fontsize=14)

    # Rotate labels and adjust font size
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved FLORES confusion heatmap to {output_path}")
