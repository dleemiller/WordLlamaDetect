#!/usr/bin/env python3
"""Generate language support documentation from FLORES evaluation results.

This script:
1. Loads FLORES evaluation results (per-language metrics)
2. Loads model config for language list
3. Optionally loads language population data
4. Generates a markdown table with sortable columns
"""

import json
from pathlib import Path

import yaml


def load_flores_results(flores_results_path: Path) -> dict:
    """Load FLORES evaluation results."""
    with open(flores_results_path) as f:
        return json.load(f)


def load_language_populations(populations_path: Path | None) -> dict:
    """Load language population data (optional)."""
    if populations_path is None or not populations_path.exists():
        return {}

    with open(populations_path) as f:
        data = yaml.safe_load(f)
        return data.get("populations", {})


def format_number(num: int) -> str:
    """Format large numbers with abbreviations (M, B)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.0f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}K"
    else:
        return str(num)


def generate_language_docs(
    flores_results_path: Path,
    model_config_path: Path,
    output_path: Path,
    populations_path: Path | None = None,
):
    """Generate language support documentation.

    Args:
        flores_results_path: Path to FLORES evaluation results JSON
        model_config_path: Path to model_config.yaml
        output_path: Path to save generated markdown
        populations_path: Optional path to language_populations.yaml
    """
    print("=" * 80)
    print("GENERATING LANGUAGE SUPPORT DOCUMENTATION")
    print("=" * 80)
    print()

    # Load data
    print(f"Loading FLORES results from {flores_results_path}...")
    flores_results = load_flores_results(flores_results_path)

    populations = {}
    if populations_path:
        print(f"Loading population data from {populations_path}...")
        populations = load_language_populations(populations_path)

    # Extract metrics
    overall = flores_results["overall"]
    per_language = flores_results["per_language"]

    print(f"\n✅ Loaded {len(per_language)} languages")
    print(f"   Overall accuracy: {overall['accuracy']:.4f}")
    print(f"   Overall F1 (macro): {overall['f1_macro']:.4f}")
    print()

    # Calculate statistics
    accuracies = [metrics["accuracy"] for metrics in per_language.values()]
    high_acc_95 = sum(1 for acc in accuracies if acc >= 0.95)
    high_acc_90 = sum(1 for acc in accuracies if acc >= 0.90)

    # Sort languages by accuracy (descending)
    sorted_languages = sorted(
        per_language.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )

    # Generate markdown
    print(f"Generating markdown to {output_path}...")

    with open(output_path, "w") as f:
        # Header
        f.write("# Supported Languages\n\n")
        f.write(
            f"WLDetect supports **{len(per_language)} languages** trained on OpenLID-v2 "
            "and evaluated on FLORES+.\n\n"
        )

        # Performance summary
        f.write("## Performance Summary\n\n")
        f.write(f"- **Average Accuracy**: {overall['accuracy']:.2%}\n")
        f.write(f"- **Macro F1 Score**: {overall['f1_macro']:.4f}\n")
        f.write(f"- **Weighted F1 Score**: {overall['f1_weighted']:.4f}\n")
        f.write(f"- **Macro Precision**: {overall['precision_macro']:.4f}\n")
        f.write(f"- **Weighted Precision**: {overall['precision_weighted']:.4f}\n")
        f.write(f"- **Macro Recall**: {overall['recall_macro']:.4f}\n")
        f.write(f"- **Weighted Recall**: {overall['recall_weighted']:.4f}\n")
        f.write(f"- **Languages ≥ 95% accuracy**: {high_acc_95}/{len(per_language)}\n")
        f.write(f"- **Languages ≥ 90% accuracy**: {high_acc_90}/{len(per_language)}\n\n")

        # Language table
        f.write("## Language List\n\n")
        f.write("Languages sorted by FLORES accuracy (highest to lowest).\n\n")

        # Table header
        if populations:
            f.write("| Language Code | Accuracy | F1 | Precision | Recall | Speakers |\n")
            f.write("|--------------|----------|-----|-----------|--------|----------|\n")
        else:
            f.write("| Language Code | Accuracy | F1 | Precision | Recall |\n")
            f.write("|--------------|----------|-----|-----------|--------|\n")

        # Table rows
        for lang_code, metrics in sorted_languages:
            acc = metrics["accuracy"]
            f1 = metrics["f1"]
            precision = metrics["precision"]
            recall = metrics["recall"]

            if populations:
                speakers = populations.get(lang_code, "Unknown")
                if isinstance(speakers, int):
                    speakers = format_number(speakers)
                f.write(
                    f"| {lang_code} | {acc:.2%} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {speakers} |\n"
                )
            else:
                f.write(
                    f"| {lang_code} | {acc:.2%} | {f1:.4f} | {precision:.4f} | {recall:.4f} |\n"
                )

        # Notes section
        f.write("\n## Notes\n\n")
        f.write("- **Language Codes**: ISO 639-3 language code + ISO 15924 script code\n")
        f.write("  - Format: `{lang}_{Script}` (e.g., `eng_Latn` for English in Latin script)\n")
        f.write("- **FLORES Evaluation**: FLORES+ dev set (1012 sentences per language)\n")
        f.write(
            "- **Removed Languages**: Languages with high confusion or insufficient training data:\n"
        )
        f.write("  - `crh_Latn` (Crimean Tatar)\n")
        f.write("  - `ltz_Latn` (Luxembourgish)\n")

    print(f"✅ Generated language documentation: {output_path}")
    print()
    print("=" * 80)
    print("DOCUMENTATION GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate language support documentation")
    parser.add_argument(
        "--flores-results",
        type=str,
        default="artifacts/gemma3-27b/flores_dev_results.json",
        help="Path to FLORES evaluation results JSON",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="artifacts/gemma3-27b/model_config.yaml",
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/languages.md",
        help="Path to save generated markdown",
    )
    parser.add_argument(
        "--populations",
        type=str,
        default="configs/language_populations.yaml",
        help="Path to language_populations.yaml (optional)",
    )

    args = parser.parse_args()

    generate_language_docs(
        flores_results_path=Path(args.flores_results),
        model_config_path=Path(args.model_config),
        output_path=Path(args.output),
        populations_path=Path(args.populations) if args.populations else None,
    )
