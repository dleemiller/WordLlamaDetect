"""Class weight calculation for language detection training."""

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("wldetect")


def load_population_data(population_path: str | Path) -> dict[str, int]:
    """Load language speaker population data from YAML file.

    Args:
        population_path: Path to language_populations.yaml

    Returns:
        Dictionary mapping language codes to speaker counts
    """
    population_path = Path(population_path)

    if not population_path.exists():
        raise FileNotFoundError(f"Population data not found: {population_path}")

    with population_path.open() as f:
        pop_data = yaml.safe_load(f)

    # Extract just the speaker counts
    populations = {lang_code: data["speakers"] for lang_code, data in pop_data.items()}

    logger.info(f"Loaded population data for {len(populations)} languages")
    return populations


def calculate_class_weights(
    languages: dict[str, int],
    strategy: str,
    population_path: str | Path,
    power: float = 1.0,
) -> np.ndarray:
    """Calculate class weights for training.

    Args:
        languages: Dictionary mapping language codes to indices
        strategy: Weighting strategy - one of:
            - "none": No weighting (all 1.0)
            - "population": Weight proportional to speaker population
            - "inverse_population": Weight inversely proportional to population
            - "sqrt_inverse_population": Square root of inverse population
            - "balanced": Sklearn-style balanced weighting (inverse frequency)
        population_path: Path to language_populations.yaml
        power: Power to apply to population-based weights (1.0=linear)

    Returns:
        NumPy array of class weights, shape (n_classes,)
    """
    n_classes = len(languages)

    if strategy is None or strategy == "none":
        logger.info("Using uniform class weights (no weighting)")
        return np.ones(n_classes, dtype=np.float32)

    # Load population data
    populations = load_population_data(population_path)

    # Ensure all languages have population data
    missing = set(languages.keys()) - set(populations.keys())
    if missing:
        raise ValueError(f"Missing population data for languages: {sorted(missing)}")

    # Create array of populations aligned to class indices
    pop_array = np.zeros(n_classes, dtype=np.float64)
    for lang_code, class_idx in languages.items():
        pop_array[class_idx] = populations[lang_code]

    logger.info(f"Population range: {pop_array.min():,} - {pop_array.max():,} speakers")

    # Calculate weights based on strategy
    if strategy == "population":
        # Weight proportional to population
        weights = pop_array**power
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    elif strategy == "log_population":
        # Weight proportional to log(population) - balanced range
        # More weight to high-resource, but not extreme
        weights = np.log10(pop_array) ** power
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    elif strategy == "inverse_population":
        # Weight inversely proportional to population
        weights = 1.0 / (pop_array**power)
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    elif strategy == "inverse_log_population":
        # Weight inversely proportional to log(population)
        # Less weight to high-resource, but not extreme
        weights = 1.0 / (np.log10(pop_array) ** power)
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    elif strategy == "sqrt_inverse_population":
        # Square root of inverse (gentler than pure inverse)
        weights = 1.0 / np.sqrt(pop_array**power)
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    elif strategy == "balanced":
        # Sklearn-style: n_samples / (n_classes * class_counts)
        # For language detection, we approximate with inverse population
        # since we don't know actual dataset distribution
        weights = 1.0 / pop_array
        # Normalize so sum = n_classes
        weights = weights / weights.mean()

    else:
        raise ValueError(f"Unknown class weighting strategy: {strategy}")

    logger.info(f"Class weights computed using strategy: {strategy}")
    logger.info(f"  Weight range: {weights.min():.6f} - {weights.max():.6f}")
    logger.info(f"  Weight mean: {weights.mean():.6f} (should be ~1.0)")
    logger.info(f"  Weight std: {weights.std():.6f}")

    # Log extreme weights
    lang_codes = sorted(languages.keys(), key=lambda k: languages[k])
    top_5_idx = np.argsort(weights)[-5:][::-1]
    bottom_5_idx = np.argsort(weights)[:5]

    logger.info("  Top 5 weighted languages:")
    for idx in top_5_idx:
        lang = lang_codes[idx]
        logger.info(f"    {lang}: {weights[idx]:.4f} (pop: {pop_array[idx]:,})")

    logger.info("  Bottom 5 weighted languages:")
    for idx in bottom_5_idx:
        lang = lang_codes[idx]
        logger.info(f"    {lang}: {weights[idx]:.4f} (pop: {pop_array[idx]:,})")

    return weights.astype(np.float32)


def get_class_weights_for_training(
    languages: dict[str, int],
    config,
) -> np.ndarray | None:
    """Get class weights for training based on config.

    Args:
        languages: Dictionary mapping language codes to indices
        config: TrainingConfig with class weighting parameters

    Returns:
        Class weights array or None if no weighting
    """
    if config.training.class_weights is None or config.training.class_weights == "none":
        return None

    weights = calculate_class_weights(
        languages=languages,
        strategy=config.training.class_weights,
        population_path=config.training.population_data_path,
        power=config.training.class_weights_power,
    )

    return weights
