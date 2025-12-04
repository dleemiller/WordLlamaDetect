"""Extract and cache embedding tensors from models."""

import logging
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from wldetect.config.models import ModelConfig
from wldetect.embeddings.loader import (
    concatenate_embeddings,
    get_model_hash,
    load_embeddings_from_model,
)

logger = logging.getLogger("wldetect")


def get_cache_path(
    model_config: ModelConfig,
    cache_dir: str = "artifacts/embeddings",
) -> Path:
    """Get the cache path for embeddings.

    Args:
        model_config: Model configuration
        cache_dir: Cache directory

    Returns:
        Path to cache file
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from model names
    model_names = [m.name for m in model_config.all_models]
    model_hash = get_model_hash(model_names)

    # Include number of languages in filename for clarity
    n_langs = model_config.n_languages
    filename = f"embeddings_{model_hash}_{n_langs}langs.safetensors"

    return cache_dir_path / filename


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Save embeddings to safetensors file.

    Args:
        embeddings: Embedding tensor (vocab_size, hidden_dim)
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": embeddings}, str(path))


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from safetensors file.

    Args:
        path: Path to embeddings file

    Returns:
        Embedding tensor (vocab_size, hidden_dim)
    """
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        return f.get_tensor("embeddings")


def load_embeddings_as_memmap(path: Path) -> np.ndarray:
    """Load embeddings as memory-mapped array for multi-worker efficiency.

    Creates a .npy file next to the safetensors file for memory mapping.
    This allows multiple workers to share the same memory without copying.

    Args:
        path: Path to embeddings safetensors file

    Returns:
        Memory-mapped embedding tensor (vocab_size, hidden_dim)
    """
    # Check if .npy memmap file exists
    memmap_path = path.with_suffix(".npy")

    if not memmap_path.exists():
        # Load from safetensors and save as npy for memmapping
        logger.info(f"Creating memory-mapped file: {memmap_path}")
        embeddings = load_embeddings(path)
        np.save(memmap_path, embeddings)

    # Load as memmap
    return np.load(memmap_path, mmap_mode="r")


def extract_embeddings(
    model_config: ModelConfig,
    cache_dir: str = "artifacts/embeddings",
    use_cache: bool = True,
    hf_cache_dir: str | None = None,
) -> np.ndarray:
    """Extract embeddings from model(s) with caching.

    Args:
        model_config: Model configuration
        cache_dir: Directory for caching extracted embeddings
        use_cache: Whether to use cached embeddings if available
        hf_cache_dir: Optional HuggingFace cache directory

    Returns:
        Embedding tensor (vocab_size, hidden_dim)
    """
    cache_path = get_cache_path(model_config, cache_dir)

    # Check cache
    if use_cache and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return load_embeddings(cache_path)

    logger.info("Extracting embeddings from model(s)...")

    # Load embeddings from each model
    embeddings_list = []
    for model in model_config.all_models:
        logger.info(f"  Loading embeddings from {model.name}")
        emb = load_embeddings_from_model(model, cache_dir=hf_cache_dir)
        embeddings_list.append(emb)

    # Concatenate if multiple models
    if len(embeddings_list) > 1:
        logger.info(f"  Concatenating {len(embeddings_list)} embedding tensors")
        embeddings = concatenate_embeddings(embeddings_list)
    else:
        embeddings = embeddings_list[0]

    vocab_size, hidden_dim = embeddings.shape
    logger.info(f"  Extracted embeddings: vocab_size={vocab_size}, hidden_dim={hidden_dim}")

    # Verify hidden_dim matches config
    if hidden_dim != model_config.hidden_dim:
        raise ValueError(
            f"Extracted hidden_dim {hidden_dim} doesn't match config {model_config.hidden_dim}"
        )

    # Save to cache
    logger.info(f"Saving embeddings to {cache_path}")
    save_embeddings(embeddings, cache_path)

    return embeddings
