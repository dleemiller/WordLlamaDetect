"""Load embedding tensors from HuggingFace model shards."""

import hashlib
import re
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

from wldetect.config.models import SingleModelConfig


def find_embedding_shard(model_name: str, shard_pattern: str) -> str | None:
    """Find the shard file containing the embedding tensor.

    Args:
        model_name: HuggingFace model name
        shard_pattern: Pattern to match shard files (e.g., "model-*.safetensors")

    Returns:
        Filename of the shard containing embeddings, or None if not found
    """
    try:
        files = list_repo_files(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to list files for model {model_name}: {e}") from e

    # Convert glob pattern to regex
    pattern = shard_pattern.replace("*", ".*").replace("?", ".")
    regex = re.compile(pattern)

    # Filter files matching the pattern
    matching_files = [f for f in files if regex.match(f)]

    if not matching_files:
        raise FileNotFoundError(
            f"No files matching pattern '{shard_pattern}' found in {model_name}"
        )

    # Typically embeddings are in the first shard (model-00001-of-*.safetensors)
    # Or in a file like model.safetensors for smaller models
    # Sort to get the first shard
    matching_files.sort()
    return matching_files[0]


def download_embedding_shard(model_config: SingleModelConfig, cache_dir: str | None = None) -> Path:
    """Download the shard containing embedding tensor.

    Args:
        model_config: Model configuration
        cache_dir: Optional cache directory (default: HF cache)

    Returns:
        Path to downloaded shard file
    """
    shard_file = find_embedding_shard(model_config.name, model_config.shard_pattern)

    try:
        downloaded_path = hf_hub_download(
            repo_id=model_config.name,
            filename=shard_file,
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download shard {shard_file} from {model_config.name}: {e}"
        ) from e

    return Path(downloaded_path)


def load_embedding_from_shard(
    shard_path: Path,
    embedding_layer_name: str,
) -> np.ndarray:
    """Load embedding tensor from a safetensors shard.

    Args:
        shard_path: Path to safetensors shard file
        embedding_layer_name: Name of embedding layer in state dict

    Returns:
        Embedding tensor as numpy array (vocab_size, hidden_dim)
    """
    try:
        # Load with PyTorch framework to handle bfloat16 and other formats
        try:
            import torch  # noqa: F401

            with safe_open(shard_path, framework="pt") as f:
                if embedding_layer_name not in f.keys():
                    available_keys = list(f.keys())
                    raise KeyError(
                        f"Embedding layer '{embedding_layer_name}' not found in shard. "
                        f"Available keys: {available_keys}"
                    )
                embeddings_tensor = f.get_tensor(embedding_layer_name)
                # Convert to float32 and return as numpy array
                embeddings = embeddings_tensor.float().cpu().numpy()
                return embeddings
        except ImportError:
            # Fall back to numpy framework if torch not available (shouldn't happen during training)
            with safe_open(shard_path, framework="numpy") as f:
                if embedding_layer_name not in f.keys():
                    available_keys = list(f.keys())
                    raise KeyError(
                        f"Embedding layer '{embedding_layer_name}' not found in shard. "
                        f"Available keys: {available_keys}"
                    ) from None
                embeddings = f.get_tensor(embedding_layer_name)
                return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from {shard_path}: {e}") from e


def load_embeddings_from_model(
    model_config: SingleModelConfig,
    cache_dir: str | None = None,
) -> np.ndarray:
    """Load embeddings from a model configuration.

    Args:
        model_config: Model configuration
        cache_dir: Optional cache directory

    Returns:
        Embedding tensor as numpy array (vocab_size, hidden_dim)
    """
    shard_path = download_embedding_shard(model_config, cache_dir=cache_dir)
    embeddings = load_embedding_from_shard(shard_path, model_config.embedding_layer_name)

    # Validate shape
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding tensor, got shape {embeddings.shape}")

    vocab_size, hidden_dim = embeddings.shape
    if hidden_dim != model_config.hidden_dim:
        raise ValueError(
            f"Embedding hidden_dim {hidden_dim} doesn't match config {model_config.hidden_dim}"
        )

    return embeddings


def concatenate_embeddings(embeddings_list: list[np.ndarray]) -> np.ndarray:
    """Concatenate embeddings from multiple models.

    Args:
        embeddings_list: List of embedding tensors (vocab_size, hidden_dim)

    Returns:
        Concatenated embeddings (vocab_size, total_hidden_dim)

    Raises:
        ValueError: If vocab sizes don't match
    """
    if not embeddings_list:
        raise ValueError("Empty embeddings list")

    if len(embeddings_list) == 1:
        return embeddings_list[0]

    # Check all have same vocab size
    vocab_sizes = [emb.shape[0] for emb in embeddings_list]
    if len(set(vocab_sizes)) > 1:
        raise ValueError(f"Cannot concatenate embeddings with different vocab sizes: {vocab_sizes}")

    # Concatenate along hidden dimension
    return np.concatenate(embeddings_list, axis=1)


def get_model_hash(model_names: list[str]) -> str:
    """Generate a hash for a list of model names.

    Args:
        model_names: List of model names

    Returns:
        Short hash string
    """
    combined = "|".join(sorted(model_names))
    return hashlib.md5(combined.encode()).hexdigest()[:12]
