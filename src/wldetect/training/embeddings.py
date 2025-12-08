"""Embeddings manager - unified class for loading and caching embeddings."""

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.numpy import save_file

from wldetect.config.models import ModelConfig, SingleModelConfig

logger = logging.getLogger("wldetect")


class EmbeddingsManager:
    """Manages embedding extraction, caching, and loading.

    This class provides a unified interface for:
    - Downloading embeddings from HuggingFace models
    - Caching embeddings locally
    - Loading embeddings (normal or memory-mapped)
    - Handling multi-model concatenation

    Example:
        >>> from wldetect.config.loader import load_model_config
        >>> config = load_model_config("configs/models/qwen.yaml")
        >>> manager = EmbeddingsManager(config)
        >>> embeddings = manager.extract_embeddings()  # Downloads + caches
        >>> embeddings = manager.extract_embeddings()  # Uses cache
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_dir: str = "artifacts/embeddings",
        hf_cache_dir: str | None = None,
    ):
        """Initialize embeddings manager.

        Args:
            model_config: Model configuration (single or multi-model)
            cache_dir: Directory for caching extracted embeddings
            hf_cache_dir: Optional HuggingFace cache directory for model shards
        """
        self.model_config = model_config
        self.cache_dir = Path(cache_dir)
        self.hf_cache_dir = hf_cache_dir

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_embeddings(self, use_cache: bool = True) -> np.ndarray:
        """Extract embeddings from model(s) with caching.

        This is the main entry point for getting embeddings. It will:
        1. Check if cached embeddings exist (if use_cache=True)
        2. If not cached, download from HuggingFace
        3. Concatenate if multiple models
        4. Validate dimensions
        5. Save to cache
        6. Return embeddings

        Args:
            use_cache: Whether to use cached embeddings if available

        Returns:
            Embedding tensor (vocab_size, hidden_dim)

        Raises:
            ValueError: If extracted dimensions don't match config
        """
        cache_path = self._get_cache_path()

        # Check cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            return self.load_cached_embeddings()

        logger.info("Extracting embeddings from model(s)...")

        # Load embeddings from each model
        embeddings_list = []
        for model in self.model_config.all_models:
            logger.info(f"  Loading embeddings from {model.name}")
            emb = self._load_single_model_embeddings(model)
            embeddings_list.append(emb)

        # Concatenate if multiple models
        if len(embeddings_list) > 1:
            logger.info(f"  Concatenating {len(embeddings_list)} embedding tensors")
            embeddings = self._concatenate_embeddings(embeddings_list)
        else:
            embeddings = embeddings_list[0]

        vocab_size, hidden_dim = embeddings.shape
        logger.info(f"  Extracted embeddings: vocab_size={vocab_size}, hidden_dim={hidden_dim}")

        # Verify hidden_dim matches config
        if hidden_dim != self.model_config.hidden_dim:
            raise ValueError(
                f"Extracted hidden_dim {hidden_dim} doesn't match config {self.model_config.hidden_dim}"
            )

        # Save to cache
        logger.info(f"Saving embeddings to {cache_path}")
        self._save_to_cache(embeddings, cache_path)

        return embeddings

    def load_cached_embeddings(self) -> np.ndarray:
        """Load embeddings from cache.

        Returns:
            Embedding tensor (vocab_size, hidden_dim)

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            raise FileNotFoundError(f"Cached embeddings not found at {cache_path}")

        with safe_open(cache_path, framework="numpy") as f:
            return f.get_tensor("embeddings")

    def load_as_memmap(self) -> np.ndarray:
        """Load embeddings as memory-mapped array for multi-worker efficiency.

        Creates a .npy file next to the safetensors cache file for memory mapping.
        This allows multiple workers to share the same memory without copying.

        Returns:
            Memory-mapped embedding tensor (vocab_size, hidden_dim)

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            raise FileNotFoundError(f"Cached embeddings not found at {cache_path}")

        # Check if .npy memmap file exists
        memmap_path = cache_path.with_suffix(".npy")

        if not memmap_path.exists():
            # Load from safetensors and save as npy for memmapping
            logger.info(f"Creating memory-mapped file: {memmap_path}")
            embeddings = self.load_cached_embeddings()
            np.save(memmap_path, embeddings)

        # Load as memmap
        return np.load(memmap_path, mmap_mode="r")

    def _get_cache_path(self) -> Path:
        """Get the cache path for embeddings.

        Returns:
            Path to cache file (e.g., embeddings_<hash>_<n_langs>langs.safetensors)
        """
        # Generate filename from model names
        model_names = [m.name for m in self.model_config.all_models]
        model_hash = self._get_model_hash(model_names)

        # Include number of languages in filename for clarity
        n_langs = self.model_config.n_languages
        filename = f"embeddings_{model_hash}_{n_langs}langs.safetensors"

        return self.cache_dir / filename

    def _load_single_model_embeddings(self, model: SingleModelConfig) -> np.ndarray:
        """Load embeddings from a single model.

        This orchestrates the full loading pipeline:
        1. Find the shard containing embeddings
        2. Download the shard from HuggingFace
        3. Load embeddings from the shard
        4. Validate dimensions

        Args:
            model: Single model configuration

        Returns:
            Embedding tensor (vocab_size, hidden_dim)

        Raises:
            ValueError: If dimensions don't match config
        """
        # Download shard
        shard_path = self._download_shard(model)

        # Load embeddings from shard
        embeddings = self._load_from_shard(shard_path, model.embedding_layer_name)

        # Validate shape
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding tensor, got shape {embeddings.shape}")

        vocab_size, hidden_dim = embeddings.shape
        if hidden_dim != model.hidden_dim:
            raise ValueError(
                f"Embedding hidden_dim {hidden_dim} doesn't match config {model.hidden_dim}"
            )

        return embeddings

    def _download_shard(self, model: SingleModelConfig) -> Path:
        """Download the shard containing embedding tensor.

        Args:
            model: Single model configuration

        Returns:
            Path to downloaded shard file

        Raises:
            RuntimeError: If download fails
        """
        shard_file = self._find_shard(model)

        try:
            downloaded_path = hf_hub_download(
                repo_id=model.name,
                filename=shard_file,
                cache_dir=self.hf_cache_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download shard {shard_file} from {model.name}: {e}"
            ) from e

        return Path(downloaded_path)

    def _find_shard(self, model: SingleModelConfig) -> str:
        """Find the shard file containing the embedding tensor.

        Args:
            model: Single model configuration

        Returns:
            Filename of the shard containing embeddings

        Raises:
            RuntimeError: If listing files fails
            FileNotFoundError: If no matching shard found
        """
        try:
            files = list_repo_files(model.name)
        except Exception as e:
            raise RuntimeError(f"Failed to list files for model {model.name}: {e}") from e

        # Convert glob pattern to regex
        pattern = model.shard_pattern.replace("*", ".*").replace("?", ".")
        regex = re.compile(pattern)

        # Filter files matching the pattern
        matching_files = [f for f in files if regex.match(f)]

        if not matching_files:
            raise FileNotFoundError(
                f"No files matching pattern '{model.shard_pattern}' found in {model.name}"
            )

        # Typically embeddings are in the first shard (model-00001-of-*.safetensors)
        # Or in a file like model.safetensors for smaller models
        # Sort to get the first shard
        matching_files.sort()
        return matching_files[0]

    def _load_from_shard(self, shard_path: Path, embedding_layer_name: str) -> np.ndarray:
        """Load embedding tensor from a safetensors shard.

        Args:
            shard_path: Path to safetensors shard file
            embedding_layer_name: Name of embedding layer in state dict

        Returns:
            Embedding tensor as numpy array (vocab_size, hidden_dim)

        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Try loading with PyTorch framework to handle bfloat16 and other formats
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
                # Fall back to numpy framework if torch not available
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

    def _concatenate_embeddings(self, embeddings_list: list[np.ndarray]) -> np.ndarray:
        """Concatenate embeddings from multiple models.

        Args:
            embeddings_list: List of embedding tensors (vocab_size, hidden_dim)

        Returns:
            Concatenated embeddings (vocab_size, total_hidden_dim)

        Raises:
            ValueError: If vocab sizes don't match or list is empty
        """
        if not embeddings_list:
            raise ValueError("Empty embeddings list")

        if len(embeddings_list) == 1:
            return embeddings_list[0]

        # Check all have same vocab size
        vocab_sizes = [emb.shape[0] for emb in embeddings_list]
        if len(set(vocab_sizes)) > 1:
            raise ValueError(
                f"Cannot concatenate embeddings with different vocab sizes: {vocab_sizes}"
            )

        # Concatenate along hidden dimension
        return np.concatenate(embeddings_list, axis=1)

    def _save_to_cache(self, embeddings: np.ndarray, path: Path) -> None:
        """Save embeddings to safetensors file.

        Args:
            embeddings: Embedding tensor (vocab_size, hidden_dim)
            path: Output path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeddings": embeddings}, str(path))

    @staticmethod
    def _get_model_hash(model_names: list[str]) -> str:
        """Generate a hash for a list of model names.

        Args:
            model_names: List of model names

        Returns:
            Short hash string (12 characters)
        """
        combined = "|".join(sorted(model_names))
        return hashlib.md5(combined.encode()).hexdigest()[:12]
