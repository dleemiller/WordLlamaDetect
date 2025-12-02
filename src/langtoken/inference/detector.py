"""NumPy-only language detection."""

from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from langtoken.config.loader import load_model_config
from langtoken.config.models import ModelConfig
from langtoken.embeddings.extractor import load_embeddings
from langtoken.inference.utils import apply_projection, avg_pool, max_pool, softmax


class LanguageDetector:
    """NumPy-only language detector.

    This class performs language detection using only NumPy operations,
    without requiring PyTorch. It loads a trained projection matrix and
    uses static embeddings for inference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        projection_matrix_name: str = "projection.safetensors",
        config_name: str = "model_config.yaml",
        embeddings_cache_dir: str = "artifacts/embeddings",
    ):
        """Initialize language detector.

        Args:
            model_dir: Directory containing trained model artifacts
            projection_matrix_name: Name of projection matrix file
            config_name: Name of model config file
            embeddings_cache_dir: Directory containing cached embeddings
        """
        model_dir = Path(model_dir)

        # Load config
        config_path = model_dir / config_name
        self.config: ModelConfig = load_model_config(config_path)

        # Load projection matrix and token weights
        projection_path = model_dir / projection_matrix_name
        self.weight, self.bias, self.token_weights = self._load_projection_matrix(projection_path)

        # Load embeddings
        embeddings_path = self._find_embeddings_cache(embeddings_cache_dir)
        self.embeddings = load_embeddings(embeddings_path)

        # Load tokenizer
        first_model = self.config.all_models[0]
        self.tokenizer = Tokenizer.from_pretrained(first_model.name)

        # Get language mapping (code -> index)
        self.language_codes = sorted(
            self.config.languages.keys(), key=lambda k: self.config.languages[k]
        )
        self.index_to_language = {i: code for code, i in self.config.languages.items()}

        # Get pooling strategy
        self.pooling = self.config.inference.pooling
        self.max_length = self.config.inference.max_sequence_length

    def _load_projection_matrix(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load projection matrix and token weights from safetensors.

        Args:
            path: Path to safetensors file

        Returns:
            Tuple of (weight, bias, token_weights) as numpy arrays
        """
        with safe_open(path, framework="numpy") as f:
            weight = f.get_tensor("weight")
            bias = f.get_tensor("bias")
            token_weights = f.get_tensor("token_weights")
        return weight, bias, token_weights

    def _find_embeddings_cache(self, cache_dir: str) -> Path:
        """Find cached embeddings file.

        Args:
            cache_dir: Cache directory

        Returns:
            Path to embeddings cache file
        """
        from langtoken.embeddings.extractor import get_cache_path

        return get_cache_path(self.config, cache_dir)

    def tokenize(self, text: str) -> np.ndarray:
        """Tokenize text.

        Args:
            text: Input text

        Returns:
            Token IDs as numpy array
        """
        # Enable truncation
        self.tokenizer.enable_truncation(max_length=self.max_length)

        # Encode
        encoding = self.tokenizer.encode(text)

        return np.array(encoding.ids, dtype=np.int64)

    def detect(self, text: str) -> dict[str, float]:
        """Detect language of text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping language codes to probabilities
        """
        # Tokenize
        token_ids = self.tokenize(text)

        if len(token_ids) == 0:
            # Return uniform distribution if no tokens
            uniform_prob = 1.0 / self.config.n_languages
            return dict.fromkeys(self.language_codes, uniform_prob)

        # Lookup embeddings
        token_embeddings = self.embeddings[token_ids]  # (seq_len, hidden_dim)

        # Apply learnable per-token weights
        # token_weights[token_ids]: (seq_len, 1)
        weighted_embeddings = token_embeddings * self.token_weights[token_ids]

        # Project
        projected = apply_projection(
            weighted_embeddings,
            self.weight,
            self.bias,
        )  # (seq_len, n_languages)

        # Pool
        if self.pooling == "max":
            pooled = max_pool(projected, axis=0)  # (n_languages,)
        elif self.pooling == "average":
            pooled = avg_pool(projected, axis=0)  # (n_languages,)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Softmax
        probs = softmax(pooled)  # (n_languages,)

        # Map to language codes
        return {self.index_to_language[i]: float(probs[i]) for i in range(self.config.n_languages)}

    def detect_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Detect language for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries mapping language codes to probabilities
        """
        return [self.detect(text) for text in texts]

    def get_top_language(self, text: str) -> str:
        """Get the most likely language for text.

        Args:
            text: Input text

        Returns:
            Language code with highest probability
        """
        probs = self.detect(text)
        return max(probs.items(), key=lambda x: x[1])[0]
