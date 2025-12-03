"""NumPy-only language detection using pre-computed lookup tables."""

from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from langtoken.config.loader import load_model_config
from langtoken.config.models import ModelConfig
from langtoken.inference.utils import avg_pool, logsumexp_pool, max_pool, softmax


class LanguageDetector:
    """NumPy-only language detector.

    This class performs language detection using only NumPy operations,
    without requiring PyTorch. It loads a pre-computed quantized lookup table
    for fast inference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        config_name: str = "model_config.yaml",
        lookup_table_name: str | None = None,
    ):
        """Initialize language detector.

        Args:
            model_dir: Directory containing trained model artifacts
            config_name: Name of model config file
            lookup_table_name: Optional lookup table filename override
        """
        model_dir = Path(model_dir)

        # Load config
        config_path = model_dir / config_name
        self.config: ModelConfig = load_model_config(config_path)

        # Determine lookup table filename
        if lookup_table_name is None:
            lookup_table_name = "lookup_table_fp8_e4m3fn.safetensors"

        # Load lookup table
        lookup_table_path = model_dir / lookup_table_name
        self.lookup_table = self._load_lookup_table(lookup_table_path)

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

    def _load_lookup_table(self, path: Path) -> np.ndarray:
        """Load pre-computed lookup table from safetensors.

        Args:
            path: Path to lookup table file

        Returns:
            Lookup table as fp32 numpy array (vocab_size, n_langs)
        """
        with safe_open(path, framework="numpy") as f:
            lookup_table = f.get_tensor("lookup_table")

            # Check if this is an fp8 file (stored as uint8 view)
            try:
                dtype_id = f.get_tensor("dtype")
                shape = f.get_tensor("shape")

                # This is a quantized fp8 file - need to reconstruct
                import ml_dtypes

                # Reshape uint8 to original shape
                lookup_table = lookup_table.reshape(shape)

                # View as appropriate fp8 dtype
                if dtype_id[0] == 0:  # fp8_e4m3fn
                    lookup_table = lookup_table.view(ml_dtypes.float8_e4m3fn)
                elif dtype_id[0] == 1:  # fp8_e5m2
                    lookup_table = lookup_table.view(ml_dtypes.float8_e5m2)

                # Dequantize to fp32
                lookup_table = lookup_table.astype(np.float32)

            except Exception:
                # Not an fp8 file, handle fp16 conversion
                if lookup_table.dtype == np.float16:
                    lookup_table = lookup_table.astype(np.float32)

        return lookup_table

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

        # Lookup pre-computed logits
        logits = self.lookup_table[token_ids]  # (seq_len, n_langs)

        # Pool
        if self.pooling == "max":
            pooled = max_pool(logits, axis=0)  # (n_languages,)
        elif self.pooling == "average":
            pooled = avg_pool(logits, axis=0)  # (n_languages,)
        elif self.pooling == "logsumexp":
            pooled = logsumexp_pool(logits, axis=0)  # (n_languages,)
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
