"""Simple API for language detection."""

from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from langtoken.config.loader import load_model_config
from langtoken.inference.utils import logsumexp_pool, softmax


class LangToken:
    """Simple language detection API.

    Examples:
        >>> lt = LangToken.load()
        >>> lang, conf = lt.predict("Hello world")
        >>> print(f"{lang}: {conf:.2%}")
        eng_Latn: 99.84%

        >>> predictions = lt.predict(["Hello", "Bonjour", "Hola"])
        >>> for lang, conf in predictions:
        ...     print(f"{lang}: {conf:.2%}")
    """

    def __init__(self, model_dir: str | Path):
        """Initialize language detector.

        Args:
            model_dir: Directory containing model artifacts
        """
        model_dir = Path(model_dir)

        # Load config
        config_path = model_dir / "model_config.yaml"
        self.config = load_model_config(config_path)

        # Load fp8 lookup table
        lookup_table_path = model_dir / "lookup_table_fp8_e4m3fn.safetensors"
        self.lookup_table = self._load_fp8_lookup_table(lookup_table_path)

        # Load tokenizer
        first_model = self.config.all_models[0]
        self.tokenizer = Tokenizer.from_pretrained(first_model.name)

        # Language mapping
        self.index_to_language = {i: code for code, i in self.config.languages.items()}

        # Config
        self.max_length = self.config.inference.max_sequence_length
        self.pooling = self.config.inference.pooling

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LangToken":
        """Load language detection model.

        Args:
            path: Path to model directory. If None, loads default model.

        Returns:
            Initialized LangToken instance
        """
        if path is None:
            # Default to checked-in model
            path = Path(__file__).parent.parent.parent / "artifacts" / "gemma3-27b"

        return cls(path)

    def _load_fp8_lookup_table(self, path: Path) -> np.ndarray:
        """Load fp8 lookup table from safetensors.

        Args:
            path: Path to fp8 lookup table file

        Returns:
            Lookup table as fp32 numpy array (vocab_size, n_langs)
        """
        import ml_dtypes

        with safe_open(path, framework="numpy") as f:
            lookup_uint8 = f.get_tensor("lookup_table")
            dtype_id = f.get_tensor("dtype")
            shape = f.get_tensor("shape")

        # Reconstruct fp8 array
        lookup_table = lookup_uint8.reshape(shape)

        # View as fp8_e4m3fn
        if dtype_id[0] != 0:
            raise ValueError(f"Expected fp8_e4m3fn (dtype_id=0), got {dtype_id[0]}")

        lookup_fp8 = lookup_table.view(ml_dtypes.float8_e4m3fn)

        # Dequantize to fp32 for inference
        return lookup_fp8.astype(np.float32)

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize single text.

        Args:
            text: Input text

        Returns:
            Token IDs as numpy array
        """
        self.tokenizer.enable_truncation(max_length=self.max_length)
        encoding = self.tokenizer.encode(text)
        return np.array(encoding.ids, dtype=np.int64)

    def _tokenize_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Tokenize multiple texts using batch encoding.

        Args:
            texts: List of input texts

        Returns:
            List of token ID arrays
        """
        self.tokenizer.enable_truncation(max_length=self.max_length)
        encodings = self.tokenizer.encode_batch(texts)
        return [np.array(enc.ids, dtype=np.int64) for enc in encodings]

    def _detect_single(self, text: str) -> tuple[str, float]:
        """Detect language for a single text.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence)
        """
        # Tokenize
        token_ids = self._tokenize(text)

        if len(token_ids) == 0:
            # Return most common language with low confidence
            return "eng_Latn", 1.0 / self.config.n_languages

        # Lookup
        logits = self.lookup_table[token_ids]  # (seq_len, n_langs)

        # Pool
        if self.pooling == "logsumexp":
            pooled = logsumexp_pool(logits, axis=0)
        else:
            pooled = np.max(logits, axis=0)

        # Softmax
        probs = softmax(pooled)

        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_lang = self.index_to_language[top_idx]
        top_conf = float(probs[top_idx])

        return top_lang, top_conf

    def _detect_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Detect language for multiple texts using batch tokenization.

        Args:
            texts: List of input texts

        Returns:
            List of (language_code, confidence) tuples
        """
        # Batch tokenize (faster than individual calls)
        all_token_ids = self._tokenize_batch(texts)

        results = []
        for token_ids in all_token_ids:
            if len(token_ids) == 0:
                # Return most common language with low confidence
                results.append(("eng_Latn", 1.0 / self.config.n_languages))
                continue

            # Lookup
            logits = self.lookup_table[token_ids]  # (seq_len, n_langs)

            # Pool
            if self.pooling == "logsumexp":
                pooled = logsumexp_pool(logits, axis=0)
            else:
                pooled = np.max(logits, axis=0)

            # Softmax
            probs = softmax(pooled)

            # Get top prediction
            top_idx = int(np.argmax(probs))
            top_lang = self.index_to_language[top_idx]
            top_conf = float(probs[top_idx])

            results.append((top_lang, top_conf))

        return results

    def predict(self, text: str | list[str]) -> tuple[str, float] | list[tuple[str, float]]:
        """Predict language for text(s).

        Args:
            text: Single text string or list of text strings

        Returns:
            For single text: (language_code, confidence)
            For list: [(language_code, confidence), ...]
        """
        if isinstance(text, str):
            return self._detect_single(text)
        else:
            return self._detect_batch(text)
