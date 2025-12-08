"""Simple API for language detection."""

import logging
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from wldetect.config.loader import load_model_config
from wldetect.softmax import softmax

logger = logging.getLogger("wldetect")


class WLDetect:
    """WordLlama language detection API.

    Examples:
        >>> wld = WLDetect.load()
        >>> lang, conf = wld.predict("Hello world")
        >>> print(f"{lang}: {conf:.2%}")
        eng_Latn: 99.84%

        >>> predictions = wld.predict(["Hello", "Bonjour", "Hola"])
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

        # Load exp lookup table
        lookup_table_path = model_dir / "lookup_table_exp.safetensors"
        if not lookup_table_path.exists():
            raise FileNotFoundError(f"Exp lookup table not found: {lookup_table_path}")

        self.lookup_table = self._load_exp_lookup_table(lookup_table_path)
        logger.info(f"Loaded exp lookup table: {lookup_table_path.name}")

        # Load tokenizer
        first_model = self.config.all_models[0]
        self.tokenizer = Tokenizer.from_pretrained(first_model.name)

        # Language mapping
        self.index_to_language = {i: code for code, i in self.config.languages.items()}

        # Config
        self.max_length = self.config.inference.max_sequence_length

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WLDetect":
        """Load language detection model.

        Args:
            path: Path to model directory. If None, loads default bundled model.

        Returns:
            Initialized WLDetect instance
        """
        if path is None:
            # Default to bundled model in package
            path = Path(__file__).parent / "models"

        return cls(path)

    def _load_exp_lookup_table(self, path: Path) -> np.ndarray:
        """Load pre-exponentiated lookup table from safetensors.

        Supports both dense (dtype_id=32) and sparse COO (dtype_id=33) formats.

        Args:
            path: Path to exp lookup table file

        Returns:
            Lookup table as fp32 array (vocab_size, n_langs)
        """
        with safe_open(path, framework="numpy") as f:
            dtype_id = f.get_tensor("dtype")[0]

            if dtype_id == 32:
                # Dense format
                lookup_table = f.get_tensor("lookup_table").astype(np.float32)
                logger.info(
                    f"Loaded dense exp lookup table: shape={lookup_table.shape}, dtype={lookup_table.dtype}"
                )
                return lookup_table

            elif dtype_id == 33:
                # Sparse COO format
                data = f.get_tensor("data").astype(np.float32)
                row = f.get_tensor("row").astype(np.int32)
                col = f.get_tensor("col").astype(np.int32)
                shape = tuple(f.get_tensor("shape"))
                threshold = f.get_tensor("threshold")[0]

                # Reconstruct dense array
                lookup_table = np.zeros(shape, dtype=np.float32)
                lookup_table[row, col] = data

                sparsity = 100.0 * (1 - len(data) / (shape[0] * shape[1]))
                logger.info(f"Loaded sparse exp lookup table: shape={shape}, dtype=float32")
                logger.info(
                    f"  Sparse storage: {len(data):,} values ({sparsity:.2f}% sparse, threshold={threshold})"
                )
                return lookup_table

            else:
                raise ValueError(
                    f"Unknown dtype_id={dtype_id}. Expected 32 (dense) or 33 (sparse COO)"
                )

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize single text.

        Args:
            text: Input text

        Returns:
            Token IDs as numpy array
        """
        self.tokenizer.enable_truncation(max_length=self.max_length)
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return np.array(encoding.ids, dtype=np.int64)

    def _tokenize_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Tokenize multiple texts using batch encoding.

        Args:
            texts: List of input texts

        Returns:
            List of token ID arrays
        """
        self.tokenizer.enable_truncation(max_length=self.max_length)
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        return [np.array(enc.ids, dtype=np.int64) for enc in encodings]

    def _detect_from_tokens(self, token_ids: np.ndarray) -> tuple[str, float] | None:
        """Core detection logic from token IDs.

        Uses pre-exponentiated lookup table:
        - Lookup exp values for each token
        - Sum the exp values (this is sum(exp(logits)))
        - Take log to get logsumexp: log(sum(exp(logits)))
        - Apply softmax to get probabilities

        Args:
            token_ids: Token ID array

        Returns:
            Tuple of (language_code, confidence) or None if empty
        """
        if len(token_ids) == 0:
            return None

        # Lookup pre-exponentiated values: (seq_len, n_langs)
        exp_values = self.lookup_table[token_ids]

        # Sum exp values: (n_langs,)
        summed = np.sum(exp_values, axis=0)

        # Take log to complete logsumexp: log(sum(exp(logits)))
        pooled = np.log(np.maximum(summed, 1e-12))

        # Softmax to get probabilities
        probs = softmax(pooled)

        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_lang = self.index_to_language[top_idx]
        top_conf = float(probs[top_idx])

        return top_lang, top_conf

    def _detect_single(self, text: str) -> tuple[str, float] | None:
        """Detect language for a single text.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence) or None if empty
        """
        token_ids = self._tokenize(text)
        return self._detect_from_tokens(token_ids)

    def _detect_batch(self, texts: list[str]) -> list[tuple[str, float] | None]:
        """Detect language for multiple texts using batch tokenization.

        Args:
            texts: List of input texts

        Returns:
            List of (language_code, confidence) tuples or None for empty texts
        """
        all_token_ids = self._tokenize_batch(texts)
        return [self._detect_from_tokens(token_ids) for token_ids in all_token_ids]

    def predict(
        self, text: str | list[str]
    ) -> tuple[str, float] | None | list[tuple[str, float] | None]:
        """Predict language for text(s).

        Args:
            text: Single text string or list of text strings

        Returns:
            For single text: (language_code, confidence) or None if empty
            For list: [(language_code, confidence), ...] with None for empty texts
        """
        if isinstance(text, str):
            return self._detect_single(text)
        else:
            return self._detect_batch(text)
