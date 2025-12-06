"""Simple API for language detection."""

from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer

from wldetect.config.loader import load_model_config
from wldetect.inference.utils import avg_pool, logsumexp_pool, max_pool, softmax


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

        # Load fp8 lookup table (prefer E3M4, fallback to E4M3FN)
        e3m4_path = model_dir / "lookup_table_fp8_e3m4.safetensors"
        e4m3fn_path = model_dir / "lookup_table_fp8_e4m3fn.safetensors"

        if e3m4_path.exists():
            lookup_table_path = e3m4_path
        elif e4m3fn_path.exists():
            lookup_table_path = e4m3fn_path
        else:
            raise FileNotFoundError(
                f"No FP8 lookup table found in {model_dir}. "
                f"Expected either {e3m4_path.name} or {e4m3fn_path.name}"
            )

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

    def _load_fp8_lookup_table(self, path: Path) -> np.ndarray:
        """Load fp8 lookup table from safetensors.

        Supports both FP8 E3M4 (dtype_id=26) and E4M3FN (dtype_id=0) formats.

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
            # Load scale factor if present (for scaled quantization)
            scale = f.get_tensor("scale")[0] if "scale" in f.keys() else 1.0

        # Reconstruct fp8 array
        lookup_table = lookup_uint8.reshape(shape)

        # Determine format and dequantize
        if dtype_id[0] == 26:
            # FP8 E3M4: 3-bit exponent, 4-bit mantissa (better precision, smaller range)
            lookup_fp8 = lookup_table.view(ml_dtypes.float8_e3m4)
        elif dtype_id[0] == 0:
            # FP8 E4M3FN: 4-bit exponent, 3-bit mantissa (larger range, less precision)
            lookup_fp8 = lookup_table.view(ml_dtypes.float8_e4m3fn)
        else:
            raise ValueError(
                f"Unknown FP8 dtype_id={dtype_id[0]}. Expected 26 (E3M4) or 0 (E4M3FN)"
            )

        # Dequantize to fp32 and apply scale factor
        lookup_fp32 = lookup_fp8.astype(np.float32)
        if scale != 1.0:
            lookup_fp32 = lookup_fp32 * scale

        return lookup_fp32

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

        Args:
            token_ids: Token ID array

        Returns:
            Tuple of (language_code, confidence) or None if empty
        """
        if len(token_ids) == 0:
            return None

        # Lookup
        logits = self.lookup_table[token_ids]  # (seq_len, n_langs)

        # Pool
        if self.pooling == "logsumexp":
            pooled = logsumexp_pool(logits, axis=0)
        elif self.pooling == "average":
            pooled = avg_pool(logits, axis=0)
        else:  # max (and other pooling methods fall back to max)
            pooled = max_pool(logits, axis=0)

        # Softmax
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
