"""Tests for WLDetect API (src/wldetect/api.py)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wldetect import WLDetect


class TestWLDetectInit:
    """Test WLDetect initialization."""

    def test_init_with_valid_directory(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test successful initialization with valid model directory."""
        wld = WLDetect(minimal_model_dir)

        assert wld.config.n_languages == 3
        assert wld.lookup_table.shape == (100, 3)
        assert wld.lookup_table.dtype == np.float32
        assert wld.max_length == 64
        assert len(wld.index_to_language) == 3
        assert wld.index_to_language[0] == "eng_Latn"

    def test_init_missing_config(self, tmp_path):
        """Test error when model_config.yaml is missing."""
        with pytest.raises(FileNotFoundError):
            WLDetect(tmp_path)

    def test_init_missing_lookup_table(self, tmp_path, minimal_model_config_dict):
        """Test error when lookup table is missing."""
        # Create config only
        config_path = tmp_path / "model_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(minimal_model_config_dict, f)

        with pytest.raises(FileNotFoundError):
            WLDetect(tmp_path)

    def test_init_with_path_object(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test initialization with Path object instead of string."""
        path_obj = Path(minimal_model_dir)
        wld = WLDetect(path_obj)

        assert wld.config.n_languages == 3
        assert isinstance(wld.lookup_table, np.ndarray)


class TestWLDetectLoad:
    """Test WLDetect.load() class method."""

    def test_load_with_explicit_path(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test load() with explicit path."""
        wld = WLDetect.load(minimal_model_dir)

        assert isinstance(wld, WLDetect)
        assert wld.config.n_languages == 3

    def test_load_with_string_path(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test load() with string path."""
        wld = WLDetect.load(str(minimal_model_dir))

        assert isinstance(wld, WLDetect)
        assert wld.lookup_table.shape == (100, 3)

    def test_load_default_bundled_model(
        self, mocker, minimal_model_dir, mock_tokenizer_from_pretrained
    ):
        """Test load() without path uses bundled model."""
        # Mock the default bundled model path
        import wldetect.api

        mock_module_path = minimal_model_dir.parent / "wldetect" / "api.py"

        # Patch __file__ to point to test directory structure
        mocker.patch.object(wldetect.api, "__file__", str(mock_module_path))

        # Create models subdirectory with required files
        models_dir = minimal_model_dir.parent / "wldetect" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Copy test files to models dir
        import shutil

        shutil.copy(minimal_model_dir / "model_config.yaml", models_dir / "model_config.yaml")
        shutil.copy(
            minimal_model_dir / "lookup_table_exp.safetensors",
            models_dir / "lookup_table_exp.safetensors",
        )

        wld = WLDetect.load()

        assert isinstance(wld, WLDetect)
        assert wld.config.n_languages == 3


class TestWLDetectPredict:
    """Test WLDetect.predict() method."""

    def test_predict_single_text(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test prediction on single text."""
        wld = WLDetect.load(minimal_model_dir)
        lang, conf = wld.predict("Hello world")

        assert isinstance(lang, str)
        assert lang in ["eng_Latn", "fra_Latn", "spa_Latn"]
        assert 0 <= conf <= 1
        assert isinstance(conf, float)

    def test_predict_batch(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test batch prediction."""
        wld = WLDetect.load(minimal_model_dir)
        results = wld.predict(["Hello", "Bonjour", "Hola"])

        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        for lang, conf in results:
            assert isinstance(lang, str)
            assert lang in ["eng_Latn", "fra_Latn", "spa_Latn"]
            assert 0 <= conf <= 1

    def test_predict_empty_text(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test prediction on empty text returns None."""
        wld = WLDetect.load(minimal_model_dir)
        result = wld.predict("")

        # Should return None for empty text
        assert result is None

    def test_predict_empty_in_batch(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test batch prediction handles empty strings."""
        wld = WLDetect.load(minimal_model_dir)
        results = wld.predict(["Hello", "", "Hola"])

        assert len(results) == 3

        # Empty text (index 1) should return None
        assert results[1] is None

        # Other results should be valid tuples
        assert results[0] is not None
        assert results[2] is not None

    def test_predict_long_text_truncation(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test that very long text is truncated correctly."""
        wld = WLDetect.load(minimal_model_dir)

        # Create text longer than max_length=64
        long_text = " ".join(["word"] * 200)
        lang, conf = wld.predict(long_text)

        # Should still work and return valid prediction
        assert isinstance(lang, str)
        assert 0 <= conf <= 1

    def test_predict_returns_correct_types(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test that predict returns correct Python types (not numpy types)."""
        wld = WLDetect.load(minimal_model_dir)

        lang, conf = wld.predict("Hello")

        # Should be Python str and float, not numpy types
        assert type(lang) is str  # noqa: E721
        assert type(conf) is float  # noqa: E721


class TestWLDetectPooling:
    """Test different pooling methods."""

    @pytest.mark.parametrize("pooling_method", ["logsumexp"])
    def test_different_pooling_methods(
        self,
        pooling_method,
        tmp_path,
        minimal_model_config_dict,
        minimal_exp_lookup_table,
        mock_tokenizer_from_pretrained,
    ):
        """Test that different pooling methods work."""
        # Create config with specific pooling
        config_dict = minimal_model_config_dict.copy()
        config_dict["inference"]["pooling"] = pooling_method

        model_dir = tmp_path / f"model_{pooling_method}"
        model_dir.mkdir()

        # Save config
        config_path = model_dir / "model_config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f)

        # Copy lookup table
        import shutil

        shutil.copy(minimal_exp_lookup_table, model_dir / "lookup_table_exp.safetensors")

        # Load and predict
        wld = WLDetect.load(model_dir)
        lang, conf = wld.predict("Hello world")

        assert isinstance(lang, str)
        assert 0 <= conf <= 1


class TestWLDetectBatchVsSingle:
    """Test consistency between batch and single predictions."""

    def test_batch_prediction_consistency(self, minimal_model_dir, mock_tokenizer_from_pretrained):
        """Test that batch prediction gives same results as individual predictions."""
        wld = WLDetect.load(minimal_model_dir)

        texts = ["Hello", "Bonjour", "Hola"]

        # Get individual predictions
        individual = [wld.predict(text) for text in texts]

        # Get batch predictions
        batch = wld.predict(texts)

        # Results should match
        assert len(individual) == len(batch)
        for ind, bat in zip(individual, batch, strict=True):
            assert ind[0] == bat[0]  # Same language
            assert abs(ind[1] - bat[1]) < 1e-6  # Same confidence (within floating point precision)


class TestWLDetectRealistic:
    """Integration tests with realistic fixtures."""

    def test_realistic_model_initialization(
        self, realistic_model_dir, mock_tokenizer_from_pretrained_realistic
    ):
        """Test initialization with realistic model (10 languages, 1000 vocab)."""
        wld = WLDetect.load(realistic_model_dir)

        assert wld.config.n_languages == 10
        assert wld.lookup_table.shape == (1000, 10)
        assert wld.max_length == 128

    def test_realistic_batch_prediction(
        self, realistic_model_dir, mock_tokenizer_from_pretrained_realistic
    ):
        """Test batch prediction with realistic model."""
        wld = WLDetect.load(realistic_model_dir)

        texts = [
            "Hello world",
            "Bonjour le monde",
            "Hola mundo",
            "Guten Tag",
            "Ciao",
            "",  # Empty text
            " ".join(["word"] * 200),  # Long text
        ]

        results = wld.predict(texts)

        assert len(results) == 7

        for i, result in enumerate(results):
            # Empty text (index 5) should return None
            if i == 5:
                assert result is None
            else:
                lang, conf = result
                assert isinstance(lang, str)
                assert 0 <= conf <= 1
