"""Tests for EmbeddingsManager class (src/wldetect/embeddings/manager.py)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

from wldetect.config.models import ModelConfig, SingleModelConfig
from wldetect.embeddings.manager import EmbeddingsManager


class TestEmbeddingsManagerInit:
    """Test EmbeddingsManager initialization."""

    def test_init_with_defaults(self, minimal_model_config):
        """Test initialization with default cache directories."""
        manager = EmbeddingsManager(minimal_model_config)

        assert manager.model_config == minimal_model_config
        assert manager.cache_dir == Path("artifacts/embeddings")
        assert manager.hf_cache_dir is None

    def test_init_with_custom_cache_dirs(self, minimal_model_config, tmp_path):
        """Test initialization with custom cache directories."""
        cache_dir = tmp_path / "custom_cache"
        hf_cache = tmp_path / "hf_cache"

        manager = EmbeddingsManager(
            minimal_model_config, cache_dir=str(cache_dir), hf_cache_dir=str(hf_cache)
        )

        assert manager.cache_dir == cache_dir
        assert manager.hf_cache_dir == str(hf_cache)
        assert cache_dir.exists()  # Should create cache dir

    def test_init_creates_cache_directory(self, minimal_model_config, tmp_path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        EmbeddingsManager(minimal_model_config, cache_dir=str(cache_dir))

        assert cache_dir.exists()


class TestEmbeddingsManagerCachePath:
    """Test cache path generation."""

    def test_cache_path_format(self, minimal_model_config, tmp_path):
        """Test that cache path follows expected format."""
        manager = EmbeddingsManager(minimal_model_config, cache_dir=str(tmp_path))

        cache_path = manager._get_cache_path()

        assert cache_path.parent == tmp_path
        assert cache_path.name.startswith("embeddings_")
        assert cache_path.name.endswith("_3langs.safetensors")
        assert ".safetensors" in cache_path.name

    def test_cache_path_deterministic(self, minimal_model_config, tmp_path):
        """Test that cache path is deterministic for same config."""
        manager1 = EmbeddingsManager(minimal_model_config, cache_dir=str(tmp_path))
        manager2 = EmbeddingsManager(minimal_model_config, cache_dir=str(tmp_path))

        path1 = manager1._get_cache_path()
        path2 = manager2._get_cache_path()

        assert path1 == path2

    def test_cache_path_different_for_different_configs(
        self, minimal_model_config, realistic_model_config, tmp_path
    ):
        """Test that different configs produce different cache paths."""
        manager1 = EmbeddingsManager(minimal_model_config, cache_dir=str(tmp_path))
        manager2 = EmbeddingsManager(realistic_model_config, cache_dir=str(tmp_path))

        path1 = manager1._get_cache_path()
        path2 = manager2._get_cache_path()

        assert path1 != path2  # Different models = different paths


class TestEmbeddingsManagerSingleModel:
    """Test loading embeddings from a single model."""

    def test_load_single_model_success(self, mocker, tmp_path):
        """Test successful single model loading."""
        # Create test embeddings
        embeddings = torch.randn(100, 128)
        shard_path = tmp_path / "model.safetensors"
        save_file({"embeddings.weight": embeddings}, str(shard_path))

        # Mock HuggingFace functions
        mocker.patch(
            "wldetect.embeddings.manager.list_repo_files", return_value=["model.safetensors"]
        )
        mocker.patch("wldetect.embeddings.manager.hf_hub_download", return_value=str(shard_path))

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0, "fra": 1}),
            cache_dir=str(tmp_path / "cache"),
        )

        loaded = manager._load_single_model_embeddings(model_config)

        assert loaded.shape == (100, 128)
        assert loaded.dtype == np.float32

    def test_load_single_model_wrong_dim(self, mocker, tmp_path):
        """Test error when model hidden_dim doesn't match config."""
        # Create embeddings with wrong dimension
        embeddings = torch.randn(100, 256)  # 256 instead of 128
        shard_path = tmp_path / "model.safetensors"
        save_file({"embeddings.weight": embeddings}, str(shard_path))

        mocker.patch(
            "wldetect.embeddings.manager.list_repo_files", return_value=["model.safetensors"]
        )
        mocker.patch("wldetect.embeddings.manager.hf_hub_download", return_value=str(shard_path))

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,  # Expects 128
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(tmp_path / "cache")
        )

        with pytest.raises(ValueError, match="hidden_dim.*doesn't match"):
            manager._load_single_model_embeddings(model_config)


class TestEmbeddingsManagerExtract:
    """Test the main extract_embeddings() method."""

    def test_extract_embeddings_caches_result(self, mocker, tmp_path):
        """Test that extract_embeddings saves to cache."""
        # Create test embeddings
        embeddings = torch.randn(100, 128)
        shard_path = tmp_path / "model.safetensors"
        save_file({"embeddings.weight": embeddings}, str(shard_path))

        mocker.patch(
            "wldetect.embeddings.manager.list_repo_files", return_value=["model.safetensors"]
        )
        mocker.patch("wldetect.embeddings.manager.hf_hub_download", return_value=str(shard_path))

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        cache_dir = tmp_path / "cache"
        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(cache_dir)
        )

        # Extract (should download and cache)
        result = manager.extract_embeddings(use_cache=False)

        # Verify cache file was created
        cache_path = manager._get_cache_path()
        assert cache_path.exists()

        # Verify result
        assert result.shape == (100, 128)
        assert result.dtype == np.float32

    def test_extract_embeddings_uses_cache(self, mocker, tmp_path):
        """Test that extract_embeddings uses cache on second call."""
        # Create cached embeddings
        cached_embeddings = np.random.randn(100, 128).astype(np.float32)

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        cache_dir = tmp_path / "cache"
        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(cache_dir)
        )

        # Pre-populate cache
        cache_path = manager._get_cache_path()
        manager._save_to_cache(cached_embeddings, cache_path)

        # Mock to ensure download is NOT called
        mock_download = mocker.patch("wldetect.embeddings.manager.hf_hub_download")

        # Extract (should use cache)
        result = manager.extract_embeddings(use_cache=True)

        # Verify download was NOT called
        mock_download.assert_not_called()

        # Verify result matches cached embeddings
        np.testing.assert_array_equal(result, cached_embeddings)

    def test_extract_embeddings_skip_cache(self, mocker, tmp_path):
        """Test that extract_embeddings skips cache when use_cache=False."""
        # Create test embeddings
        embeddings = torch.randn(100, 128)
        shard_path = tmp_path / "model.safetensors"
        save_file({"embeddings.weight": embeddings}, str(shard_path))

        mocker.patch(
            "wldetect.embeddings.manager.list_repo_files", return_value=["model.safetensors"]
        )
        mock_download = mocker.patch(
            "wldetect.embeddings.manager.hf_hub_download", return_value=str(shard_path)
        )

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        cache_dir = tmp_path / "cache"
        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(cache_dir)
        )

        # Pre-populate cache with different data
        cache_path = manager._get_cache_path()
        old_embeddings = np.zeros((100, 128), dtype=np.float32)
        manager._save_to_cache(old_embeddings, cache_path)

        # Extract with use_cache=False (should ignore cache)
        result = manager.extract_embeddings(use_cache=False)

        # Verify download WAS called (cache was skipped)
        mock_download.assert_called_once()

        # Result should be new embeddings, not cached zeros
        assert not np.allclose(result, old_embeddings)


class TestEmbeddingsManagerLoadCached:
    """Test loading from cache."""

    def test_load_cached_embeddings_success(self, tmp_path):
        """Test loading cached embeddings."""
        embeddings = np.random.randn(100, 128).astype(np.float32)

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(tmp_path)
        )

        # Save to cache
        cache_path = manager._get_cache_path()
        manager._save_to_cache(embeddings, cache_path)

        # Load from cache
        loaded = manager.load_cached_embeddings()

        np.testing.assert_array_equal(loaded, embeddings)

    def test_load_cached_embeddings_missing(self, tmp_path):
        """Test error when cache doesn't exist."""
        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(tmp_path)
        )

        with pytest.raises(FileNotFoundError, match="Cached embeddings not found"):
            manager.load_cached_embeddings()


class TestEmbeddingsManagerMemmap:
    """Test memory-mapped loading."""

    def test_load_as_memmap_creates_npy(self, tmp_path):
        """Test that load_as_memmap creates .npy file."""
        embeddings = np.random.randn(100, 128).astype(np.float32)

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(tmp_path)
        )

        # Save to cache
        cache_path = manager._get_cache_path()
        manager._save_to_cache(embeddings, cache_path)

        # Load as memmap (should create .npy)
        memmap_path = cache_path.with_suffix(".npy")
        assert not memmap_path.exists()

        loaded = manager.load_as_memmap()

        # Verify .npy was created
        assert memmap_path.exists()

        # Verify data matches
        np.testing.assert_array_equal(loaded, embeddings)

    def test_load_as_memmap_reuses_npy(self, tmp_path):
        """Test that load_as_memmap reuses existing .npy file."""
        embeddings = np.random.randn(100, 128).astype(np.float32)

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(tmp_path)
        )

        # Save to cache
        cache_path = manager._get_cache_path()
        manager._save_to_cache(embeddings, cache_path)

        # First load (creates .npy)
        loaded1 = manager.load_as_memmap()

        # Get modification time of .npy
        memmap_path = cache_path.with_suffix(".npy")
        mtime1 = memmap_path.stat().st_mtime

        # Second load (should reuse .npy, not recreate)
        loaded2 = manager.load_as_memmap()

        mtime2 = memmap_path.stat().st_mtime

        # Modification time should be same (file not recreated)
        assert mtime1 == mtime2

        # Data should match
        np.testing.assert_array_equal(loaded1, loaded2)


class TestEmbeddingsManagerMultiModel:
    """Test multi-model concatenation."""

    def test_extract_multi_model_concatenates(self, mocker, tmp_path):
        """Test that multi-model extraction concatenates embeddings."""
        # Create embeddings for two models
        emb1 = torch.randn(100, 128)
        emb2 = torch.randn(100, 256)

        shard1 = tmp_path / "model1.safetensors"
        shard2 = tmp_path / "model2.safetensors"

        save_file({"embeddings.weight": emb1}, str(shard1))
        save_file({"embeddings.weight": emb2}, str(shard2))

        def mock_download(repo_id, **kwargs):
            if "model1" in repo_id:
                return str(shard1)
            else:
                return str(shard2)

        def mock_list_files(repo_id):
            if "model1" in repo_id:
                return ["model1.safetensors"]
            else:
                return ["model2.safetensors"]

        mocker.patch("wldetect.embeddings.manager.list_repo_files", side_effect=mock_list_files)
        mocker.patch("wldetect.embeddings.manager.hf_hub_download", side_effect=mock_download)

        model1 = SingleModelConfig(
            name="test/model1",
            type="test",
            hidden_dim=128,
            shard_pattern="model1.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        model2 = SingleModelConfig(
            name="test/model2",
            type="test",
            hidden_dim=256,
            shard_pattern="model2.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        # Multi-model config
        config = ModelConfig(models=[model1, model2], languages={"eng": 0})

        manager = EmbeddingsManager(config, cache_dir=str(tmp_path / "cache"))

        result = manager.extract_embeddings(use_cache=False)

        # Should be concatenated (128 + 256 = 384)
        assert result.shape == (100, 384)
        assert result.dtype == np.float32


class TestEmbeddingsManagerIntegration:
    """Integration tests with realistic scenarios."""

    def test_full_workflow_with_caching(self, mocker, tmp_path):
        """Test complete workflow: extract, cache, reload."""
        embeddings = torch.randn(100, 128)
        shard_path = tmp_path / "model.safetensors"
        save_file({"embeddings.weight": embeddings}, str(shard_path))

        mocker.patch(
            "wldetect.embeddings.manager.list_repo_files", return_value=["model.safetensors"]
        )
        mocker.patch("wldetect.embeddings.manager.hf_hub_download", return_value=str(shard_path))

        model_config = SingleModelConfig(
            name="test/model",
            type="test",
            hidden_dim=128,
            shard_pattern="model.safetensors",
            embedding_layer_name="embeddings.weight",
        )

        cache_dir = tmp_path / "cache"
        manager = EmbeddingsManager(
            ModelConfig(model=model_config, languages={"eng": 0}), cache_dir=str(cache_dir)
        )

        # Step 1: Extract (downloads and caches)
        result1 = manager.extract_embeddings()
        assert result1.shape == (100, 128)

        # Step 2: Extract again (uses cache)
        result2 = manager.extract_embeddings()
        np.testing.assert_array_equal(result1, result2)

        # Step 3: Load cached directly
        result3 = manager.load_cached_embeddings()
        np.testing.assert_array_equal(result1, result3)

        # Step 4: Load as memmap
        result4 = manager.load_as_memmap()
        np.testing.assert_array_equal(result1, result4)
