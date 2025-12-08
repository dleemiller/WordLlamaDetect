"""Basic tests to verify setup."""


def test_imports():
    """Test that package can be imported."""
    import wldetect

    assert wldetect.__version__.startswith("0.1.")


def test_config_imports():
    """Test config module imports."""
    from wldetect.config import loader, models

    assert hasattr(models, "ModelConfig")
    assert hasattr(models, "TrainingConfig")
    assert hasattr(loader, "load_model_config")
    assert hasattr(loader, "load_training_config")


def test_embeddings_imports():
    """Test embeddings module imports."""
    from wldetect.training.embeddings import EmbeddingsManager

    assert hasattr(EmbeddingsManager, "extract_embeddings")
    assert hasattr(EmbeddingsManager, "load_cached_embeddings")
    assert hasattr(EmbeddingsManager, "load_as_memmap")


def test_softmax_imports():
    """Test softmax module imports."""
    from wldetect import WLDetect, softmax

    assert hasattr(WLDetect, "load")
    assert hasattr(softmax, "softmax")
