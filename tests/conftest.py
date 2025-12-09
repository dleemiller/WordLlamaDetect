"""Shared test fixtures for WLDetect tests."""

import shutil

import numpy as np
import pytest
import yaml

Dataset = pytest.importorskip("datasets").Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# ==== Model Config Fixtures ====


@pytest.fixture
def minimal_model_config_dict():
    """Minimal valid model config as dict (3 languages, 100 vocab)."""
    return {
        "model": {
            "name": "test/tiny-model",
            "type": "test",
            "hidden_dim": 128,
        },
        "languages": {
            "eng_Latn": 0,
            "fra_Latn": 1,
            "spa_Latn": 2,
        },
        "inference": {
            "max_sequence_length": 64,
            "pooling": "logsumexp",
        },
    }


@pytest.fixture
def realistic_model_config_dict():
    """Realistic model config (10 languages, 1000 vocab)."""
    return {
        "model": {
            "name": "test/realistic-model",
            "type": "test",
            "hidden_dim": 256,
        },
        "languages": {
            "eng_Latn": 0,
            "fra_Latn": 1,
            "spa_Latn": 2,
            "deu_Latn": 3,
            "ita_Latn": 4,
            "por_Latn": 5,
            "rus_Cyrl": 6,
            "zho_Hans": 7,
            "jpn_Jpan": 8,
            "ara_Arab": 9,
        },
        "inference": {
            "max_sequence_length": 128,
            "pooling": "logsumexp",
        },
    }


@pytest.fixture
def minimal_model_config(minimal_model_config_dict):
    """ModelConfig instance from minimal dict."""
    from wldetect.config import ModelConfig

    return ModelConfig(**minimal_model_config_dict)


@pytest.fixture
def realistic_model_config(realistic_model_config_dict):
    """ModelConfig instance from realistic dict."""
    from wldetect.config import ModelConfig

    return ModelConfig(**realistic_model_config_dict)


# ==== Exp Lookup Table Fixtures ====


@pytest.fixture
def minimal_exp_lookup_table(tmp_path):
    """Create minimal exp lookup table for testing (100 tokens × 3 languages)."""
    from safetensors.numpy import save_file

    # Small vocab (100 tokens), 3 languages
    vocab_size, n_langs = 100, 3

    # Create random exp values (positive, as they come from exp())
    np.random.seed(42)
    lookup_exp = np.abs(np.random.randn(vocab_size, n_langs).astype(np.float32)) + 0.1

    # Save as dense exp format
    path = tmp_path / "lookup_table_exp.safetensors"
    save_file(
        {
            "lookup_table": lookup_exp,
            "dtype": np.array([32], dtype=np.uint8),  # 32 = dense exp format
            "shape": np.array([vocab_size, n_langs], dtype=np.int64),
        },
        str(path),
    )

    return path


@pytest.fixture
def realistic_exp_lookup_table(tmp_path):
    """Create realistic exp lookup table (1000 tokens × 10 languages)."""
    from safetensors.numpy import save_file

    vocab_size, n_langs = 1000, 10

    # Create random exp values (positive, as they come from exp())
    np.random.seed(42)
    # Add some structure: tokens have language affinity
    base_exp = np.abs(np.random.randn(vocab_size, n_langs).astype(np.float32)) + 0.1

    # Make some tokens language-specific (higher exp values for specific language)
    for i in range(vocab_size):
        preferred_lang = i % n_langs
        base_exp[i, preferred_lang] += 3.0

    # Save as dense exp format
    path = tmp_path / "lookup_table_exp.safetensors"
    save_file(
        {
            "lookup_table": base_exp,
            "dtype": np.array([32], dtype=np.uint8),  # 32 = dense exp format
            "shape": np.array([vocab_size, n_langs], dtype=np.int64),
        },
        str(path),
    )

    return path


# ==== Tokenizer Fixtures ====


@pytest.fixture
def minimal_tokenizer(tmp_path):
    """Create a minimal fast tokenizer (100 tokens)."""
    # Create simple vocab
    vocab = {f"token_{i}": i for i in range(98)}
    vocab["<unk>"] = 98
    vocab["<pad>"] = 99

    # Create tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Save and reload (mimics from_pretrained)
    path = tmp_path / "minimal_tokenizer.json"
    tokenizer.save(str(path))

    return Tokenizer.from_file(str(path))


@pytest.fixture
def realistic_tokenizer(tmp_path):
    """Create a realistic fast tokenizer (1000 tokens) with BPE."""
    from tokenizers import normalizers

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Add normalization and pre-tokenization
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Train on sample corpus
    trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["<unk>", "<pad>"])
    corpus = [
        "Hello world",
        "Bonjour le monde",
        "Hola mundo",
        "Guten Tag Welt",
        "Ciao mondo",
        "Olá mundo",
        "Привет мир",
        "你好世界",
        "こんにちは世界",
        "مرحبا العالم",
    ] * 50  # Repeat for sufficient training data

    tokenizer.train_from_iterator(corpus, trainer)

    # Save and reload
    path = tmp_path / "realistic_tokenizer.json"
    tokenizer.save(str(path))

    return Tokenizer.from_file(str(path))


# ==== Complete Model Directory Fixtures ====


@pytest.fixture
def minimal_model_dir(tmp_path, minimal_model_config_dict, minimal_exp_lookup_table):
    """Complete minimal model directory with all required files."""
    model_dir = tmp_path / "minimal_model"
    model_dir.mkdir()

    # Save model config
    config_path = model_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(minimal_model_config_dict, f)

    # Copy lookup table
    shutil.copy(minimal_exp_lookup_table, model_dir / "lookup_table_exp.safetensors")

    return model_dir


@pytest.fixture
def realistic_model_dir(tmp_path, realistic_model_config_dict, realistic_exp_lookup_table):
    """Complete realistic model directory with all required files."""
    model_dir = tmp_path / "realistic_model"
    model_dir.mkdir()

    # Save model config
    config_path = model_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(realistic_model_config_dict, f)

    # Copy lookup table
    shutil.copy(realistic_exp_lookup_table, model_dir / "lookup_table_exp.safetensors")

    return model_dir


# ==== Dataset Fixtures ====
# ==== Mocking Fixtures ====


@pytest.fixture
def mock_tokenizer_from_pretrained(mocker, minimal_tokenizer):
    """Mock Tokenizer.from_pretrained to avoid network calls."""
    mocker.patch("tokenizers.Tokenizer.from_pretrained", return_value=minimal_tokenizer)
    return minimal_tokenizer


@pytest.fixture
def mock_tokenizer_from_pretrained_realistic(mocker, realistic_tokenizer):
    """Mock Tokenizer.from_pretrained with realistic tokenizer."""
    mocker.patch("tokenizers.Tokenizer.from_pretrained", return_value=realistic_tokenizer)
    return realistic_tokenizer


# ==== Test Embeddings Fixtures ====
