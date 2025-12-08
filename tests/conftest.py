"""Shared test fixtures for WLDetect tests."""

import shutil

import ml_dtypes
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
    from wldetect.config.models import ModelConfig

    return ModelConfig(**minimal_model_config_dict)


@pytest.fixture
def realistic_model_config(realistic_model_config_dict):
    """ModelConfig instance from realistic dict."""
    from wldetect.config.models import ModelConfig

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


# ==== FP8 Lookup Table Fixtures (Legacy, for compatibility tests) ====


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


# ==== FP8 Lookup Table Fixtures (Legacy, for compatibility tests) ====


@pytest.fixture
def minimal_fp8_lookup_table(tmp_path):
    """Create minimal fp8 lookup table for testing (100 tokens × 3 languages)."""
    from safetensors.numpy import save_file

    # Small vocab (100 tokens), 3 languages
    vocab_size, n_langs = 100, 3

    # Create random logits in fp32
    np.random.seed(42)
    lookup_fp32 = np.random.randn(vocab_size, n_langs).astype(np.float32)

    # Quantize to fp8_e3m4
    lookup_fp8 = lookup_fp32.astype(ml_dtypes.float8_e3m4)
    lookup_uint8 = lookup_fp8.view(np.uint8)

    # Save with metadata
    path = tmp_path / "lookup_table_fp8_e3m4.safetensors"
    save_file(
        {
            "lookup_table": lookup_uint8,
            "dtype": np.array([26], dtype=np.uint8),  # 26 = fp8_e3m4
            "shape": np.array([vocab_size, n_langs], dtype=np.int64),
        },
        str(path),
    )

    return path


@pytest.fixture
def minimal_fp16_lookup_table(tmp_path):
    """Create minimal fp16 lookup table (100 tokens × 3 languages)."""
    from safetensors.numpy import save_file

    vocab_size, n_langs = 100, 3
    np.random.seed(123)
    lookup_fp16 = (np.random.randn(vocab_size, n_langs) * 0.5).astype(np.float16)

    path = tmp_path / "lookup_table_fp16.safetensors"
    save_file(
        {
            "lookup_table": lookup_fp16,
            "dtype": np.array([31], dtype=np.uint8),  # 31 = fp16 raw
            "shape": np.array([vocab_size, n_langs], dtype=np.int64),
        },
        str(path),
    )

    return path


@pytest.fixture
def realistic_fp8_lookup_table(tmp_path):
    """Create realistic fp8 lookup table (1000 tokens × 10 languages)."""
    from safetensors.numpy import save_file

    vocab_size, n_langs = 1000, 10

    # Create random logits with more realistic patterns
    np.random.seed(42)
    # Add some structure: tokens have language affinity
    base_logits = np.random.randn(vocab_size, n_langs).astype(np.float32) * 2.0

    # Make some tokens language-specific (higher logits for specific language)
    for i in range(vocab_size):
        preferred_lang = i % n_langs
        base_logits[i, preferred_lang] += 3.0

    # Quantize to fp8_e3m4
    lookup_fp8 = base_logits.astype(ml_dtypes.float8_e3m4)
    lookup_uint8 = lookup_fp8.view(np.uint8)

    # Save with metadata
    path = tmp_path / "lookup_table_fp8_e3m4.safetensors"
    save_file(
        {
            "lookup_table": lookup_uint8,
            "dtype": np.array([26], dtype=np.uint8),
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
def minimal_model_dir_fp16(tmp_path, minimal_model_config_dict, minimal_exp_lookup_table):
    """Model directory - now uses exp format (fp16 is legacy)."""
    model_dir = tmp_path / "minimal_model_fp16"
    model_dir.mkdir()

    # Save model config
    config_path = model_dir / "model_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(minimal_model_config_dict, f)

    # Copy lookup table (now uses exp format)
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


@pytest.fixture
def minimal_openlid_dataset():
    """Mock minimal dataset (30 samples, 3 languages)."""
    data = {
        "text": ["Hello world", "Bonjour le monde", "Hola mundo"] * 10,
        "language": ["eng_Latn", "fra_Latn", "spa_Latn"] * 10,
    }
    return Dataset.from_dict(data)


@pytest.fixture
def realistic_openlid_dataset():
    """Mock realistic dataset (200 samples, 10 languages)."""
    texts = {
        "eng_Latn": ["Hello world", "How are you", "Good morning"],
        "fra_Latn": ["Bonjour le monde", "Comment allez-vous", "Bon matin"],
        "spa_Latn": ["Hola mundo", "Cómo estás", "Buenos días"],
        "deu_Latn": ["Hallo Welt", "Wie geht es dir", "Guten Morgen"],
        "ita_Latn": ["Ciao mondo", "Come stai", "Buongiorno"],
        "por_Latn": ["Olá mundo", "Como você está", "Bom dia"],
        "rus_Cyrl": ["Привет мир", "Как дела", "Доброе утро"],
        "zho_Hans": ["你好世界", "你好吗", "早上好"],
        "jpn_Jpan": ["こんにちは世界", "お元気ですか", "おはよう"],
        "ara_Arab": ["مرحبا العالم", "كيف حالك", "صباح الخير"],
    }

    data = {"text": [], "language": []}
    for lang, samples in texts.items():
        for sample in samples * 7:  # 21 samples per language = 210 total
            data["text"].append(sample)
            data["language"].append(lang)

    return Dataset.from_dict(data)


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


@pytest.fixture
def mock_load_dataset(mocker, minimal_openlid_dataset):
    """Mock datasets.load_dataset to return minimal dataset."""
    from datasets import DatasetDict

    mocker.patch(
        "datasets.load_dataset",
        return_value=DatasetDict({"train": minimal_openlid_dataset}),
    )


@pytest.fixture
def mock_load_dataset_realistic(mocker, realistic_openlid_dataset):
    """Mock datasets.load_dataset to return realistic dataset."""
    from datasets import DatasetDict

    mocker.patch(
        "datasets.load_dataset",
        return_value=DatasetDict({"train": realistic_openlid_dataset}),
    )


@pytest.fixture
def mock_hf_hub_download(mocker, tmp_path):
    """Mock HuggingFace hub downloads."""

    def mock_download(repo_id, filename, **kwargs):
        # Return path to a dummy file
        dummy_file = tmp_path / filename
        dummy_file.touch()
        return str(dummy_file)

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=mock_download)


@pytest.fixture
def mock_list_repo_files(mocker):
    """Mock HuggingFace repo file listing."""
    # Patch where it's imported in the loader module
    mocker.patch(
        "wldetect.embeddings.loader.list_repo_files",
        return_value=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
        ],
    )


# ==== Test Embeddings Fixtures ====


@pytest.fixture
def minimal_embeddings():
    """Create minimal test embeddings (100 tokens × 128 dim)."""
    np.random.seed(42)
    return np.random.randn(100, 128).astype(np.float32)


@pytest.fixture
def realistic_embeddings():
    """Create realistic test embeddings (1000 tokens × 256 dim)."""
    np.random.seed(42)
    return np.random.randn(1000, 256).astype(np.float32)
