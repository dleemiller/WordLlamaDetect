"""Tests for dataset loading and filtering."""

import pytest

Dataset = pytest.importorskip("datasets").Dataset
DatasetDict = pytest.importorskip("datasets").DatasetDict
dataset_module = pytest.importorskip("wldetect.data.dataset")

from wldetect.config import DatasetConfig

# Import functions from the skipped module
balance_dataset = dataset_module.balance_dataset
filter_dataset_by_languages = dataset_module.filter_dataset_by_languages
get_language_distribution = dataset_module.get_language_distribution
load_openlid_dataset = dataset_module.load_openlid_dataset
prepare_dataset = dataset_module.prepare_dataset


@pytest.fixture
def mock_openlid_data():
    """Create mock OpenLID dataset."""
    # 30 samples: 10 eng_Latn, 10 fra_Latn, 10 spa_Latn
    data = {
        "text": [f"Sample {i}" for i in range(30)],
        "language": (["eng_Latn"] * 10 + ["fra_Latn"] * 10 + ["spa_Latn"] * 10),
    }
    return Dataset.from_dict(data)


class TestLoadOpenlidDataset:
    """Test loading OpenLID dataset."""

    def test_load_openlid_success(self, mocker, mock_openlid_data):
        """Test successful dataset loading."""
        # Mock load_dataset to return our test data
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        result = load_openlid_dataset()

        assert "train" in result
        assert len(result["train"]) == 30

    def test_load_openlid_with_cache_dir(self, mocker, mock_openlid_data):
        """Test loading with custom cache directory."""
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mock_load = mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        load_openlid_dataset(cache_dir="/tmp/cache")

        mock_load.assert_called_once_with("laurievb/OpenLID-v2", cache_dir="/tmp/cache")

    def test_load_openlid_missing_train_split(self, mocker):
        """Test error when train split is missing."""
        mock_dataset = DatasetDict({"test": Dataset.from_dict({"text": [], "language": []})})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        with pytest.raises(ValueError, match="does not have a 'train' split"):
            load_openlid_dataset()


class TestFilterDatasetByLanguages:
    """Test language filtering."""

    def test_filter_keeps_specified_languages(self, mock_openlid_data):
        """Test filtering keeps only specified languages."""
        filtered = filter_dataset_by_languages(mock_openlid_data, ["eng_Latn", "fra_Latn"])

        assert len(filtered) == 20  # 10 eng + 10 fra
        langs = set(filtered["language"])
        assert langs == {"eng_Latn", "fra_Latn"}

    def test_filter_single_language(self, mock_openlid_data):
        """Test filtering to single language."""
        filtered = filter_dataset_by_languages(mock_openlid_data, ["spa_Latn"])

        assert len(filtered) == 10
        assert all(lang == "spa_Latn" for lang in filtered["language"])

    def test_filter_no_matches(self, mock_openlid_data):
        """Test filtering with no matching languages."""
        filtered = filter_dataset_by_languages(mock_openlid_data, ["deu_Latn", "ita_Latn"])

        assert len(filtered) == 0

    def test_filter_all_languages(self, mock_openlid_data):
        """Test filtering with all languages included."""
        filtered = filter_dataset_by_languages(
            mock_openlid_data, ["eng_Latn", "fra_Latn", "spa_Latn"]
        )

        assert len(filtered) == 30


class TestBalanceDataset:
    """Test dataset balancing."""

    def test_balance_reduces_samples(self, mock_openlid_data):
        """Test balancing limits samples per language."""
        balanced = balance_dataset(mock_openlid_data, max_samples_per_language=5)

        assert len(balanced) == 15  # 3 languages × 5 samples
        distribution = get_language_distribution(balanced)
        assert all(count == 5 for count in distribution.values())

    def test_balance_preserves_order_within_language(self, mock_openlid_data):
        """Test balancing keeps first N samples per language."""
        balanced = balance_dataset(mock_openlid_data, max_samples_per_language=3)

        # Check that we get the first 3 samples from each language
        eng_samples = [
            text
            for text, lang in zip(balanced["text"], balanced["language"], strict=True)
            if lang == "eng_Latn"
        ]
        assert eng_samples == ["Sample 0", "Sample 1", "Sample 2"]

    def test_balance_no_change_when_limit_exceeds_samples(self, mock_openlid_data):
        """Test balancing with limit higher than available samples."""
        balanced = balance_dataset(mock_openlid_data, max_samples_per_language=20)

        assert len(balanced) == 30  # No reduction

    def test_balance_uneven_distribution(self):
        """Test balancing with uneven language distribution."""
        # 5 eng, 15 fra, 3 spa
        data = {
            "text": [f"Sample {i}" for i in range(23)],
            "language": ["eng_Latn"] * 5 + ["fra_Latn"] * 15 + ["spa_Latn"] * 3,
        }
        dataset = Dataset.from_dict(data)

        balanced = balance_dataset(dataset, max_samples_per_language=4)

        distribution = get_language_distribution(balanced)
        assert distribution["eng_Latn"] == 4
        assert distribution["fra_Latn"] == 4
        assert distribution["spa_Latn"] == 3  # Only 3 available


class TestGetLanguageDistribution:
    """Test language distribution calculation."""

    def test_distribution_balanced(self, mock_openlid_data):
        """Test distribution with balanced dataset."""
        distribution = get_language_distribution(mock_openlid_data)

        assert distribution == {
            "eng_Latn": 10,
            "fra_Latn": 10,
            "spa_Latn": 10,
        }

    def test_distribution_unbalanced(self):
        """Test distribution with unbalanced dataset."""
        data = {
            "text": ["a"] * 20 + ["b"] * 5,
            "language": ["eng_Latn"] * 20 + ["fra_Latn"] * 5,
        }
        dataset = Dataset.from_dict(data)

        distribution = get_language_distribution(dataset)

        assert distribution == {"eng_Latn": 20, "fra_Latn": 5}

    def test_distribution_empty(self):
        """Test distribution with empty dataset."""
        dataset = Dataset.from_dict({"text": [], "language": []})

        distribution = get_language_distribution(dataset)

        assert distribution == {}


class TestPrepareDataset:
    """Test dataset preparation pipeline."""

    def test_prepare_with_filter_and_balance(self, mocker, mock_openlid_data):
        """Test full pipeline with filtering and balancing."""
        # Mock load_dataset
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        config = DatasetConfig(
            name="test/dataset",
            filter_languages=True,
            max_samples_per_language=5,
            shuffle_seed=42,
        )

        result = prepare_dataset(
            config,
            language_codes=["eng_Latn", "fra_Latn"],
        )

        # Should have only eng and fra, 5 samples each
        assert "train" in result
        assert len(result["train"]) == 10
        distribution = get_language_distribution(result["train"])
        assert distribution == {"eng_Latn": 5, "fra_Latn": 5}

    def test_prepare_no_filter(self, mocker, mock_openlid_data):
        """Test preparation without language filtering."""
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        config = DatasetConfig(
            name="test/dataset",
            filter_languages=False,
            max_samples_per_language=None,
            shuffle_seed=None,
        )

        result = prepare_dataset(config, language_codes=[])

        assert len(result["train"]) == 30  # All samples kept

    def test_prepare_only_balance(self, mocker, mock_openlid_data):
        """Test preparation with balancing but no filtering."""
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        config = DatasetConfig(
            name="test/dataset",
            filter_languages=False,
            max_samples_per_language=3,
            shuffle_seed=None,
        )

        result = prepare_dataset(config, language_codes=[])

        assert len(result["train"]) == 9  # 3 languages × 3 samples

    def test_prepare_with_shuffle(self, mocker, mock_openlid_data):
        """Test shuffling is applied with seed."""
        mock_dataset = DatasetDict({"train": mock_openlid_data})
        mocker.patch("wldetect.data.dataset.load_dataset", return_value=mock_dataset)

        config = DatasetConfig(
            name="test/dataset",
            filter_languages=False,
            max_samples_per_language=None,
            shuffle_seed=42,
        )

        result1 = prepare_dataset(config, language_codes=[])
        result2 = prepare_dataset(config, language_codes=[])

        # With same seed, should get same shuffle order
        assert result1["train"]["text"] == result2["train"]["text"]
