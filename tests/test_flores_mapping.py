"""Tests for FLORES language mapping consistency."""

import json
from pathlib import Path

import pytest

from wldetect.data.flores import (
    create_flores_dataset,
    get_flores_language_distribution,
    load_flores_languages,
    map_flores_to_model_languages,
)


@pytest.fixture(scope="module")
def model_languages() -> dict[str, int]:
    """Load model languages from ds_langs.json (training set labels)."""
    langs_path = Path("ds_langs.json")
    if not langs_path.exists():
        pytest.skip("ds_langs.json not found; mapping test skipped")
    langs = json.loads(langs_path.read_text())
    return {lang: idx for idx, lang in enumerate(langs)}


def test_flores_to_model_mapping_alignment(model_languages):
    """Ensure FLORES codes that exist in training labels map directly; unmapped are surfaced."""
    # Codes present in ds_langs should map to themselves
    for code in list(model_languages.keys())[:20]:
        mapped = map_flores_to_model_languages(code, model_languages)
        assert mapped == code

    # Representative codes that may not align should return None (no remapping)
    possibly_unmapped = ["ajp_Arab", "arb_Latn", "kon_Latn", "zho_Hans"]
    for code in possibly_unmapped:
        mapped = map_flores_to_model_languages(code, model_languages)
        if mapped is not None:
            assert mapped in model_languages


class TestLoadFloresLanguages:
    """Test loading FLORES languages from HuggingFace."""

    def test_load_flores_success(self, mocker):
        """Test successful FLORES loading."""
        # Mock dataset with sample data
        mock_dataset = [
            {"iso_639_3": "eng", "iso_15924": "Latn", "text": "Hello world"},
            {"iso_639_3": "eng", "iso_15924": "Latn", "text": "Another sentence"},
            {"iso_639_3": "fra", "iso_15924": "Latn", "text": "Bonjour"},
        ]
        mocker.patch("datasets.load_dataset", return_value=mock_dataset)

        result = load_flores_languages(split="dev")

        assert "eng_Latn" in result
        assert "fra_Latn" in result
        assert len(result["eng_Latn"]) == 2
        assert len(result["fra_Latn"]) == 1
        assert result["eng_Latn"][0] == "Hello world"

    def test_load_flores_with_custom_dataset(self, mocker):
        """Test loading with custom dataset name."""
        mock_dataset = []
        mock_load = mocker.patch("datasets.load_dataset", return_value=mock_dataset)

        load_flores_languages(
            split="devtest",
            hf_dataset="custom/dataset",
            cache_dir="/tmp/cache",
        )

        mock_load.assert_called_once_with(
            "custom/dataset",
            split="devtest",
            cache_dir="/tmp/cache",
        )


class TestCreateFloresDataset:
    """Test creating FLORES evaluation dataset."""

    def test_create_dataset_all_mapped(self, mocker):
        """Test creating dataset when all languages map."""
        mock_languages = {
            "eng_Latn": ["Hello", "World"],
            "fra_Latn": ["Bonjour"],
        }
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0, "fra_Latn": 1}
        dataset, mapped, skipped = create_flores_dataset(model_langs, split="dev")

        assert len(dataset) == 3  # 2 eng + 1 fra
        assert len(mapped) == 2
        assert len(skipped) == 0
        assert mapped == {"eng_Latn": "eng_Latn", "fra_Latn": "fra_Latn"}

    def test_create_dataset_with_unmapped(self, mocker):
        """Test creating dataset with some unmapped languages."""
        mock_languages = {
            "eng_Latn": ["Hello"],
            "deu_Latn": ["Hallo"],  # Not in model
            "fra_Latn": ["Bonjour"],
        }
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0, "fra_Latn": 1}
        dataset, mapped, skipped = create_flores_dataset(model_langs, split="dev")

        assert len(dataset) == 2  # Only eng and fra
        assert len(mapped) == 2
        assert skipped == ["deu_Latn"]

    def test_create_dataset_structure(self, mocker):
        """Test dataset structure is correct."""
        mock_languages = {"eng_Latn": ["Hello", "World"]}
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0}
        dataset, _, _ = create_flores_dataset(model_langs, split="dev")

        assert all("text" in item and "language" in item for item in dataset)
        assert dataset[0] == {"text": "Hello", "language": "eng_Latn"}
        assert dataset[1] == {"text": "World", "language": "eng_Latn"}


class TestGetFloresLanguageDistribution:
    """Test getting FLORES language distribution."""

    def test_distribution_all_mapped(self, mocker):
        """Test distribution when all languages map."""
        mock_languages = {
            "eng_Latn": ["a", "b", "c"],
            "fra_Latn": ["d", "e"],
        }
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0, "fra_Latn": 1}
        distribution = get_flores_language_distribution(model_langs, split="dev")

        assert distribution == {"eng_Latn": 3, "fra_Latn": 2}

    def test_distribution_with_unmapped(self, mocker):
        """Test distribution excludes unmapped languages."""
        mock_languages = {
            "eng_Latn": ["a", "b"],
            "deu_Latn": ["c", "d", "e"],  # Not in model
            "fra_Latn": ["f"],
        }
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0, "fra_Latn": 1}
        distribution = get_flores_language_distribution(model_langs, split="dev")

        assert distribution == {"eng_Latn": 2, "fra_Latn": 1}
        assert "deu_Latn" not in distribution

    def test_distribution_empty(self, mocker):
        """Test distribution with no mapped languages."""
        mock_languages = {"deu_Latn": ["a", "b"]}
        mocker.patch("wldetect.data.flores.load_flores_languages", return_value=mock_languages)

        model_langs = {"eng_Latn": 0, "fra_Latn": 1}
        distribution = get_flores_language_distribution(model_langs, split="dev")

        assert distribution == {}
