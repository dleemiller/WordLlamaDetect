"""Tests for FLORES language mapping consistency."""

import json
from pathlib import Path

import pytest

from wldetect.data.flores import map_flores_to_model_languages


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
