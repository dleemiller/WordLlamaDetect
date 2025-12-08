"""Report unmapped languages between FLORES+ and training labels."""

import json
from pathlib import Path

import pytest

flores_module = pytest.importorskip("wldetect.data.flores")
load_flores_languages = flores_module.load_flores_languages
map_flores_to_model_languages = flores_module.map_flores_to_model_languages


@pytest.fixture(scope="module")
def model_languages() -> dict[str, int]:
    langs_path = Path("ds_langs.json")
    if not langs_path.exists():
        pytest.skip("ds_langs.json not found; alignment test skipped")
    langs = json.loads(langs_path.read_text())
    return {lang: idx for idx, lang in enumerate(langs)}


def test_report_unmapped_flores_languages(model_languages):
    """Print unmapped FLORES codes and training-only codes for inspection."""
    flores_langs = load_flores_languages(
        split="dev",
        hf_dataset="openlanguagedata/flores_plus",
        cache_dir=None,
    )
    flores_codes = set(flores_langs.keys())
    model_codes = set(model_languages.keys())

    unmapped_from_flores = sorted(
        code
        for code in flores_codes
        if map_flores_to_model_languages(code, model_languages) is None
    )
    missing_from_flores = sorted(code for code in model_codes if code not in flores_codes)

    print(
        "\nUnmapped FLORES -> model:",
        unmapped_from_flores[:50],
        f"total={len(unmapped_from_flores)}",
    )
    print(
        "\nModel labels not in FLORES:",
        missing_from_flores[:50],
        f"total={len(missing_from_flores)}",
    )

    # The test should not fail, only report; assert True to mark as pass after printing.
    assert True
