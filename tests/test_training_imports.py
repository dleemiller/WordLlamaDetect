"""Test that training imports work correctly.

This test would have caught the bug where LanguageDetectionDataset
was being imported from the wrong module.
"""


def test_training_datasets_imports():
    """Test that datasets module exports expected classes."""
    from wldetect.training.datasets import LanguageDetectionDataset, collate_fn

    assert LanguageDetectionDataset is not None
    assert collate_fn is not None
    assert callable(collate_fn)


def test_trainer_imports():
    """Test that trainer module exports expected classes."""
    from wldetect.training.trainer import Trainer

    assert Trainer is not None


def test_model_imports():
    """Test that model module exports expected classes."""
    from wldetect.training.model import LanguageDetectionModel

    assert LanguageDetectionModel is not None


def test_train_command_imports():
    """Test that train command can import all required training components.

    This simulates what the train command does when it imports dependencies.
    If this test passes, the train command should be able to import successfully.
    """
    # These imports should succeed
    from wldetect.training.datasets import LanguageDetectionDataset, collate_fn
    from wldetect.training.embeddings import EmbeddingsManager
    from wldetect.training.model import LanguageDetectionModel
    from wldetect.training.trainer import Trainer

    # Verify all imports are valid
    assert LanguageDetectionDataset is not None
    assert collate_fn is not None
    assert EmbeddingsManager is not None
    assert LanguageDetectionModel is not None
    assert Trainer is not None


def test_dataset_not_in_trainer():
    """Test that LanguageDetectionDataset is NOT in trainer module.

    This is a negative test to ensure the dataset class stays in the
    datasets module where it belongs.
    """
    from wldetect.training import trainer

    # LanguageDetectionDataset should NOT be in trainer module
    assert not hasattr(trainer, "LanguageDetectionDataset")


def test_collate_fn_not_in_trainer():
    """Test that collate_fn is NOT in trainer module.

    This is a negative test to ensure the collate function stays in the
    datasets module where it belongs.
    """
    from wldetect.training import trainer

    # collate_fn should NOT be in trainer module
    assert not hasattr(trainer, "collate_fn")
