"""Tests for custom loss functions."""

import torch

from wldetect.training.losses import FocalLoss


def test_focal_loss_matches_cross_entropy_when_gamma_zero():
    """Focal loss with gamma=0 should equal cross entropy."""
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])

    focal_loss = FocalLoss(gamma=0.0)
    ce_loss = torch.nn.CrossEntropyLoss()

    assert torch.allclose(focal_loss(logits, targets), ce_loss(logits, targets), atol=1e-6)


def test_focal_loss_applies_alpha_weights():
    """Alpha should scale losses per class."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])

    focal_loss = FocalLoss(gamma=0.0, alpha=[1.0, 2.0])
    base_losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, targets)
    expected = (base_losses * torch.tensor([1.0, 2.0])).mean()

    assert torch.allclose(focal_loss(logits, targets), expected, atol=1e-6)
