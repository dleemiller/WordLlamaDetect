"""Loss functions for training."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: Focusing parameter that down-weights easy examples.
        alpha: Optional weighting factor (scalar or per-class sequence).
        reduction: Reduction mode, one of "none", "mean", or "sum".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        elif isinstance(alpha, Sequence):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model outputs of shape (batch_size, num_classes).
            targets: Ground-truth class indices of shape (batch_size,).

        Returns:
            Loss value (scalar unless reduction="none").
        """
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.ndim == 0:
                loss = alpha * loss
            else:
                loss = alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
