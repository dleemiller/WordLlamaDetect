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
        label_smoothing: Optional label smoothing factor in [0,1).
        reduction: Reduction mode, one of "none", "mean", or "sum".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
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
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0 and num_classes > 1:
            smooth = self.label_smoothing / (num_classes - 1)
            true_dist = torch.full_like(log_probs, smooth)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * true_dist).sum(dim=1)
        focal_term = (1 - pt) ** self.gamma
        ce_loss = -(true_dist * log_probs).sum(dim=1)
        loss = focal_term * ce_loss

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
