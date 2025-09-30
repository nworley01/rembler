from __future__ import annotations

import torch

from rembler.utils.sleep_utils import int_to_stage


class PerClassMetric:
    """Custom metric to compute per-class accuracy."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()
        self.int_to_stage = int_to_stage
        self.correct: list[int] = [0] * self.num_classes
        self.total: list[int] = [0] * self.num_classes

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        assert preds_np.shape == targets_np.shape, (
            "Predictions and targets must have the same shape"
        )
        for cls in range(self.num_classes):
            cls_mask = targets_np == cls
            self.correct[cls] += (preds_np[cls_mask] == targets_np[cls_mask]).sum()
            self.total[cls] += cls_mask.sum()

    def compute(self) -> torch.Tensor:
        accuracies = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls in range(self.num_classes):
            if self.total[cls] > 0:
                accuracies[cls] = self.correct[cls] / self.total[cls]
        return accuracies

    def reset(self) -> None:
        self.correct = [0] * self.num_classes
        self.total = [0] * self.num_classes

    def __str__(self) -> str:
        accuracies = self.compute()
        return ", ".join(
            [
                f"Class {self.int_to_stage.get(i, i)}: {acc:.4f}"
                for i, acc in enumerate(accuracies)
            ]
        )
