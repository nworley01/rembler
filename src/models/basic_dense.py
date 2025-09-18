import torch
from torch import nn


class SimpleDense(nn.Module):
    """Simple dense model for quick experimentation."""

    def __init__(self, in_channels: int, num_classes: int, sequence_length: int = 25000) -> None:
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length*in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.classifier(x)