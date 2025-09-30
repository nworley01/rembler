import torch
from torch import nn


class SmallCNN(nn.Module):
    """Fallback model for quick experimentation."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=128, padding="valid"),
            nn.GroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(32),
            nn.Conv1d(32, 64, kernel_size=64, padding="valid"),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(16),
            nn.Conv1d(64, 128, kernel_size=32, padding="valid"),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        return self.classifier(x)
