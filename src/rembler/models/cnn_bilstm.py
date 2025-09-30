"""CNN + bidirectional LSTM architecture for sleep stage classification."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CNNBiLSTMConfig:
    in_channels: int
    num_classes: int
    conv_channels: Sequence[int] = (32, 64, 128)
    conv_kernel_size: int = 127
    conv_pool_size: int = 16
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    classifier_hidden_size: int = 128
    classifier_dropout: float = 0.3


def _make_conv_stack(
    in_channels: int,
    channels: Iterable[int],
    kernel_size: int,
    pool_size: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = in_channels
    padding = kernel_size // 2
    for out_channels in channels:
        layers.append(nn.Conv1d(current, out_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.GroupNorm(32, out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool1d(pool_size))
        current = out_channels
    return nn.Sequential(*layers)


class CNNBiLSTM(nn.Module):
    """Stack of 1D convolutions followed by a bidirectional LSTM head."""

    def __init__(self, config: CNNBiLSTMConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            if "in_channels" not in kwargs or "num_classes" not in kwargs:
                raise ValueError("Either provide config or in_channels and num_classes")
            config = CNNBiLSTMConfig(**kwargs)  # type: ignore[arg-type]
        self.config = config

        self.conv = _make_conv_stack(
            in_channels=config.in_channels,
            channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            pool_size=config.conv_pool_size,
        )

        lstm_input_size = config.conv_channels[-1] if len(config.conv_channels) else config.in_channels
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        classifier_in = config.lstm_hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, config.classifier_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        sequence = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(sequence)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)


def build_model(**kwargs) -> CNNBiLSTM:
    """Factory compatible with the training script."""
    if isinstance(kwargs.get("config"), CNNBiLSTMConfig):
        config = kwargs["config"]
        return CNNBiLSTM(config=config)
    return CNNBiLSTM(**kwargs)
