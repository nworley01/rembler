"""Implicit convolutional model that operates in the frequency domain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch
from torch import nn


@dataclass
class ImplicitCNNConfig:
    """Configuration for :class:`ImplicitFrequencyCNN`."""

    in_channels: int
    out_channels: int
    hidden_dim: int = 64
    mlp_layers: int = 2
    activation: Type[nn.Module] = nn.GELU
    bias: bool = True


class ImplicitFilterGenerator(nn.Module):
    """Generates length-matched filters with a small MLP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        mlp_layers: int,
        activation: Type[nn.Module],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = 1
        for _ in range(max(mlp_layers, 1)):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, in_channels * out_channels))
        self.mlp = nn.Sequential(*layers)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= 0:
            raise ValueError("Input sequence length must be positive")
        positions = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype).unsqueeze(-1)
        filters = self.mlp(positions)  # (L, out*in)
        filters = filters.view(length, self.out_channels, self.in_channels)
        return filters.permute(1, 2, 0).contiguous()  # (out, in, L)


class ImplicitFrequencyCNN(nn.Module):
    """Applies implicit convolution via frequency-domain multiplication."""

    def __init__(self, config: ImplicitCNNConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            if "in_channels" not in kwargs or "out_channels" not in kwargs:
                raise ValueError("Provide either config or in_channels/out_channels")
            config = ImplicitCNNConfig(**kwargs)  # type: ignore[arg-type]
        self.config = config

        self.filter_generator = ImplicitFilterGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_dim=config.hidden_dim,
            mlp_layers=config.mlp_layers,
            activation=config.activation,
        )
        self.bias = nn.Parameter(torch.zeros(config.out_channels)) if config.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, time), got {tuple(x.shape)}")
        batch, channels, time_steps = x.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, but got {channels}"
            )

        freq_x = torch.fft.rfft(x, dim=-1)

        filters_time = self.filter_generator(length=time_steps, device=x.device, dtype=x.dtype)
        freq_filters = torch.fft.rfft(filters_time, dim=-1)

        freq_x = freq_x.unsqueeze(1)
        freq_filters = freq_filters.unsqueeze(0)
        conv_freq = (freq_x * freq_filters).sum(dim=2)

        out = torch.fft.irfft(conv_freq, n=time_steps, dim=-1)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out


def build_model(**kwargs: Any) -> ImplicitFrequencyCNN:
    config = kwargs.get("config")
    if isinstance(config, ImplicitCNNConfig):
        return ImplicitFrequencyCNN(config=config)
    return ImplicitFrequencyCNN(**kwargs)
