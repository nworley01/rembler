from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    output_dir: Path
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    log_interval: int = 25
    checkpoint_name: str = "best.pt"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name


@dataclass
class ModelConfig:
    input_channels: int = 2
    num_classes: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden_size: int = 32
    num_layers: int = 2
    causal: bool = False
    bidirectional: bool = True


@dataclass
class DatasetConfig:
    train_data: Path
    val_data: Path
    bout_length: int = 10  # in seconds
    sample_rate: int = 500  # in Hz
    bout_context: int = 50  # in seconds, 0 means no context
    augment: bool = False
    shuffle: bool = True
    pin_memory: bool = False


@dataclass
class HardwareConfig:
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)
