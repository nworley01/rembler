from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SleepStageDataset(Dataset):
    """Dataset wrapper expecting an .npz with 'signals' and 'labels' arrays."""

    def __init__(self, npz_path: Path, dtype: torch.dtype = torch.float32) -> None:
        self.path = npz_path
        with np.load(npz_path) as arrays:
            if "signals" not in arrays or "labels" not in arrays:
                raise KeyError(f"{npz_path} must contain 'signals' and 'labels' entries")
            signals = arrays["signals"]
            labels = arrays["labels"]
        if signals.ndim != 3:
            raise ValueError(f"Expected signals shape (N, C, T), got {signals.shape}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels shape (N,), got {labels.shape}")
        if signals.shape[0] != labels.shape[0]:
            raise ValueError("Signals and labels must have the same number of samples")
        self.signals = torch.as_tensor(signals, dtype=dtype)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self._num_classes = int(self.labels.max().item()) + 1
        self._num_channels = int(self.signals.shape[1])

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[index], self.labels[index]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def class_frequencies(self) -> torch.Tensor:
        counts = torch.bincount(self.labels, minlength=self._num_classes)
        return counts.to(torch.float32) / counts.sum()
