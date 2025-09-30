import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rembler.data.datasets import SleepStageDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, float]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    logging.info("Saved checkpoint to %s", path)


def compute_class_weights(dataset: SleepStageDataset) -> torch.Tensor:
    freqs = dataset.class_frequencies()
    weights = 1.0 / torch.sqrt(freqs + 1e-12)
    weights = weights / weights.sum() * freqs.numel()
    return weights
