"""PyTorch training loop for sleep stage classification."""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    train_data: Path
    val_data: Path
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
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


class SmallCNN(nn.Module):
    """Fallback model for quick experimentation."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        return self.classifier(x)


def build_model(in_channels: int, num_classes: int) -> nn.Module:
    try:
        from src.models.cnn_bilstm import build_model as build_external  # type: ignore
    except ImportError:
        build_external = None
    if callable(build_external):
        return build_external(in_channels=in_channels, num_classes=num_classes)
    try:
        from src.models.cnn_bilstm import CNNBiLSTM  # type: ignore

        return CNNBiLSTM(in_channels=in_channels, num_classes=num_classes)
    except (ImportError, AttributeError):
        logging.warning("Falling back to SmallCNN; implement src/models/cnn_bilstm.py for custom model.")
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: SleepStageDataset) -> torch.Tensor:
    freqs = dataset.class_frequencies()
    weights = 1.0 / torch.sqrt(freqs + 1e-12)
    weights = weights / weights.sum() * freqs.numel()
    return weights


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, (signals, labels) in enumerate(dataloader, start=1):
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        if log_interval and step % log_interval == 0:
            logging.info("train step=%d loss=%.4f", step, loss.item())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def save_checkpoint(path: Path, model: nn.Module, metadata: Dict[str, float]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    logging.info("Saved checkpoint to %s", path)


def log_metrics(epoch: int, phase: str, metrics: Dict[str, float]) -> None:
    formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logging.info("%s epoch=%d %s", phase, epoch, formatted)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a sleep stage classifier")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to training npz file")
    parser.add_argument("--val-data", type=Path, required=True, help="Path to validation npz file")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory to write checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device, help="'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--checkpoint-name", type=str, default=TrainConfig.checkpoint_name)
    args = parser.parse_args()
    return TrainConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=None if args.grad_clip <= 0 else args.grad_clip,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
        checkpoint_name=args.checkpoint_name,
    )


def main() -> None:
    config = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Config: %s", json.dumps({k: str(v) for k, v in asdict(config).items()}))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = config.resolve_device()
    logging.info("Using device: %s", device)

    train_ds = SleepStageDataset(config.train_data)
    val_ds = SleepStageDataset(config.val_data)
    model = build_model(train_ds.num_channels, train_ds.num_classes).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config.grad_clip, config.log_interval)
        val_metrics = evaluate(model, val_loader, criterion, device)
        log_metrics(epoch, "train", train_metrics)
        log_metrics(epoch, "val", val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(config.checkpoint_path, model, {"val_accuracy": best_val_acc, "epoch": epoch})

    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    logging.info("Training complete. Best val accuracy=%.4f", best_val_acc)


if __name__ == "__main__":
    main()
