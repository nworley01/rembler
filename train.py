"""PyTorch training loop for sleep stage classification."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from rembler.data import dataset_utils as du
from rembler.data import sleep_utils as su
from rembler.data.datasets import SleepStageDataset
from rembler.evaluation.metrics.custom_metrics import PerClassMetric
from rembler.training.hardware_utils import resolve_device


@dataclass
class TrainConfig:
    train_data: Path
    val_data: Path
    output_dir: Path
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    log_interval: int = 48
    checkpoint_name: str = "best.pt"
    model_type: str = "small_cnn"  # "small_cnn", "cnn_bilstm", "implicit_cnn"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name


def build_model(model_type: str, in_channels: int, num_classes: int) -> nn.Module:
    if model_type == "small_cnn":
        from rembler.models.basic_cnn import SmallCNN

        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_type == "simple_dense":
        from rembler.models.basic_dense import SimpleDense

        return SimpleDense(
            in_channels=in_channels, num_classes=num_classes, sequence_length=25000
        )
    elif model_type == "cnn_bilstm":
        from rembler.models.cnn_bilstm import CNNBiLSTM

        return CNNBiLSTM(in_channels=in_channels, num_classes=num_classes)
    elif model_type == "implicit_cnn":
        from rembler.models.implicit_cnn import ImplicitFrequencyCNN

        return ImplicitFrequencyCNN(in_channels=in_channels, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, batch in enumerate(dataloader, start=1):
        signals = batch["data"].to(device)
        labels = batch["label"].to(device)
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


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_metric = PerClassMetric(num_classes=dataloader.dataset.num_classes)
    with torch.no_grad():
        for _step, batch in enumerate(dataloader):
            signals = batch["data"].to(device)
            labels = batch["label"].to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            per_class_metric.update(preds, labels)
    metrics_dict = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }
    for metric, value in enumerate(per_class_metric.compute()):
        if isinstance(value, torch.Tensor):
            metrics_dict[f"{su.int_to_stage.get(metric, metric)}_accuracy"] = (
                value.item()
            )

    return metrics_dict


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, float]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    logging.info("Saved checkpoint to %s", path)


def log_metrics(epoch: int, phase: str, metrics: dict[str, float]) -> None:
    formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logging.info("%s epoch=%d %s", phase, epoch, formatted)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a sleep stage classifier")
    parser.add_argument(
        "--train-data", type=Path, required=False, help="Path to training npz file"
    )
    parser.add_argument(
        "--val-data", type=Path, required=False, help="Path to validation npz file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=TrainConfig.learning_rate
    )
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument(
        "--device",
        type=str,
        default=TrainConfig.device,
        help="'auto', 'cpu', 'cuda', ...",
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument(
        "--checkpoint-name", type=str, default=TrainConfig.checkpoint_name
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=TrainConfig.model_type,
        choices=["small_cnn", "cnn_bilstm", "implicit_cnn", "simple_dense"],
    )
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
        model_type=args.model_type,
    )


def main() -> None:
    config = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info(
        "Config: %s", json.dumps({k: str(v) for k, v in asdict(config).items()})
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = resolve_device(config.device)
    logging.info("Using device: %s", device)

    df = pd.read_csv("data/full_sleep_stage_matched_train_test_split.csv")
    train_ds = du.CustomDataset(
        training_dataframe=df.query("role == 'train'"),
        hdf5_path="/Volumes/DataCave/rembler_data/training_datasets/5bout_noncausal_context.h5",
        signal_names=["eeg", "emg"],
    )
    val_ds = du.CustomDataset(
        training_dataframe=df.query("role == 'test'"),
        hdf5_path="/Volumes/DataCave/rembler_data/training_datasets/5bout_noncausal_context.h5",
        signal_names=["eeg", "emg"],
    )
    model = build_model(config.model_type, train_ds.num_channels, 3).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.grad_clip,
            config.log_interval,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        log_metrics(epoch, "train", train_metrics)
        log_metrics(epoch, "val", val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(
                config.checkpoint_path,
                model,
                {"val_accuracy": best_val_acc, "epoch": epoch},
            )

    with (config.output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    logging.info("Training complete. Best val accuracy=%.4f", best_val_acc)


if __name__ == "__main__":
    main()
