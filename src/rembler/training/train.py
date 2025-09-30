import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from rembler.data import sleep_utils as su
from rembler.evaluation.metrics.custom_metrics import PerClassMetric


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
