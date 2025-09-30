"""PyTorch training loop for sleep stage classification."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import hydra
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from rembler.configs.configs import TrainConfig
from rembler.data import dataset_utils as du
from rembler.interface.logger import log_metrics
from rembler.models.build import build_model
from rembler.training.hardware_utils import resolve_device
from rembler.training.train import evaluate, train_one_epoch
from rembler.training.train_utils import (
    compute_class_weights,
    save_checkpoint,
    set_seed,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    config = yaml.safe_load(OmegaConf.to_yaml(cfg))
    config = TrainConfig(**config["train_spec"])
    # config = parse_args()
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
