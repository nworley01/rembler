import argparse
from pathlib import Path

from rembler.configs.configs import TrainConfig


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
