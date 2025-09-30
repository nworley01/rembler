import logging


def log_metrics(epoch: int, phase: str, metrics: dict[str, float]) -> None:
    formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logging.info("%s epoch=%d %s", phase, epoch, formatted)
