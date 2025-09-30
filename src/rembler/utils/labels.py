from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Canonical class map
CLASS_MAP = {
    "WAKE": 0,
    "W": 0,
    "WA": 0,
    "NREM": 1,
    "N": 1,
    "SWS": 1,
    "N2": 1,
    "N3": 1,  # rodents often just SWS/NREM
    "REM": 2,
    "R": 2,
    "P": 2,  # P for paradoxical
}
UNKNOWN = -1


def normalize_stage(label: str) -> int:
    if label is None:
        return UNKNOWN
    key = re.sub(r"\s+", "", label.upper())
    return CLASS_MAP.get(key, UNKNOWN)


def labels_from_annotations(
    annotations: list[tuple[float, float, str]], total_sec: float, epoch_sec: int
) -> np.ndarray:
    """Rasterize EDF annotations into per-epoch labels (UNKNOWN if uncovered)."""
    n_epochs = int(np.floor(total_sec / epoch_sec))
    y = np.full(n_epochs, UNKNOWN, dtype=np.int16)
    for onset, dur, desc in annotations:
        cls = normalize_stage(desc)
        if cls == UNKNOWN:
            continue
        start_ep = max(0, int(np.floor(onset / epoch_sec)))
        end_ep = min(n_epochs, int(np.ceil((onset + dur) / epoch_sec)))
        y[start_ep:end_ep] = cls
    return y


def labels_from_csv(
    csv_path: str, epoch_sec: int, total_sec: float | None = None
) -> np.ndarray:
    """
    CSV schema options:
      A) per-epoch: columns ['epoch', 'stage']  (0-based epoch index)
      B) intervals: columns ['onset_sec','duration_sec','stage']
    """
    df = pd.read_csv(csv_path)
    if {"epoch", "stage"}.issubset(df.columns):
        n_epochs = (
            int(df["epoch"].max() + 1)
            if total_sec is None
            else int(np.floor(total_sec / epoch_sec))
        )
        y = np.full(n_epochs, UNKNOWN, dtype=np.int16)
        for _, r in df.iterrows():
            y[int(r["epoch"])] = normalize_stage(str(r["stage"]))
        return y
    elif {"onset_sec", "duration_sec", "stage"}.issubset(df.columns):
        assert total_sec is not None, "total_sec required for interval label CSV"
        ann = list(
            df[["onset_sec", "duration_sec", "stage"]].itertuples(
                index=False, name=None
            )
        )
        return labels_from_annotations(ann, total_sec, epoch_sec)
    else:
        raise ValueError("Unrecognized CSV label schema.")
