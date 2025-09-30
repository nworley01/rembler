from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd


def list_files_with_extension(directory: str, ext: str) -> pd.DataFrame:
    """List all files in a directory and return as a DataFrame."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(ext)]
    return pd.DataFrame(files, columns=["filename"])


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def save_to_subject_hdf5(
    source: str,
    subject: str,
    day: str,
    index: int,
    signal_type: str,
    signal: np.ndarray,
) -> None:
    """Save signals and labels to an subject specific HDF5 file."""
    path = os.path.join(source, subject, "h5")
    with h5py.File(path, "a") as f:
        f.create_dataset(f"{day}/{index}/{signal_type}", data=signal)
        f.flush()


def load_from_subject_hdf5(
    source: str, subject: str, day: str, index: int, signal_type: str
) -> np.ndarray:
    """Load signals and labels from an subject specific HDF5 file."""
    path = os.path.join(source, subject, "h5")
    with h5py.File(path, "r") as f:
        signal = f[f"{day}/{index}/{signal_type}"][:]
    return signal


def aggregate_csvs(
    files: list[str],
    dir: str | None = None,
) -> pd.DataFrame:
    """Aggregate multiple CSV files into a single DataFrame."""
    if dir is None:
        dir = os.getcwd()
    dfs = []
    for file in files:
        subject_session_day = file.replace(".csv", "").split(" ")
        assert file.endswith(".csv")  # sanity check
        df = pd.read_csv(os.path.join(dir, file))
        df["subject"] = subject_session_day[0]
        df["session"] = subject_session_day[1]
        if len(subject_session_day) > 2:
            df["day"] = subject_session_day[2]
        else:
            df["day"] = 1
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
