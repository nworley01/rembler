import os

import h5py
import numpy as np
import pandas as pd


def list_edf_files(directory: str) -> pd.DataFrame:
    """List all EDF files in a directory and return as a DataFrame."""
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    return edf_files


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_to_subject_hdf5(source: str, subject: str, day: str, index: int, signal_type: str, signal: np.ndarray) -> None:
    """Save signals and labels to an subject specific HDF5 file."""
    path = os.path.join(source, subject, "h5")
    with h5py.File(path, 'a') as f:
        f.create_dataset(f'{day}/{index}/{signal_type}', data=signal)
        f.flush()

def load_from_subject_hdf5(source: str, subject: str, day: str, index: int, signal_type: str) -> np.ndarray:
    """Load signals and labels from an subject specific HDF5 file."""
    path = os.path.join(source, subject, "h5")
    with h5py.File(path, 'r') as f:
        signal = f[f'{day}/{index}/{signal_type}'][:]
    return signal

