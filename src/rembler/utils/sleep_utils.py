"""Utility helpers for preparing rodent sleep-stage data from EDF recordings.

This module gathers small convenience routines that are shared across
pre-processing scripts and notebooks.  Functions are intentionally lightweight
and avoid side effects beyond the explicit file I/O helpers so that they remain
easy to test.  Many of the routines operate on `mne.io.Raw` objects and Pandas
``DataFrame`` instances and therefore assume that the caller has already
performed any necessary input validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

int_to_stage = {
    0: "A",
    1: "R",
    2: "S",
    3: "X",
}

stage_to_int = {v: k for k, v in int_to_stage.items()}


def verify_edf(
    edf_file: str | Path,
    duration: int = 24,
    sfreq: int = 500,
) -> bool:
    """Validate that an EDF recording matches basic assumptions.

    Parameters
    ----------
    edf_file:
        Either a filesystem path to an EDF recording or an already-instantiated
        :class:`mne.io.BaseRaw` object.
    duration:
        Expected recording duration in hours.  The check is inclusive, i.e. the
        recording must be at least this long.
    sfreq:
        Expected sampling frequency in Hertz.

    Returns
    -------
    bool
        ``True`` when the recording passes all checks, ``False`` otherwise.

    Notes
    -----
    In addition to ensuring that the file is long enough, this routine checks
    that the canonical EEG/EMG/activity channels are present so downstream code
    can rely on them without additional guards.
    """
    if isinstance(edf_file, (str, Path)):
        edf_file = mne.io.read_raw_edf(edf_file, preload=False)
    elif isinstance(edf_file, mne.io.edf.edf.RawEDF):
        edf_file = edf_file
    required_channels = {"EEG", "EMG", "Signal-Sleep", "Activity"}
    if not required_channels.issubset(set(edf_file.ch_names)):
        print(f"Missing required channels in {edf_file}")
        return False
    if edf_file.info["sfreq"] != sfreq:
        print(
            f"EDF file {edf_file} has sampling frequency {edf_file.info['sfreq']}, expected {sfreq}"
        )
        return False
    if edf_file.n_times < duration * 3600 * edf_file.info["sfreq"]:
        print(f"EDF file {edf_file} is shorter than {duration} hours")
        return False
    return True


def extract_sleep_stages_from_edf(
    edf_path: str,
    signals: list[str],
    bout_length: int = 10,
    bout_context: int = 5,
    causal: bool = False,
) -> pd.DataFrame:
    """Load an EDF recording and return a per-bout sleep-stage table.

    The helper performs a lightweight structural validation, then delegates the
    actual sleep-stage extraction to :func:`create_sleep_stage_dataframe`.
    ``signals`` and buffering parameters are accepted for compatibility with
    older code paths that decorate the returned frame with signal excerpts.

    Parameters
    ----------
    edf_path:
        Filesystem path to the EDF recording.
    signals:
        List of signal names of interest (currently unused but retained for
        compatibility with existing notebooks).
    bout_length:
        Epoch duration in seconds used when slicing the raw signal.
    bout_context:
        Number of neighbouring bouts to include when building contextual signal
        windows.
    causal:
        When ``True`` only preceding context is added to the bout window.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``sleep`` labels and start/stop indices for every bout.
    """
    # Load the EDF file
    edf_data = mne.io.read_raw_edf(edf_path, preload=True, verbose="WARNING")
    # Verify the EDF file
    assert verify_edf(edf_data), f"EDF file {edf_path} failed verification"
    # Extract sleep stages
    return create_sleep_stage_dataframe(edf_data)


def save_sleep_stages_datatable(df: pd.DataFrame, filename: str, out_dir: str) -> str:
    """Persist a sleep-stage table as CSV and return the destination path."""
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    return file_path


def create_sleep_stage_dataframe(edf_data: Any) -> pd.DataFrame:
    """Create a per-bout summary table from an EDF recording.

    Each row corresponds to a 10-second window with start/stop sample indices,
    activity metadata and optional context strings for quick inspection.
    """
    sleep_stages = extract_sleep_stages(edf_data)
    df = (
        pd.Series(sleep_stages)
        .map(int_to_stage)
        .rename("sleep")
        .to_frame()
        .assign(start=lambda df: df.index * 5000)
        .assign(stop=lambda df: (df.index + 1) * 5000)
        .assign(activity=extract_activity_signal(edf_data))
        .assign(context=lambda df: extract_sleep_context(df))
    )
    return df


def subsample_sleep_dataframe(
    df: pd.DataFrame,
    require_full_context: bool = True,
) -> pd.DataFrame:
    """Return a balanced sample of sleep stages by matching REM frequency."""
    df = df.loc[df.context.apply(lambda c: len(c) == 5)] if require_full_context else df
    r_bouts_count = df.sleep.value_counts()["R"]
    return (
        df[df.sleep != "X"]
        .groupby("sleep")
        .sample(n=r_bouts_count, random_state=0)
        .sort_index()
        .reset_index(drop=True)
    )


def decode_sleep_signal(arr: np.ndarray) -> np.ndarray:
    """Convert the scorer's analog signal into coarse integer stages."""
    return np.digitize(arr, bins=[1.5, 2.5, 3.5])


def extract_sleep_stages(
    raw_edf: Any,
    bin_length_in_seconds: int = 10,
    signal_name: str = "Signal-Sleep",
) -> np.ndarray:
    """Extract epoch-level stage labels from the scorer channel."""
    bin_length_in_samples = bin_length_in_seconds * raw_edf.info["sfreq"]
    start = int(bin_length_in_samples / 2)
    step = bin_length_in_samples
    bout_center_points = np.arange(start, raw_edf.n_times, step, dtype=int)
    sleep_signal_bin_center_points = raw_edf[signal_name][0][0, bout_center_points]
    return decode_sleep_signal(sleep_signal_bin_center_points)


def get_bout_signal(full_signals: Any, row: Any, leading_buffer: int = 0, trailing_buffer: int = 0) -> Any:
    """Slice the multichannel signal matrix to the window associated with a row."""
    return full_signals[:, row.start - leading_buffer : row.stop + trailing_buffer]


def extract_activity_signal(edf_data: Any, samples_per_bout: int = 5000) -> pd.Series:
    """Compute a per-bout activity proxy using the rolling max over the channel."""
    activity_channel = edf_data.get_data(picks=["Activity"])[0]
    return (
        pd.Series(activity_channel)
        .rolling(window=samples_per_bout, closed="both", center=True)
        .max()[2500:][::samples_per_bout]
        .reset_index(drop=True)
    )


def extract_sleep_context(df: pd.DataFrame) -> pd.Series:
    """Encode the local neighbourhood of each bout as a short context string."""
    sleep_string = "".join(df.sleep.values)
    return df.index.map(lambda x: sleep_string[max(x - 2, 0) : x + 3])


def determine_buffering(bout_length: int, bout_context: int, sampling_rate: int, causal: bool = False) -> tuple[int, int]:
    """Derive leading/trailing buffer sizes for contextual signal windows."""
    if causal:
        leading_buffer = (bout_context - 1) * bout_length * sampling_rate
        trailing_buffer = 0
    else:
        leading_buffer = int((bout_context - 1) / 2 * bout_length * sampling_rate)
        trailing_buffer = leading_buffer
    assert leading_buffer % 1 == 0, (
        "Leading buffer must be an integer number of samples"
    )
    assert trailing_buffer % 1 == 0, (
        "Trailing buffer must be an integer number of samples"
    )
    return int(leading_buffer), int(trailing_buffer)
