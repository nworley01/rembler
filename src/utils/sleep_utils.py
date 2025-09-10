import mne
import numpy as np
import pandas as pd

int_to_stage = {0: "A",
                1: "R",
                2: "S",
                3: "X",
               }

def process_edf(edf_path: str, signals: list = ["EEG", "EMG"], bout_length: int = 10, bout_context: int = 5, causal=False,) -> pd.DataFrame:
    """
    Process the EDF file and extract relevant data
    """
    # Load the EDF file
    edf_data = mne.io.read_raw_edf(edf_path, preload=True)
    signals = edf_data.get_data(picks=["EEG", "EMG"])
    # Extract sleep stages
    sleep_stages = create_sleep_stage_dataframe(edf_data)
    sleep_stages_subsample = subsample_sleep_dataframe(sleep_stages)
    leading_buffer, trailing_buffer = determine_buffering(bout_length, bout_context, 500, causal)
    sleep_stages_subsample = sleep_stages_subsample.assign(signal=sleep_stages_subsample.apply(lambda row: get_bout_signal(signals,
                                                                                 row,
                                                                                 leading_buffer=leading_buffer,
                                                                                 trailing_buffer=trailing_buffer
                                                                                 ), axis=1))

    return sleep_stages_subsample
    

def create_sleep_stage_dataframe(edf_data: str) -> pd.DataFrame:
    """
    Create a sleep stage DataFrame from a raw EDF file.
    Each row corresponds to a 10-second bout with start and stop indices.
    Arguments:
    - edf_data: MNE Raw object containing the EDF data.
    Returns:
    - A pandas DataFrame with columns: 'sleep', 'start', 'stop'.
    """
    sleep_stages = extract_sleep_stages(edf_data)
    df = (pd.Series(sleep_stages)
    .map(int_to_stage)
    .rename("sleep")
    .to_frame()
    .assign(start = lambda df: df.index * 5000)
    .assign(stop = lambda df: (df.index +1 )* 5000)
    )
    return df

def subsample_sleep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subsamples the sleep dataframe to have balanced classes by 
    downsampling to match "R" counts.
    """
    r_bouts_count = df.sleep.value_counts()["R"]
    return (df[df.sleep != "X"]
            .groupby("sleep")
            .sample(n=r_bouts_count, random_state=0)
            .sort_index()
            )

def decode_sleep_signal(arr: np.ndarray) -> np.ndarray:
    """
    Converts sleep signal into integer encodings. 
    ~ 10 is Artifact 
    ~ 3 is Sleep
    ~ 2 is REM
    ~ 1 Awake
    """
    return np.digitize(arr, bins=[1.5, 2.5, 3.5])

def extract_sleep_stages(raw_edf, bin_length_in_seconds=10, signal_name="Signal-Sleep",) -> np.ndarray:
    """
    Extracts sleep stages from raw EDF data in bins of specified length.
    Arguments:
    - raw_edf: MNE Raw object containing the EDF data.
    - bin_length_in_seconds: Length of each bin in seconds (default is 10 seconds).
    - signal_name: Name of the sleep signal channel in the EDF data (default is "Signal-Sleep").
    Returns:
    - A numpy array of sleep stages corresponding to each bin.
    """
    bin_length_in_samples = bin_length_in_seconds * raw_edf.info["sfreq"]
    start = int(bin_length_in_samples / 2)
    step = bin_length_in_samples
    bout_center_points = np.arange( start, raw_edf.n_times, step, dtype=int)
    sleep_signal_bin_center_points = raw_edf[signal_name][0][0, bout_center_points]
    return decode_sleep_signal(sleep_signal_bin_center_points)

def get_bout_signal(full_signals, row, leading_buffer=0, trailing_buffer=0):
    return full_signals[:, row.start-leading_buffer:row.stop+trailing_buffer]

def determine_buffering(bout_length, bout_context, sampling_rate, causal=False):
    if causal:
        leading_buffer = (bout_context - 1) * bout_length * sampling_rate
        trailing_buffer = 0
    else:
        leading_buffer = (bout_context - 1) / 2 * bout_length * sampling_rate
        trailing_buffer = leading_buffer
    assert leading_buffer % 1 == 0, "Leading buffer must be an integer number of samples"
    assert trailing_buffer % 1 == 0, "Trailing buffer must be an integer number of samples"
    return int(leading_buffer), int(trailing_buffer)