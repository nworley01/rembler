import pandas as pd
import numpy as np

int_to_stage = {0: "A",
                1: "R",
                2: "S",
                3: "X",
               }


def decode_sleep_signal(arr):
    """
    Converts sleep signal into integer encodings. 
    
    
    
    ~ 10 is Artifact 
    ~ 3 is Sleep
    ~ 2 is REM
    ~ 1 Awake
    """
    return np.digitize(arr, bins=[1.5, 2.5, 3.5])

def extract_sleep_stages(raw_edf, bin_length_seconds=10, signal_name="Signal-Sleep"):
    """
    """
    bin_length_samples = bin_length_seconds * raw_edf.info["sfreq"]
    start = int(bin_length_samples / 2)
    step = bin_length_samples
    center_points = np.arange( start, raw_edf.n_times, step, dtype=int)
    sleep_signal_bin_center_points = raw_edf[signal_name][0][0, center_points]
    return decode_sleep_signal(sleep_signal_bin_center_points)