import os
from dataclasses import dataclass

import h5py
import mne
import pandas as pd

from rembler.utils import sleep_utils as su

RAW_EDF_DIR = "/Volumes/DataCave/rembler_data/raw_edf"
OUTPUT_DIR = "/Volumes/DataCave/rembler_data/training_datasets"


@dataclass
class Config:
    bout_length: float
    bout_context: float
    sample_rate: float
    causal: bool
    signals_to_extract: list[str]
    filename: str
    training_dataframe: str


config = Config(
    bout_length=10.0,
    bout_context=5.0,
    sample_rate=500.0,
    causal=False,
    signals_to_extract=["EEG", "EMG"],
    filename="5bout_noncausal_context.h5",
    training_dataframe="data/full_sleep_stage_matched_train_test_split.csv",
)


def generate_training_file(config: Config):
    path = os.path.join(OUTPUT_DIR, config.filename)
    df = pd.read_csv(config.training_dataframe)
    file_components = df.loc[df["role"] == "train"][
        ["subject", "session", "day"]
    ].drop_duplicates(subset=["subject", "session"])

    with h5py.File(path, "a") as f:
        for idx, row in file_components.iterrows():
            if row["session"] == "Baseline":
                edf_filename = os.path.join(
                    RAW_EDF_DIR, f"{row['subject']} {row['session']}.edf"
                )
            else:
                edf_filename = os.path.join(
                    RAW_EDF_DIR, f"{row['subject']} {row['session']} {row['day']}.edf"
                )
            # subset to rows relevant to this file
            df_sub = df.query(
                f"subject == '{row['subject']}' & session == '{row['session']}' & day == '{row['day']}'"
            )
            # read the edf file
            edf_data = mne.io.read_raw_edf(
                edf_filename, preload=True, verbose="WARNING"
            )
            signals = edf_data.get_data(config.signals_to_extract)
            leading_buffer, trailing_buffer = su.determine_buffering(
                config.bout_length,
                config.bout_context,
                config.sample_rate,
                config.causal,
            )
            # loop over each row in the subset dataframe
            for idx, row in df_sub.iterrows():
                # extract the relevant signal segments
                bout_signals = su.get_bout_signal(
                    signals, row, leading_buffer, trailing_buffer
                )
                # save the extracted segments to the HDF5 file
                for i, signal_type in enumerate(config.signals_to_extract):
                    f.create_dataset(
                        f"{row['bout_id']}/{signal_type.lower()}", data=bout_signals[i]
                    )
                    f.flush()


if __name__ == "__main__":
    generate_training_file(config)
