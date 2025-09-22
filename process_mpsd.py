import os

from src.utils import io_utils, sleep_utils

EDF_DIR = "/Volumes/DataCave/rembler_data/raw_edf"
OUT_DIR = "/Volumes/DataCave/rembler_data/processed"

edf_files = io_utils.list_edf_files(EDF_DIR)

for file in edf_files:
    df = sleep_utils.extract_sleep_stages_from_edf(os.path.join(EDF_DIR, file))
    sleep_utils.save_sleep_stages_datatable(df, file.replace(".edf", ".csv"), OUT_DIR)
    print(f"Processed {file} and saved to {OUT_DIR}")

