from rembler.utils.dataset_utils import split_train_test
from rembler.utils.io_utils import aggregate_csvs, list_files_with_extension
from rembler.utils.sleep_utils import subsample_sleep_dataframe

# generate a list of all CSV files in the processed data directory
csv_files = list_files_with_extension(
    "/Volumes/DataCave/rembler_data/processed", ".csv"
)

# aggregate all CSV files into a single DataFrame
full = aggregate_csvs(
    csv_files.filename,
    dir="/Volumes/DataCave/rembler_data/processed",
)

# downsample and match minority class (sleep) across subjects and sessions
stage_matched = (
    full.query(
        "session != 'SD' and subject != 'MPSD2'"
    )  # exclude sleep deprivation sessions and problematic subject
    .groupby(["subject", "session"])
    .apply(subsample_sleep_dataframe, include_groups=True)
).reset_index(drop=True)

final = (
    stage_matched.groupby(
        [
            "subject",
            "session",
        ]
    )
    .apply(split_train_test, include_groups=True)
    .reset_index(drop=True)
)

final["bout_id"] = final.index

final.to_csv("data/full_sleep_stage_matched_train_test_split.csv", index=False)
