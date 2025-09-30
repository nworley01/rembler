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
        "session != 'SD' and subject == 'MPSD2'"
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
    .apply(lambda df: df, include_groups=True)  # apply no splitting to keep all data
    .reset_index(drop=True)
)

final["role"] = "test"
final["bout_id"] = final.index

final.to_csv("data/single_subject_sleep_stage_matched_test_set.csv", index=False)
