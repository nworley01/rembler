import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd

from rembler.utils.io_utils import (
    list_files_with_extension,
    ensure_dir,
    save_to_subject_hdf5,
    load_from_subject_hdf5,
    aggregate_csvs
)


@pytest.mark.unit
@patch('os.listdir')
def test_list_files_with_extension_success(mock_listdir):
    """Test successful file listing with specific extension."""
    mock_listdir.return_value = [
        "data1.csv", "data2.CSV", "data3.txt", "data4.csv", "readme.md"
    ]

    result = list_files_with_extension("/test/dir", ".csv")

    expected_df = pd.DataFrame(["data1.csv", "data2.CSV", "data4.csv"], columns=["filename"])
    pd.testing.assert_frame_equal(result, expected_df)
    mock_listdir.assert_called_once_with("/test/dir")


@pytest.mark.unit
@patch('os.listdir')
def test_list_files_with_extension_case_insensitive(mock_listdir):
    """Test that file extension matching is case insensitive."""
    mock_listdir.return_value = [
        "file1.TXT", "file2.txt", "file3.Txt", "file4.dat"
    ]

    result = list_files_with_extension("/test/dir", ".txt")

    expected_df = pd.DataFrame(["file1.TXT", "file2.txt", "file3.Txt"], columns=["filename"])
    pd.testing.assert_frame_equal(result, expected_df)


@pytest.mark.unit
@patch('os.listdir')
def test_list_files_with_extension_no_matches(mock_listdir):
    """Test behavior when no files match the extension."""
    mock_listdir.return_value = ["file1.txt", "file2.dat", "file3.log"]

    result = list_files_with_extension("/test/dir", ".csv")

    expected_df = pd.DataFrame([], columns=["filename"])
    pd.testing.assert_frame_equal(result, expected_df)


@pytest.mark.unit
@patch('os.listdir')
def test_list_files_with_extension_empty_directory(mock_listdir):
    """Test behavior with empty directory."""
    mock_listdir.return_value = []

    result = list_files_with_extension("/test/dir", ".csv")

    expected_df = pd.DataFrame([], columns=["filename"])
    pd.testing.assert_frame_equal(result, expected_df)


@pytest.mark.unit
@patch('os.listdir', side_effect=FileNotFoundError("Directory not found"))
def test_list_files_with_extension_directory_not_found(mock_listdir):
    """Test handling of non-existent directory."""
    with pytest.raises(FileNotFoundError):
        list_files_with_extension("/nonexistent/dir", ".csv")


@pytest.mark.unit
@patch('os.makedirs')
def test_ensure_dir_creates_directory(mock_makedirs):
    """Test that ensure_dir creates directory."""
    ensure_dir("/test/new/dir")

    mock_makedirs.assert_called_once_with("/test/new/dir", exist_ok=True)


@pytest.mark.unit
@patch('os.makedirs')
def test_ensure_dir_handles_existing_directory(mock_makedirs):
    """Test that ensure_dir handles existing directory gracefully."""
    # exist_ok=True should prevent errors for existing directories
    ensure_dir("/existing/dir")

    mock_makedirs.assert_called_once_with("/existing/dir", exist_ok=True)


@pytest.mark.unit
@patch('os.makedirs', side_effect=PermissionError("Permission denied"))
def test_ensure_dir_permission_error(mock_makedirs):
    """Test handling of permission errors."""
    with pytest.raises(PermissionError):
        ensure_dir("/protected/dir")


@pytest.mark.unit
@patch('h5py.File')
@patch('os.path.join')
def test_save_to_subject_hdf5_success(mock_join, mock_h5py_file):
    """Test successful saving to HDF5 file."""
    mock_join.return_value = "/source/subject/h5"
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    test_signal = np.array([1, 2, 3, 4, 5])
    save_to_subject_hdf5("source", "subject", "day1", 0, "eeg", test_signal)

    mock_join.assert_called_once_with("source", "subject", "h5")
    mock_h5py_file.assert_called_once_with("/source/subject/h5", "a")
    mock_file.create_dataset.assert_called_once_with("day1/0/eeg", data=test_signal)
    mock_file.flush.assert_called_once()


@pytest.mark.unit
@patch('h5py.File')
@patch('os.path.join')
def test_save_to_subject_hdf5_different_signal_types(mock_join, mock_h5py_file):
    """Test saving different signal types."""
    mock_join.return_value = "/source/subject/h5"
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    eeg_signal = np.array([1.0, 2.0, 3.0])
    emg_signal = np.array([0.1, 0.2, 0.3])

    save_to_subject_hdf5("source", "subject", "day1", 0, "eeg", eeg_signal)
    save_to_subject_hdf5("source", "subject", "day1", 0, "emg", emg_signal)

    assert mock_file.create_dataset.call_count == 2
    mock_file.create_dataset.assert_any_call("day1/0/eeg", data=eeg_signal)
    mock_file.create_dataset.assert_any_call("day1/0/emg", data=emg_signal)


@pytest.mark.unit
@patch('h5py.File')
@patch('os.path.join')
def test_load_from_subject_hdf5_success(mock_join, mock_h5py_file):
    """Test successful loading from HDF5 file."""
    mock_join.return_value = "/source/subject/h5"
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    expected_signal = np.array([1, 2, 3, 4, 5])
    mock_file.__getitem__.return_value.__getitem__.return_value = expected_signal

    result = load_from_subject_hdf5("source", "subject", "day1", 0, "eeg")

    mock_join.assert_called_once_with("source", "subject", "h5")
    mock_h5py_file.assert_called_once_with("/source/subject/h5", "r")
    mock_file.__getitem__.assert_called_once_with("day1/0/eeg")
    np.testing.assert_array_equal(result, expected_signal)


@pytest.mark.unit
@patch('h5py.File')
@patch('os.path.join')
def test_load_from_subject_hdf5_key_error(mock_join, mock_h5py_file):
    """Test handling of missing dataset in HDF5 file."""
    mock_join.return_value = "/source/subject/h5"
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file
    mock_file.__getitem__.side_effect = KeyError("Dataset not found")

    with pytest.raises(KeyError):
        load_from_subject_hdf5("source", "subject", "day1", 0, "nonexistent")


@pytest.mark.unit
@patch('pandas.read_csv')
@patch('os.path.join')
@patch('os.getcwd')
def test_aggregate_csvs_success(mock_getcwd, mock_join, mock_read_csv):
    """Test successful CSV aggregation."""
    mock_getcwd.return_value = "/current/dir"
    mock_join.side_effect = lambda d, f: f"{d}/{f}"

    # Mock CSV data
    df1 = pd.DataFrame({"value": [1, 2], "metric": ["a", "b"]})
    df2 = pd.DataFrame({"value": [3, 4], "metric": ["c", "d"]})
    mock_read_csv.side_effect = [df1, df2]

    files = ["subject1 session1 day1.csv", "subject2 session2.csv"]
    result = aggregate_csvs(files)

    expected_df = pd.DataFrame({
        "value": [1, 2, 3, 4],
        "metric": ["a", "b", "c", "d"],
        "subject": ["subject1", "subject1", "subject2", "subject2"],
        "session": ["session1", "session1", "session2", "session2"],
        "day": [1, 1, 1, 1]
    })

    pd.testing.assert_frame_equal(result.sort_index(axis=1), expected_df.sort_index(axis=1))


@pytest.mark.unit
@patch('pandas.read_csv')
@patch('os.path.join')
def test_aggregate_csvs_with_custom_directory(mock_join, mock_read_csv):
    """Test CSV aggregation with custom directory."""
    mock_join.side_effect = lambda d, f: f"{d}/{f}"

    df1 = pd.DataFrame({"value": [1, 2]})
    mock_read_csv.return_value = df1

    files = ["test file.csv"]
    result = aggregate_csvs(files, "/custom/dir")

    mock_join.assert_called_with("/custom/dir", "test file.csv")


@pytest.mark.unit
@patch('pandas.read_csv')
@patch('os.path.join')
@patch('os.getcwd')
def test_aggregate_csvs_three_part_filename(mock_getcwd, mock_join, mock_read_csv):
    """Test CSV aggregation with three-part filename including day."""
    mock_getcwd.return_value = "/current/dir"
    mock_join.side_effect = lambda d, f: f"{d}/{f}"

    df1 = pd.DataFrame({"value": [1, 2]})
    mock_read_csv.return_value = df1

    files = ["subject1 session1 day3.csv"]
    result = aggregate_csvs(files)

    assert result["day"].iloc[0] == "day3"
    assert result["subject"].iloc[0] == "subject1"
    assert result["session"].iloc[0] == "session1"


@pytest.mark.unit
def test_aggregate_csvs_invalid_filename():
    """Test handling of files without .csv extension."""
    files = ["invalid_file.txt"]

    with pytest.raises(AssertionError):
        aggregate_csvs(files)


@pytest.mark.unit
@patch('pandas.read_csv', side_effect=FileNotFoundError("File not found"))
@patch('os.path.join')
@patch('os.getcwd')
def test_aggregate_csvs_file_not_found(mock_getcwd, mock_join, mock_read_csv):
    """Test handling of missing CSV files."""
    mock_getcwd.return_value = "/current/dir"
    mock_join.side_effect = lambda d, f: f"{d}/{f}"

    files = ["nonexistent.csv"]

    with pytest.raises(FileNotFoundError):
        aggregate_csvs(files)


@pytest.mark.integration
def test_list_files_with_extension_integration():
    """Integration test with real temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = ["data1.csv", "data2.CSV", "data3.txt", "data4.csv"]
        for filename in test_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test content")

        result = list_files_with_extension(temp_dir, ".csv")

        expected_files = ["data1.csv", "data2.CSV", "data4.csv"]
        assert len(result) == 3
        assert set(result["filename"].tolist()) == set(expected_files)


@pytest.mark.integration
def test_ensure_dir_integration():
    """Integration test with real directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        new_dir = os.path.join(temp_dir, "new", "nested", "directory")

        ensure_dir(new_dir)

        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)


@pytest.mark.integration
def test_hdf5_save_load_integration():
    """Integration test for HDF5 save and load operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_signal = np.random.rand(1000).astype(np.float32)
        source = temp_dir
        subject = "test_subject"
        day = "day1"
        index = 0
        signal_type = "eeg"

        # Create subject directory
        subject_dir = os.path.join(source, subject)
        os.makedirs(subject_dir, exist_ok=True)

        # Save data
        save_to_subject_hdf5(source, subject, day, index, signal_type, test_signal)

        # Load data
        loaded_signal = load_from_subject_hdf5(source, subject, day, index, signal_type)

        # Verify
        np.testing.assert_array_equal(test_signal, loaded_signal)


@pytest.mark.integration
def test_aggregate_csvs_integration():
    """Integration test with real CSV files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV files
        df1 = pd.DataFrame({"value": [1, 2], "metric": ["a", "b"]})
        df2 = pd.DataFrame({"value": [3, 4], "metric": ["c", "d"]})

        file1 = "subject1 session1 day1.csv"
        file2 = "subject2 session2.csv"

        df1.to_csv(os.path.join(temp_dir, file1), index=False)
        df2.to_csv(os.path.join(temp_dir, file2), index=False)

        # Test aggregation
        result = aggregate_csvs([file1, file2], temp_dir)

        assert len(result) == 4
        assert "subject" in result.columns
        assert "session" in result.columns
        assert "day" in result.columns
        assert set(result["subject"].unique()) == {"subject1", "subject2"}
        assert set(result["session"].unique()) == {"session1", "session2"}