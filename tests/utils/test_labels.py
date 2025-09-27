import tempfile
import os
from unittest.mock import patch, mock_open
import pytest
import numpy as np
import pandas as pd

from rembler.utils.labels import (
    normalize_stage,
    labels_from_annotations,
    labels_from_csv,
    CLASS_MAP,
    UNKNOWN
)


@pytest.mark.unit
def test_normalize_stage_wake_variations():
    """Test normalization of various wake stage labels."""
    wake_labels = ["WAKE", "W", "WA", "wake", "w", "wa", " WAKE ", "Wake"]

    for label in wake_labels:
        result = normalize_stage(label)
        assert result == 0, f"Failed for label: {label}"


@pytest.mark.unit
def test_normalize_stage_nrem_variations():
    """Test normalization of various NREM stage labels."""
    nrem_labels = ["NREM", "N", "SWS", "N2", "N3", "nrem", "n", "sws", " NREM "]

    for label in nrem_labels:
        result = normalize_stage(label)
        assert result == 1, f"Failed for label: {label}"


@pytest.mark.unit
def test_normalize_stage_rem_variations():
    """Test normalization of various REM stage labels."""
    rem_labels = ["REM", "R", "P", "rem", "r", "p", " REM ", "Rem"]

    for label in rem_labels:
        result = normalize_stage(label)
        assert result == 2, f"Failed for label: {label}"


@pytest.mark.unit
def test_normalize_stage_unknown_labels():
    """Test handling of unknown or invalid stage labels."""
    unknown_labels = ["UNKNOWN", "X", "ARTIFACT", "invalid", "123", ""]

    for label in unknown_labels:
        result = normalize_stage(label)
        assert result == UNKNOWN, f"Failed for label: {label}"


@pytest.mark.unit
def test_normalize_stage_none_input():
    """Test handling of None input."""
    result = normalize_stage(None)
    assert result == UNKNOWN


@pytest.mark.unit
def test_normalize_stage_whitespace_handling():
    """Test proper whitespace handling in labels."""
    labels_with_whitespace = [
        "W A K E",  # spaces within
        " W ",      # leading/trailing spaces
        "\tREM\n",  # tabs and newlines
        "N R E M"   # spaces within
    ]

    expected_results = [UNKNOWN, 0, 2, UNKNOWN]  # "W A K E" and "N R E M" become unknown after space removal

    for label, expected in zip(labels_with_whitespace, expected_results):
        result = normalize_stage(label)
        assert result == expected, f"Failed for label: '{label}'"


@pytest.mark.unit
def test_class_map_constants():
    """Test that CLASS_MAP contains expected mappings."""
    expected_mappings = {
        "WAKE": 0, "W": 0, "WA": 0,
        "NREM": 1, "N": 1, "SWS": 1, "N2": 1, "N3": 1,
        "REM": 2, "R": 2, "P": 2,
    }

    for key, expected_value in expected_mappings.items():
        assert CLASS_MAP[key] == expected_value


@pytest.mark.unit
def test_unknown_constant():
    """Test UNKNOWN constant value."""
    assert UNKNOWN == -1


@pytest.mark.unit
def test_labels_from_annotations_basic():
    """Test basic annotation to labels conversion."""
    annotations = [
        (0.0, 10.0, "WAKE"),    # 0-10s: WAKE
        (10.0, 20.0, "NREM"),   # 10-30s: NREM
        (30.0, 10.0, "REM"),    # 30-40s: REM
    ]
    total_sec = 50.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([0, 1, -1, 2, -1], dtype=np.int16)  # epoch 2 uncovered, epoch 4 uncovered
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_labels_from_annotations_overlapping():
    """Test handling of overlapping annotations (later ones overwrite)."""
    annotations = [
        (0.0, 20.0, "WAKE"),    # 0-20s: WAKE
        (10.0, 10.0, "REM"),    # 10-20s: REM (overwrites part of WAKE)
    ]
    total_sec = 30.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([0, 2, -1], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_labels_from_annotations_partial_epochs():
    """Test handling of annotations that partially cover epochs."""
    annotations = [
        (5.0, 10.0, "WAKE"),    # 5-15s: WAKE (covers epoch 0 partially, epoch 1 fully)
        (25.0, 5.0, "REM"),     # 25-30s: REM (covers epoch 2 partially)
    ]
    total_sec = 40.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([0, 0, 2, -1], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_labels_from_annotations_unknown_stages():
    """Test handling of unknown stage labels in annotations."""
    annotations = [
        (0.0, 10.0, "WAKE"),
        (10.0, 10.0, "UNKNOWN"),  # Unknown stage
        (20.0, 10.0, "ARTIFACT"), # Another unknown stage
    ]
    total_sec = 30.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([0, -1, -1], dtype=np.int16)  # Unknown stages are skipped
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_labels_from_annotations_empty():
    """Test handling of empty annotations list."""
    annotations = []
    total_sec = 20.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([-1, -1], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_labels_from_annotations_out_of_bounds():
    """Test handling of annotations extending beyond total_sec."""
    annotations = [
        (0.0, 10.0, "WAKE"),
        (20.0, 20.0, "REM"),    # Extends beyond total_sec
    ]
    total_sec = 30.0
    epoch_sec = 10

    result = labels_from_annotations(annotations, total_sec, epoch_sec)

    expected = np.array([0, -1, 2], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_per_epoch_format(mock_read_csv):
    """Test CSV loading with per-epoch format."""
    mock_df = pd.DataFrame({
        'epoch': [0, 1, 2, 3],
        'stage': ['WAKE', 'NREM', 'REM', 'WAKE']
    })
    mock_read_csv.return_value = mock_df

    result = labels_from_csv("test.csv", epoch_sec=10)

    expected = np.array([0, 1, 2, 0], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)
    mock_read_csv.assert_called_once_with("test.csv")


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_per_epoch_with_total_sec(mock_read_csv):
    """Test CSV loading with per-epoch format and specified total_sec."""
    mock_df = pd.DataFrame({
        'epoch': [0, 1, 2],
        'stage': ['WAKE', 'NREM', 'REM']
    })
    mock_read_csv.return_value = mock_df

    result = labels_from_csv("test.csv", epoch_sec=10, total_sec=50.0)

    expected = np.array([0, 1, 2, -1, -1], dtype=np.int16)  # 5 epochs total
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_interval_format(mock_read_csv):
    """Test CSV loading with interval format."""
    mock_df = pd.DataFrame({
        'onset_sec': [0.0, 10.0, 20.0],
        'duration_sec': [10.0, 10.0, 10.0],
        'stage': ['WAKE', 'NREM', 'REM']
    })
    mock_read_csv.return_value = mock_df

    result = labels_from_csv("test.csv", epoch_sec=10, total_sec=30.0)

    expected = np.array([0, 1, 2], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_interval_format_no_total_sec(mock_read_csv):
    """Test CSV interval format requires total_sec."""
    mock_df = pd.DataFrame({
        'onset_sec': [0.0, 10.0],
        'duration_sec': [10.0, 10.0],
        'stage': ['WAKE', 'NREM']
    })
    mock_read_csv.return_value = mock_df

    with pytest.raises(AssertionError, match="total_sec required"):
        labels_from_csv("test.csv", epoch_sec=10)


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_unrecognized_format(mock_read_csv):
    """Test handling of unrecognized CSV format."""
    mock_df = pd.DataFrame({
        'time': [0, 1, 2],
        'label': ['A', 'B', 'C']
    })
    mock_read_csv.return_value = mock_df

    with pytest.raises(ValueError, match="Unrecognized CSV label schema"):
        labels_from_csv("test.csv", epoch_sec=10)


@pytest.mark.unit
@patch('pandas.read_csv')
def test_labels_from_csv_mixed_known_unknown_stages(mock_read_csv):
    """Test CSV loading with mix of known and unknown stages."""
    mock_df = pd.DataFrame({
        'epoch': [0, 1, 2, 3],
        'stage': ['WAKE', 'UNKNOWN', 'REM', 'ARTIFACT']
    })
    mock_read_csv.return_value = mock_df

    result = labels_from_csv("test.csv", epoch_sec=10)

    expected = np.array([0, -1, 2, -1], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
@patch('pandas.read_csv', side_effect=FileNotFoundError("File not found"))
def test_labels_from_csv_file_not_found(mock_read_csv):
    """Test handling of missing CSV file."""
    with pytest.raises(FileNotFoundError):
        labels_from_csv("nonexistent.csv", epoch_sec=10)


@pytest.mark.integration
def test_labels_from_csv_real_file_per_epoch():
    """Integration test with real CSV file - per-epoch format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_epochs.csv")

        # Create test CSV with per-epoch format
        df = pd.DataFrame({
            'epoch': [0, 1, 2, 3, 4],
            'stage': ['WAKE', 'NREM', 'REM', 'WAKE', 'NREM']
        })
        df.to_csv(csv_path, index=False)

        result = labels_from_csv(csv_path, epoch_sec=10)

        expected = np.array([0, 1, 2, 0, 1], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.integration
def test_labels_from_csv_real_file_interval():
    """Integration test with real CSV file - interval format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test_intervals.csv")

        # Create test CSV with interval format
        df = pd.DataFrame({
            'onset_sec': [0.0, 15.0, 25.0],
            'duration_sec': [15.0, 10.0, 15.0],
            'stage': ['WAKE', 'NREM', 'REM']
        })
        df.to_csv(csv_path, index=False)

        result = labels_from_csv(csv_path, epoch_sec=10, total_sec=40.0)

        expected = np.array([0, 0, 1, 2], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.integration
def test_end_to_end_annotation_processing():
    """End-to-end test of annotation processing pipeline."""
    # Create annotations
    annotations = [
        (0.0, 30.0, "WAKE"),
        (30.0, 60.0, "NREM"),
        (90.0, 30.0, "REM"),
        (120.0, 30.0, "UNKNOWN")  # Should be ignored
    ]

    result = labels_from_annotations(annotations, total_sec=150.0, epoch_sec=30)

    expected = np.array([0, 1, -1, 2, -1], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)

    # Verify that 5 epochs were created (150s / 30s per epoch)
    assert len(result) == 5

    # Verify specific epoch assignments
    assert result[0] == 0   # WAKE
    assert result[1] == 1   # NREM
    assert result[2] == -1  # Uncovered (60-90s)
    assert result[3] == 2   # REM
    assert result[4] == -1  # Uncovered (last epoch, UNKNOWN was ignored)