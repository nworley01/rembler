import tempfile
import os
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd
import torch
import h5py

from rembler.utils.dataset_utils import (
    split_train_test,
    CustomSampler,
    CustomDataset
)


@pytest.mark.unit
@patch('sklearn.model_selection.train_test_split')
def test_split_train_test_success(mock_train_test_split):
    """Test successful train/test split with stratification."""
    # Create mock input DataFrame
    input_df = pd.DataFrame({
        'sleep': ['A', 'R', 'S', 'A', 'R', 'S'] * 200,  # 1200 rows total
        'data': range(1200)
    })

    # Create mock split results (train: 768, test: 432)
    train_indices = list(range(768))
    test_indices = list(range(768, 1200))

    mock_train_df = input_df.iloc[train_indices].copy()
    mock_test_df = input_df.iloc[test_indices].copy()

    mock_train_test_split.return_value = (mock_train_df, mock_test_df)

    split_train_test(input_df)

    # Verify train_test_split was called with correct parameters
    mock_train_test_split.assert_called_once_with(
        input_df,
        stratify=input_df.sleep,
        train_size=256 * 3,  # 768
        random_state=0,
    )


@pytest.mark.unit
@patch('sklearn.model_selection.train_test_split')
def test_split_train_test_fixed_train_size(mock_train_test_split):
    """Test that train_size is fixed at 768 (256 * 3)."""
    input_df = pd.DataFrame({
        'sleep': ['A'] * 1000,
        'data': range(1000)
    })

    mock_train_df = pd.DataFrame({'sleep': ['A'] * 768, 'data': range(768)})
    mock_test_df = pd.DataFrame({'sleep': ['A'] * 232, 'data': range(768, 1000)})

    mock_train_test_split.return_value = (mock_train_df, mock_test_df)

    split_train_test(input_df)

    args, kwargs = mock_train_test_split.call_args
    assert kwargs['train_size'] == 768
    assert kwargs['random_state'] == 0


@pytest.mark.unit
@patch('sklearn.model_selection.train_test_split')
def test_split_train_test_stratification(mock_train_test_split):
    """Test that stratification uses sleep column."""
    input_df = pd.DataFrame({
        'sleep': ['A', 'R', 'S'],
        'other': [1, 2, 3]
    })

    mock_train_df = pd.DataFrame({'sleep': ['A'], 'other': [1]})
    mock_test_df = pd.DataFrame({'sleep': ['R', 'S'], 'other': [2, 3]})

    mock_train_test_split.return_value = (mock_train_df, mock_test_df)

    split_train_test(input_df)

    args, kwargs = mock_train_test_split.call_args
    pd.testing.assert_series_equal(kwargs['stratify'], input_df.sleep)


@pytest.mark.unit
def test_custom_sampler_initialization():
    """Test CustomSampler initialization."""
    df = pd.DataFrame({'data': [1, 2, 3]})
    sampler = CustomSampler(df)

    # The sampler stores the dataframe (though not used in current implementation)
    assert hasattr(sampler, 'train_indices')
    assert hasattr(sampler, 'test_indices')


@pytest.mark.unit
def test_custom_sampler_len():
    """Test CustomSampler __len__ method."""
    df = pd.DataFrame({'data': [1, 2, 3, 4, 5]})

    # Mock the required attributes since they're not set in __init__
    with patch.object(CustomSampler, 'df', df):
        sampler = CustomSampler(df)
        assert len(sampler) == 5


@pytest.mark.unit
def test_custom_sampler_iter():
    """Test CustomSampler __iter__ method."""
    df = pd.DataFrame({'data': [1, 2, 3]})
    sampler = CustomSampler(df)

    # Mock the indices that __iter__ uses
    sampler.train_indices = [0, 1]
    sampler.test_indices = [2]

    result = list(iter(sampler))
    expected = [0, 1, 2]

    assert result == expected


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1, 'S': 2})
def test_custom_dataset_initialization():
    """Test CustomDataset initialization."""
    df = pd.DataFrame({
        'sleep': ['A', 'R', 'S'],
        'bout_id': ['bout1', 'bout2', 'bout3']
    })

    dataset = CustomDataset(df, "/fake/path.h5", ["eeg", "emg"])

    assert len(dataset.df) == 3
    assert dataset.hdf5_path == "/fake/path.h5"
    assert dataset.signal_names == ["eeg", "emg"]
    assert dataset._num_channels == 2
    assert dataset._num_classes == 3
    assert dataset.hf is None  # Not opened yet

    # Check labels tensor
    expected_labels = torch.tensor([0, 1, 2], dtype=torch.int64)
    torch.testing.assert_close(dataset.labels, expected_labels)


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1})
def test_custom_dataset_properties():
    """Test CustomDataset properties."""
    df = pd.DataFrame({
        'sleep': ['A', 'R', 'A'],
        'bout_id': ['bout1', 'bout2', 'bout3']
    })

    dataset = CustomDataset(df, "/fake/path.h5", ["eeg"])

    assert dataset.num_classes == 2  # A and R
    assert dataset.num_channels == 1  # eeg only


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1, 'S': 2})
def test_custom_dataset_class_frequencies():
    """Test CustomDataset class_frequencies method."""
    df = pd.DataFrame({
        'sleep': ['A', 'A', 'R', 'S'],  # 2 A's, 1 R, 1 S
        'bout_id': ['bout1', 'bout2', 'bout3', 'bout4']
    })

    dataset = CustomDataset(df, "/fake/path.h5")

    frequencies = dataset.class_frequencies()

    # Expected: [2/4, 1/4, 1/4] = [0.5, 0.25, 0.25]
    expected = torch.tensor([0.5, 0.25, 0.25])
    torch.testing.assert_close(frequencies, expected)


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0})
@patch('h5py.File')
def test_custom_dataset_getitem(mock_h5py_file):
    """Test CustomDataset __getitem__ method."""
    df = pd.DataFrame({
        'sleep': ['A'],
        'bout_id': ['bout1']
    })

    # Mock HDF5 file structure
    mock_file = MagicMock()
    mock_h5py_file.return_value = mock_file

    # Mock signal data
    eeg_data = np.array([1.0, 2.0, 3.0])
    emg_data = np.array([0.1, 0.2, 0.3])

    mock_file.__getitem__.side_effect = lambda key: {
        'bout1/eeg': eeg_data,
        'bout1/emg': emg_data
    }[key]

    dataset = CustomDataset(df, "/fake/path.h5", ["eeg", "emg"])

    result = dataset[0]

    # Verify HDF5 file was opened
    mock_h5py_file.assert_called_once_with("/fake/path.h5", "r")

    # Verify result structure
    assert 'data' in result
    assert 'label' in result

    # Check data shape (2 channels x 3 samples)
    expected_data = torch.from_numpy(np.vstack([eeg_data, emg_data])).float()
    torch.testing.assert_close(result['data'], expected_data)

    # Check label
    expected_label = torch.tensor(0)  # 'A' maps to 0
    torch.testing.assert_close(result['label'], expected_label)


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0})
@patch('h5py.File')
def test_custom_dataset_getitem_single_signal(mock_h5py_file):
    """Test CustomDataset __getitem__ with single signal."""
    df = pd.DataFrame({
        'sleep': ['A'],
        'bout_id': ['bout1']
    })

    mock_file = MagicMock()
    mock_h5py_file.return_value = mock_file

    eeg_data = np.array([1.0, 2.0, 3.0])
    mock_file.__getitem__.return_value = eeg_data

    dataset = CustomDataset(df, "/fake/path.h5", ["eeg"])

    result = dataset[0]

    # Check data shape (1 channel x 3 samples)
    expected_data = torch.from_numpy(eeg_data[np.newaxis, :]).float()
    torch.testing.assert_close(result['data'], expected_data)


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {})
def test_custom_dataset_unknown_stage():
    """Test CustomDataset handling of unknown sleep stage."""
    df = pd.DataFrame({
        'sleep': ['UNKNOWN'],
        'bout_id': ['bout1']
    })

    with patch('h5py.File'):
        dataset = CustomDataset(df, "/fake/path.h5")

        # stage_to_int.get should return the original value if not found
        result = dataset[0]

        # The label should be the original string since it's not in the mapping
        assert result['label'] == 'UNKNOWN'


@pytest.mark.unit
@patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1})
def test_custom_dataset_len():
    """Test CustomDataset __len__ method."""
    df = pd.DataFrame({
        'sleep': ['A', 'R', 'A'],
        'bout_id': ['bout1', 'bout2', 'bout3']
    })

    dataset = CustomDataset(df, "/fake/path.h5")

    assert len(dataset) == 3


@pytest.mark.integration
def test_split_train_test_integration():
    """Integration test for split_train_test with real data."""
    # Create a dataset large enough for the fixed train size
    np.random.seed(42)
    sleep_stages = np.random.choice(['A', 'R', 'S'], size=1000)

    df = pd.DataFrame({
        'sleep': sleep_stages,
        'data': np.random.randn(1000),
        'bout_id': [f'bout_{i}' for i in range(1000)]
    })

    result = split_train_test(df)

    # Verify basic structure
    assert 'role' in result.columns
    assert len(result) == 1000

    # Verify roles
    train_count = (result['role'] == 'train').sum()
    test_count = (result['role'] == 'test').sum()

    assert train_count == 768  # 256 * 3
    assert test_count == 232   # 1000 - 768
    assert train_count + test_count == len(result)


@pytest.mark.integration
def test_custom_dataset_with_real_hdf5():
    """Integration test with real HDF5 file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        h5_path = os.path.join(temp_dir, "test_data.h5")

        # Create test HDF5 file
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("bout1/eeg", data=np.array([1.0, 2.0, 3.0]))
            f.create_dataset("bout1/emg", data=np.array([0.1, 0.2, 0.3]))
            f.create_dataset("bout2/eeg", data=np.array([4.0, 5.0, 6.0]))
            f.create_dataset("bout2/emg", data=np.array([0.4, 0.5, 0.6]))

        # Create dataset
        df = pd.DataFrame({
            'sleep': ['A', 'R'],
            'bout_id': ['bout1', 'bout2']
        })

        with patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1}):
            dataset = CustomDataset(df, h5_path, ["eeg", "emg"])

            # Test first sample
            sample1 = dataset[0]
            assert sample1['data'].shape == (2, 3)  # 2 channels, 3 samples
            torch.testing.assert_close(sample1['data'][0], torch.tensor([1.0, 2.0, 3.0]))
            torch.testing.assert_close(sample1['data'][1], torch.tensor([0.1, 0.2, 0.3]))
            assert sample1['label'] == 0

            # Test second sample
            sample2 = dataset[1]
            assert sample2['data'].shape == (2, 3)
            torch.testing.assert_close(sample2['data'][0], torch.tensor([4.0, 5.0, 6.0]))
            torch.testing.assert_close(sample2['data'][1], torch.tensor([0.4, 0.5, 0.6]))
            assert sample2['label'] == 1


@pytest.mark.integration
def test_end_to_end_dataset_workflow():
    """End-to-end test of dataset workflow."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'sleep': np.random.choice(['A', 'R', 'S'], size=n_samples),
        'bout_id': [f'bout_{i}' for i in range(n_samples)],
        'other_feature': np.random.randn(n_samples)
    })

    # Split train/test
    split_df = split_train_test(df)

    # Verify split worked
    assert len(split_df) == n_samples
    assert 'role' in split_df.columns

    train_data = split_df[split_df['role'] == 'train']
    test_data = split_df[split_df['role'] == 'test']

    assert len(train_data) == 768
    assert len(test_data) == 232

    # Test that we can create datasets from splits
    with tempfile.TemporaryDirectory() as temp_dir:
        h5_path = os.path.join(temp_dir, "test.h5")

        # Create minimal HDF5 file for testing
        with h5py.File(h5_path, "w") as f:
            for i in range(10):  # Just first 10 for testing
                f.create_dataset(f"bout_{i}/eeg", data=np.random.randn(100))

        # Create dataset from first 10 samples
        small_df = train_data.head(10)

        with patch('rembler.utils.sleep_utils.stage_to_int', {'A': 0, 'R': 1, 'S': 2}):
            dataset = CustomDataset(small_df, h5_path, ["eeg"])

            assert len(dataset) == 10
            assert dataset.num_channels == 1

            # Test sampling
            sample = dataset[0]
            assert 'data' in sample
            assert 'label' in sample
            assert sample['data'].shape[0] == 1  # 1 channel