import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from rembler.utils import sleep_utils as su


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        stratify=df.sleep,
        train_size=256 * 3,
        random_state=0,
    )
    train_df["role"] = "train"
    test_df["role"] = "test"
    return pd.concat([train_df, test_df])


class CustomSampler(Sampler):
    def __init__(self, df: pd.DataFrame):
        pass

    def __iter__(self):
        return iter(self.train_indices + self.test_indices)

    def __len__(self) -> int:
        return len(self.df)


class CustomDataset(Dataset):
    def __init__(
        self,
        training_dataframe: pd.DataFrame,
        hdf5_path: str,
        signal_names: list[str] = None,
    ):
        self.df = training_dataframe
        self.hdf5_path = hdf5_path
        self.signal_names = signal_names  # Example signal names
        self.labels = torch.as_tensor(
            self.df.sleep.map(su.stage_to_int).values, dtype=torch.int64
        )
        self._num_channels = len(self.signal_names)
        self._num_classes = self.df.sleep.nunique()

        self.hf = (
            None  # HDF5 file handle, opened in __getitem__ for multiprocessing safety
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        if self.hf is None:
            # Open the HDF5 file handle only when __getitem__ is called
            # This ensures each worker in DataLoader has its own file handle
            self.hf = h5py.File(self.hdf5_path, "r")
        row = self.df.iloc[idx]

        data = np.vstack(
            [self.hf[f"{row.bout_id}/{signal}"][:] for signal in self.signal_names]
        )
        # Convert to PyTorch tensor if needed and apply any transformations
        data = torch.from_numpy(data).float()  # Example: assuming float data
        label = torch.tensor(
            su.stage_to_int.get(row.sleep, row.sleep)
        )  # Assuming 'sleep' column contains the label

        return {"data": data, "label": label}

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def class_frequencies(self) -> torch.Tensor:
        counts = torch.bincount(self.labels, minlength=self._num_classes)
        return counts.to(torch.float32) / counts.sum()
