import os
import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch as th
from torch.utils.data import DataLoader, Dataset

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler


class FnnDataset(Dataset):
    def __init__(self, X, y):
        self.X = th.tensor(X, dtype=th.float32)
        self.y = th.tensor(y, dtype=th.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CnnDataset(Dataset):
    def __init__(self, X, y):
        self.X = th.tensor(X, dtype=th.float32)
        self.y = th.tensor(y, dtype=th.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].unsqueeze(0)  # add channel dimension
        return x_sample, self.y[idx]


class SlidingWindowDataset(Dataset):
    def __init__(self, data, targets, seq_length=10, stride=1, transform=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        self.stride = stride
        self.transform = transform
        self.num_sequences = (len(self.data) - seq_length) // stride + 1
        assert self.num_sequences > 0, "Not enough data to form even one sequence!"

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length
        # shape: (seq_length, num_features)
        x_seq = self.data[start_idx:end_idx]
        y_label = self.targets[end_idx - 1]
        if self.transform:
            x_seq = self.transform(x_seq)
        # Convert to tensors
        x_seq = th.as_tensor(x_seq, dtype=th.float32)
        y_label = th.as_tensor(y_label, dtype=th.long)
        return x_seq, y_label


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        gdlc_folder,
        dataset,
        batch_size=128,
        multi_class=True,
        network_features=None,
        **dataset_kwargs
    ):

        super().__init__()
        self.gdlc_folder = gdlc_folder
        self.batch_size = batch_size
        self.multi_class = multi_class
        self.dataset = dataset
        self.network_features = network_features
        self.dataset_kwargs = dataset_kwargs
        self.model_type = "fnn"
        self.num_features = 0

    def setup(self, stage=None):
        self.X_train = pd.read_parquet(os.path.join(
            self.gdlc_folder, "training.parquet"))
        self.X_val = pd.read_parquet(os.path.join(
            self.gdlc_folder, "validation.parquet"))
        self.X_test = pd.read_parquet(os.path.join(
            self.gdlc_folder, "testing.parquet"))

        if self.multi_class:
            self.y_train = np.array(self.X_train[self.dataset.class_num_col])
            self.y_val = np.array(self.X_val[self.dataset.class_num_col])
            self.y_test = np.array(self.X_test[self.dataset.class_num_col])
        else:
            self.y_train = np.array(self.X_train[self.dataset.label_col])
            self.y_val = np.array(self.X_val[self.dataset.label_col])
            self.y_test = np.array(self.X_test[self.dataset.label_col])

        if self.network_features:
            cols_to_norm = list(set(list(self.X_train.columns)) - set(list([self.dataset.label_col, self.dataset.class_num_col])) - set(
                self.dataset.drop_columns) - set(self.dataset.weak_columns) - set(self.network_features))
        else:
            cols_to_norm = list(set(list(self.X_train.columns)) - set(list([self.dataset.label_col, self.dataset.class_num_col])) - set(
                self.dataset.drop_columns) - set(self.dataset.weak_columns))

        scaler = StandardScaler()

        self.X_train = scaler.fit_transform(self.X_train[cols_to_norm])
        self.X_val = scaler.transform(self.X_val[cols_to_norm])
        self.X_test = scaler.transform(self.X_test[cols_to_norm])

        self.num_features = self.X_train.shape[-1]

        classes = np.unique(self.y_train)
        weights = class_weight.compute_class_weight(
            'balanced', classes=classes, y=self.y_train)

        # if self.using_masking:
        #     weights = np.insert(weights, self.masked_class, 0)
        self.class_weights = th.FloatTensor(weights)

    def set_model_type(self, model_type):
        self.model_type = model_type

    def _get_dataloader(self, split):
        if split == "train":
            X, y = self.X_train, self.y_train
            shuffle = True
        elif split == "val":
            X, y = self.X_val, self.y_val
            shuffle = False
        elif split == "test":
            X, y = self.X_test, self.y_test
            shuffle = False
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.model_type == "fnn":
            dataset = FnnDataset(X, y)
        elif self.model_type in ["cnn", "cnn_lstm"]:
            dataset = CnnDataset(X, y)
        elif self.model_type in ["lstm", "gru"]:
            dataset = SlidingWindowDataset(
                X, y, seq_length=self.dataset_kwargs["sequence_length"], stride=self.dataset_kwargs["stride"])

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
