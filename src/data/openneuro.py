from typing import Literal
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split


class OpenNeuroDataModule(pl.LightningDataModule):
    def __init__(self,
                 modality: Literal['fmri', 'eeg', 'meg'] = 'fmri',
                 dataset=None,
                 test_size: float = .2,
                 batch_size: int = 32,
                 shuffle: bool = False):
        """"""
        super().__init__()
        self.modality = modality
        self.dataset = dataset or OpenNeuroDataModule.load_default_dataset()
        self.n_subjects = len(self.dataset)
        self.test_size = test_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    @classmethod
    def load_default_dataset(cls,
                             modality: Literal['fmri', 'eeg', 'meg'] = 'fmri',
                             n_subjects=10,
                             n_timepoints=125,
                             features_dim=(10,)):

        # raise NotImplementedError("Please define a default dataset to load.")
        # shape: fmri: (n_subjects, n_timepoints, x, y, z)
        # shape: eeg: (n_subjects, n_timepoints, n_channels)
        # shape: meg: (n_subjects, n_timepoints, n_channels)
        data = torch.randn(n_subjects, n_timepoints, *features_dim)
        subject_labels = torch.arange(0, n_subjects)    # shape: (n_subjects,)
        labels = torch.randint(0, 2, (n_subjects,))     # shape: (n_subjects,)
        dataset = TensorDataset(data, subject_labels, labels)
        return dataset

    def get_example_batch(self):
        return self.dataset[0:1]  # return first sample

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        full = self.dataset
        test_size = int(len(full) * self.test_size)
        train_size = int(len(full)) - test_size
        self.train_data, self.test_data = random_split(full, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        # TODO: Implement separate test data loader
        return self.val_dataloader()
