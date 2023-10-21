import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.utils.data import random_split


class OpenNeuroDataModule(pl.LightningDataModule):
    def __init__(self, dataset=None, test_size=.2, batch_size=32, shuffle=False):
        super().__init__()
        self.dataset = dataset or self.load_default_dataset()
        self.test_size = test_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    @classmethod
    def load_default_dataset(cls):
        raise NotImplementedError("Please define a default dataset to load.")

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
