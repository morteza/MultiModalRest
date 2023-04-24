import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import os


class Julia2018DataModule(pl.LightningDataModule):
    def __init__(self, X, y_cls, y_subject, train_ratio=0.75, batch_size=8, shuffle=False):
        super().__init__()
        self.X = X
        self.y_cls = y_cls
        self.y_subject = y_subject
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = os.cpu_count()

    def prepare_data(self):
        trn_idx, val_idx = train_test_split(
            torch.arange(0, self.y_subject.max() + 1),
            train_size=self.train_ratio,
            stratify=self.y_cls)

        self.trn_data = torch.utils.data.TensorDataset(self.X[trn_idx], self.y_cls[trn_idx], self.y_subject[trn_idx])
        self.val_data = torch.utils.data.TensorDataset(self.X[val_idx], self.y_cls[val_idx], self.y_subject[val_idx])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trn_data,
                                           batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=self.num_workers)

