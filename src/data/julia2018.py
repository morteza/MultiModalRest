import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import os
import xarray as xr
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


class Julia2018DataModule(pl.LightningDataModule):
    def __init__(self, segment_size=31, train_ratio=0.75, split_on='subject', batch_size=8, shuffle=False,
                 example_path='data/julia2018/timeseries_dosenbach2010.nc5'):

        super().__init__()
        self.segment_size = segment_size
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = os.cpu_count()
        self.split_on = split_on
        self._n_features = None  # will be set in load_from_file

        # identify subjects
        self.subjects = xr.open_dataset(example_path, engine='h5netcdf')['subject'].values
        self.subject_encoder = LabelEncoder().fit(self.subjects)
        self.label_encoder = LabelEncoder().fit([subj[:4] for subj in self.subjects])

    @property
    def n_features(self):
        if self._n_features is None:
            raise ValueError('n_features is not set. Call prepare_data() first.')
        return self._n_features

    def get_atlas_name(self, file_name):
        name = file_name.replace('-', '_').replace('_2mm', '')
        name = name.replace('difumo_', 'difumo')
        return name.split('_', 1)[1]

    def load_from_file(self,
                       path='data/julia2018',
                       selected_parcellations=['dosenbach2010', 'gordon2014', 'difumo64']):

        # load time-series and concatenate them along the regions dimension
        ts_files = [f for f in Path(path).glob('*.nc5')
                    if self.get_atlas_name(f.stem) in selected_parcellations]

        X_regions = []

        for ts_file in ts_files:

            ds = xr.open_dataset(ts_file, engine='h5netcdf').map(lambda x: x.values)

            # time-series
            ts = torch.tensor(ds['timeseries'].values).float()  # (n_subjects, n_timepoints, n_regions)
            X_regions.append(ts)

        X = torch.cat(X_regions, dim=-1)

        self._n_features = X.size(-1)

        # subjects
        y_subject = self.subject_encoder.transform(ds['subject'].values)
        y_subject = torch.tensor(y_subject)  # (n_subjects, n_regions)

        # labels
        y_lbl = self.label_encoder.transform(ds['subject'].values)
        y_lbl = torch.tensor(y_lbl)  # (n_subjects, n_regions)

        X = X - X.mean(dim=1, keepdim=True)
        X = F.normalize(X, dim=1)

        return X, y_lbl, y_subject

    def segment(self, X, *args):
        """helper to segment a tensor into segments of size `segment_size` along the second dimension"""
        X_seg = X.unfold(1, self.segment_size, self.segment_size).permute(0, 1, 3, 2)
        n_segments = X_seg.size(1)

        X_seg = X_seg.flatten(0, 1)

        if len(args) == 0:
            return X_seg

        # continue with the segmentation of the args
        args_seg = []
        for arg in args:
            arg_seg = arg.reshape(-1, 1).repeat(1, n_segments).flatten()
            args_seg.append(arg_seg)

        return X_seg, *args_seg

    def prepare_data(self):

        X, y_lbl, y_subject = self.load_from_file()

        if self.split_on == 'subject':
            split_dim = 0
            n_subjects = X.size(0)

            trn_idx, val_idx = train_test_split(
                torch.arange(0, n_subjects),
                train_size=self.train_ratio,
                stratify=y_lbl)

            # segment time-series and split subjects into train and validation
            self.trn_data = torch.utils.data.TensorDataset(
                *self.segment(X[trn_idx], y_lbl[trn_idx], y_subject[trn_idx])
            )
            self.val_data = torch.utils.data.TensorDataset(
                *self.segment(X[val_idx], y_lbl[val_idx], y_subject[val_idx])
            )
        elif self.split_on == 'sequence':
            split_dim = 1
            seq_len = X.size(1)
            trn_idx = torch.arange(0, int(seq_len * self.train_ratio))
            val_idx = torch.arange(trn_idx.max() + 1, seq_len)

            # segment time-series and split subjects into train and validation
            self.trn_data = torch.utils.data.TensorDataset(
                *self.segment(torch.index_select(X, split_dim, trn_idx), y_lbl, y_subject)
            )
            self.val_data = torch.utils.data.TensorDataset(
                *self.segment(torch.index_select(X, split_dim, val_idx), y_lbl, y_subject)
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trn_data,
                                           batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=self.num_workers)
