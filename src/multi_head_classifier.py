
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from .spatiotemporal_auto_encoder import SpatioTemporalAutoEncoder


class MultiHeadClassifier(pl.LightningModule):
    def __init__(self, n_inputs, n_spatial_features, n_temporal_features, n_subjects):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 124, 32)
        self.n_spatial_features = n_spatial_features
        self.n_temporal_features = n_temporal_features
        self.n_subjects = n_subjects

        self.auto_encoder = SpatioTemporalAutoEncoder(n_inputs, n_spatial_features, n_temporal_features)

        # LABEL CLASSIFIER
        self.head_cls = nn.Sequential(
            nn.Linear(n_temporal_features, n_temporal_features // 2),
            nn.ReLU(),
            nn.Linear(n_temporal_features // 2, 2),
            nn.Softmax(dim=1)
        )

        # SUBJECT CLASSIFIER
        self.head_subj_cls = nn.Linear(n_temporal_features, n_subjects)

    def forward(self, x):

        x_recon, h = self.auto_encoder(x)

        y_cls = self.head_cls(h)
        y_subj = self.head_subj_cls(h)

        return x_recon, y_cls, y_subj

    def training_step(self, batch, batch_idx):
        x, y_cls, y_subj  = batch
        x_past = x[:, :-1, :]
        x_future = x[:, :1, :]
        x_recon, y_cls_hat, y_subj_hat = self(x_past)

        # loss
        loss_recon = F.mse_loss(x_recon, x_future)
        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        loss = loss_recon + loss_cls
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss_cls', loss_cls)
        self.log('train/loss', loss)

        # metrics
        accuracy = torchmetrics.functional.accuracy(y_cls_hat.argmax(dim=1), y_cls, task='binary')
        f1 = torchmetrics.functional.f1_score(y_cls_hat.argmax(dim=1), y_cls, task='binary')

        self.log('train/accuracy', accuracy)
        self.log('train/f1', f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_subj  = batch
        x_past = x[:, :-1, :]
        x_future = x[:, 1:, :]
        x_recon, y_cls_hat, y_subj_hat = self(x_past)

        # loss
        loss_recon = F.mse_loss(x_recon, x_future)
        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        loss = loss_recon + loss_cls
        self.log('val/loss_recon', loss_recon)
        self.log('val/loss_cls', loss_cls)
        self.log('val/loss', loss)

        # metrics
        accuracy = torchmetrics.functional.accuracy(y_cls_hat.argmax(dim=1), y_cls, task='binary')
        f1 = torchmetrics.functional.f1_score(y_cls_hat.argmax(dim=1), y_cls, task='binary')

        self.log('val/accuracy', accuracy)
        self.log('val/f1', f1)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
