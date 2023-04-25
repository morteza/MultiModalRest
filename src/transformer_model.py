import torch.nn as nn
import torch
import math
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional as metrics


class PositionalEncoding(nn.Module):

    def __init__(self, n_features: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_features, 2) * (-math.log(10000.0) / n_features))
        pe = torch.zeros(max_len, 1, n_features)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len = x.size(1)

        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TransformerModel(pl.LightningModule):
    def __init__(self, n_features=64, n_outputs=2, n_heads=1, n_layers=2, dropout=0):
        super().__init__()

        # self.pos_encoder = PositionalEncoding(n_features)
        encoder_layer = nn.TransformerEncoderLayer(n_features, n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.decoder = nn.Sequential(
            nn.Linear(n_features, n_outputs),
            nn.Sigmoid()
        )

        # self.model = nn.Transformer(d_model=n_features, batch_first=True)

    def forward(self, x):

        src = x  # + self.pos_encoder(x)
        # src = x[:, :-1, :]
        # tgt = x[:, 1:, :]

        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device)

        z = self.encoder(src, mask)
        z = z.permute(0, 2, 1)  # (batch, features, seq_len)

        z = nn.AvgPool1d(seq_len)(z).squeeze()

        out = self.decoder(z)

        return out

    def training_step(self, batch, batch_idx):
        x, y_cls, y_subj  = batch
        y_cls_hat = self(x)

        # loss
        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        loss = loss_cls
        self.log('train/loss', loss)

        # metrics
        accuracy = metrics.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        f1 = metrics.f1_score(y_cls_hat.argmax(dim=1), y_cls, task='binary')

        self.log('train/accuracy', accuracy)
        self.log('train/f1', f1)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_subj  = batch
        y_cls_hat = self(x)

        # loss
        loss_cls = F.cross_entropy(y_cls_hat, y_cls)
        loss = loss_cls
        self.log('val/loss', loss)

        # metrics
        accuracy = metrics.accuracy(y_cls_hat, y_cls, task='multiclass', num_classes=2)
        f1 = metrics.f1_score(y_cls_hat.argmax(dim=1), y_cls, task='binary')

        self.log('val/accuracy', accuracy)
        self.log('val/f1', f1)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
