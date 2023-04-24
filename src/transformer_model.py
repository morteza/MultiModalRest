import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len = x.size(1)

        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_features=64, n_outputs=2, n_heads=4, n_layers=3, dropout=0, device='cpu'):
        super().__init__()
        self.device = device

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(n_features)
        encoder_layer = nn.TransformerEncoderLayer(n_features, n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # self.decoder = nn.Linear(n_features, n_outputs)
        # self.model = nn.Transformer(d_model=n_features, batch_first=True)

    def forward(self, x):

        src = x
        # src = x[:, :-1, :]
        # tgt = x[:, 1:, :]

        seq_len = src.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=self.device)

        out = self.encoder(src, mask)

        return out
