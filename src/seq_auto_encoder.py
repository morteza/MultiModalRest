import torch
from torch import nn


class SeqAutoEncoder(nn.Module):
    def __init__(self, n_inputs, n_features, hidden_size):
        super(SeqAutoEncoder, self).__init__()

        self.n_inputs = n_inputs
        self.hidden_size = hidden_size

        self.fe_encoder = nn.Sequential(
            nn.Conv1d(n_inputs, n_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Conv1d(n_features * 2, n_features, 1),
        )

        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)
        self.fc_decoder = nn.Linear(n_features, n_features)

        self.fe_decoder = nn.Sequential(
            nn.Conv1d(n_features, n_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Conv1d(n_features * 2, n_inputs, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        n_timepoints = x.size(1)

        # feature extraction
        x = x.permute(0, 2, 1)
        x_features = self.fe_encoder(x)
        x_features = x_features.permute(0, 2, 1)

        # encode
        y_enc, (h_enc, c_enc) = self.encoder(x_features)
        x_enc = torch.rand(batch_size, n_timepoints, self.hidden_size, device=h_enc.device)

        # decode
        y_dec, (h_dec, c_dec) = self.decoder(x_enc, (h_enc, c_enc))

        # x_recon = self.fc_decoder(y_dec)
        y_dec = y_dec.permute(0, 2, 1)
        x_recon = self.fe_decoder(y_dec)
        x_recon = x_recon.permute(0, 2, 1)

        h = h_enc[-1, :, :]  # last hidden state of encoder

        return x_recon, h
