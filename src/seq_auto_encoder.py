import torch
from torch import nn


class SeqAutoEncoder(nn.Module):
    def __init__(self, n_inputs, hidden_size=64):
        super(SeqAutoEncoder, self).__init__()

        self.n_inputs = n_inputs
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(n_inputs, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_inputs, batch_first=True)

        self.fc_decoder = nn.Linear(n_inputs, n_inputs)

    def forward(self, x):
        batch_size = x.size(0)
        n_timepoints = x.size(1)

        # encode
        y_enc, (h_enc, c_enc) = self.encoder(x)
        x_enc = torch.rand(batch_size, n_timepoints, self.hidden_size, device=h_enc.device)

        # decode
        y_dec, (h_dec, c_dec) = self.decoder(x_enc, (h_enc, c_enc))
        x_recon = self.fc_decoder(y_dec)

        h = h_enc[-1, :, :]  # last hidden state of encoder

        return x_recon, h
