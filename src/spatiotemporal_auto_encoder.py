
import torch
from torch import nn


class SpatioTemporalAutoEncoder(nn.Module):
    def __init__(self, n_inputs, n_spatial_features, n_temporal_features):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 124, 32)
        self.n_spatial_features = n_spatial_features
        self.n_temporal_features = n_temporal_features

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(n_inputs, n_spatial_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Conv1d(n_spatial_features * 2, n_spatial_features, 1),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose1d(n_spatial_features, n_inputs, kernel_size=3, padding=1),
        )

        self.temporal_encoder = nn.LSTM(n_spatial_features, n_temporal_features, batch_first=True)
        self.temporal_decoder = nn.LSTM(n_temporal_features, n_spatial_features, batch_first=True)

    def forward(self, x):

        batch_size = x.size(0)
        n_timepoints = x.size(1)

        # spatial encoding
        x_s = self.spatial_encoder(x.permute(0, 2, 1)).permute(0, 2, 1)

        # TODO: flatten

        # temporal encoding
        y_t_enc, (h_t, c_t) = self.temporal_encoder(x_s)
        x_t = torch.rand(batch_size, n_timepoints, self.n_temporal_features, device=h_t.device)

        # temporal decoding
        y_t_dec, (h_t, c_t) = self.temporal_decoder(x_t, (h_t, c_t))
        h = h_t[-1, :, :]  # last hidden state of encoder

        # TODO deflatten

        # spatial decoding
        x_recon = self.spatial_decoder(y_t_dec.permute(0, 2, 1))

        return x_recon, h
