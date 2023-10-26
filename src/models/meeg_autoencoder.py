
import torch
from torch import nn


class MEEGAutoEncoder(nn.Module):
    def __init__(self, n_spatial_features, n_temporal_features, example_input_array):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 124, 32)
        self.n_spatial_features = n_spatial_features
        self.n_temporal_features = n_temporal_features

        # skip batch size (and then skip time dimension)
        example_input = example_input_array[0]  # select input, shape: (n_timepoints, *n_features)
        n_channels = torch.prod(torch.tensor(example_input.shape[2:])).item()
        kernel_size = 1

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_spatial_features * 2, kernel_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(n_spatial_features * 2, n_spatial_features, 1),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose1d(n_spatial_features, n_spatial_features * 2,
                               kernel_size),
            nn.ReLU(),
            nn.ConvTranspose1d(n_spatial_features * 2, n_channels, kernel_size),
        )

        self.temporal_encoder = nn.LSTM(n_spatial_features, n_temporal_features,
                                        batch_first=True)
        self.temporal_decoder = nn.LSTM(n_temporal_features, n_temporal_features,
                                        batch_first=True)

        # self.fc_decoder = nn.Linear(n_spatial_features, n_timepoints * n_temporal_features)
        # self.fc_encoder = nn.Linear(n_spatial_features, n_timepoints * n_temporal_features)

    def encode(self, x):

        x = x.permute(0, 2, 1)  # (batch_size, n_channels, n_timepoints)

        # spatial encoding
        x = self.spatial_encoder(x)
        x = x.permute(0, 2, 1)  # (batch_size, n_timepoints, n_channels)

        # temporal encoding
        _, (h, c) = self.temporal_encoder(x)

        return h, c

    def decode(self, h, c, n_timepoints):

        batch_size = h.shape[1]

        # initialize input to temporal decoder
        x = torch.zeros(batch_size, n_timepoints, self.n_spatial_features)

        # temporal decoding
        x_recon, _ = self.temporal_decoder(x, (h, c))
        x_recon = x_recon.permute(0, 2, 1)

        # spatial decoding
        x_recon = self.spatial_decoder(x_recon)
        x_recon = x_recon.permute(0, 2, 1)

        return x_recon

    def forward(self, x):

        n_timepoints = x.shape[1]

        h, c = self.encode(x)
        x_recon = self.decode(h, c, n_timepoints)

        h = h.squeeze(0)  # Remove the layer dimension

        return x_recon, h
