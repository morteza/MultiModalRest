
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

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_spatial_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(n_spatial_features * 2, n_spatial_features, 1),
        )

        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose1d(n_spatial_features, n_spatial_features * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_spatial_features * 2, n_channels, kernel_size=3, padding=1),
        )

        self.temporal_encoder = nn.LSTM(n_spatial_features, n_temporal_features,
                                        batch_first=True)
        self.temporal_decoder = nn.LSTM(n_temporal_features, n_spatial_features, batch_first=True)

    def encode(self, x):


    def forward(self, x):

        batch_size = x.size(0)
        n_timepoints = x.size(1)

        print('x.shape', x.shape)

        # flatten features
        x_flat = x.view(batch_size, n_timepoints, -1)
        x_flat = x_flat.permute(0, 2, 1)  # (batch_size, n_features, n_timepoints)

        print('x_flat.shape', x_flat.shape)

        # spatial encoding
        y_s_enc = self.spatial_encoder(x_flat)
        y_s_enc = y_s_enc.permute(0, 2, 1)  # (batch_size, n_timepoints, n_features)

        print('y_s_enc.shape', y_s_enc.shape)

        # temporal encoding
        y_t_enc, (h, _) = self.temporal_encoder(y_s_enc)
        h = h[-1]  # select last layer, shape: (batch_size, n_temporal_features)
        h = h.unsqueeze(1).repeat(1, x.size(1), 1)

        print('h.shape', h.shape)
        print('y_t_enc.shape', y_t_enc.shape)

        x_dec = h.permute(1, 0, 2).repeat(1, n_timepoints, 1)

        print('x_dec.shape', x_dec.shape)

        # y_t_enc, (h, c_t) = self.temporal_encoder(y_s_enc)
        # h_enc = h.permute(1, 0, 2).repeat(1, n_timepoints, 1)

        # temporal decoding
        # y_t_dec, (_, _) = self.temporal_decoder(h_enc)
        y_t_dec, _ = self.temporal_decoder(x_dec)

        print('y_t_dec.shape', y_t_dec.shape)

        # space decoding
        y_s_dec = self.spatial_decoder(y_t_dec.permute(0, 2, 1)).permute(0, 2, 1)

        print('y_s_dec.shape', y_s_dec.shape)

        x_recon = y_s_dec.view(*x.shape)

        return x_recon, h
