from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, n_inputs, n_features, kernel_size=1):
        super(FeatureExtractor, self).__init__()

        self.n_inputs = n_inputs
        self.n_features = n_features

        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_inputs, n_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Conv1d(n_features * 2, n_features, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(n_features, n_features * 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Conv1d(n_features * 2, n_inputs, 1),
        )

    def forward(self, x):
        # x: batch, n_timepoints, n_inputs
        x = x.permute(0, 2, 1)

        x_features = self.encoder(x)

        x_recon = self.decoder(x_features)

        x_features = x_features.permute(0, 2, 1)
        x_recon = x_recon.permute(0, 2, 1)

        return x_features, x_recon
