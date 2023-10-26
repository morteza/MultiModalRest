# /mnt/data/autoencoder_cnn_lstm_fixed_v6.py

class EncoderFixedV2(nn.Module):
    def __init__(self, n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim):
        super(EncoderFixedV2, self).__init__()
        
        # CNN layers for spatial features
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # LSTM layer for temporal features
        self.lstm = nn.LSTM(128, temporal_hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(n_timepoints * temporal_hidden_dim, spatial_hidden_dim)

    def forward(self, x):
        # Rearrange dimensions to (batch, n_channels, n_timepoints)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Rearrange dimensions back to (batch, n_timepoints, n_channels)
        x = x.permute(0, 2, 1)
        
        # LSTM for temporal features
        x, _ = self.lstm(x)
        
        x = x.reshape(x.size(0), -1)  # Flatten
        h = self.fc(x)
        return h
    
class DecoderFixedV6(nn.Module):
    def __init__(self, n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim):
        super(DecoderFixedV6, self).__init__()
        
        self.fc = nn.Linear(spatial_hidden_dim, n_timepoints * temporal_hidden_dim)
        
        # LSTM layer for temporal features
        self.lstm = nn.LSTM(temporal_hidden_dim, 64, batch_first=True)
        
        # CNN layers for spatial features
        self.deconv1 = nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, h):
        h = self.fc(h)
        h = h.view(h.size(0), n_timepoints, temporal_hidden_dim)  # Reshape to fit LSTM input
        
        # LSTM for temporal features
        h, _ = self.lstm(h)
        
        # Rearrange dimensions to (batch, n_channels, n_timepoints)
        h = h.permute(0, 2, 1)
        
        # CNN for spatial features
        h = F.relu(self.deconv1(h))
        
        # Rearrange dimensions back to (batch, n_timepoints, n_channels)
        x_reconstructed = F.relu(self.deconv2(h))
        x_reconstructed = x_reconstructed.permute(0, 2, 1)
        
        return x_reconstructed

# Replace the old decoder with the new fixed version and test again
class AutoencoderFixedV6(nn.Module):
    def __init__(self, n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim):
        super(AutoencoderFixedV6, self).__init__()
        self.encoder = EncoderFixedV2(n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim)
        self.decoder = DecoderFixedV6(n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        x_reconstructed = self.decoder(h)
        return x_reconstructed

# Test the modified Autoencoder model to make sure the dimension issue is resolved
model_fixed_v6 = AutoencoderFixedV6(n_timepoints, n_channels, temporal_hidden_dim, spatial_hidden_dim)
reconstructed_fixed_v6 = model_fixed_v6(train_data)
loss_fixed_v6 = criterion(reconstructed_fixed_v6, train_data)

loss_fixed_v6.item()
