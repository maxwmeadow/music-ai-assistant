"""
Hum2Melody CRNN Model
Convolutional Recurrent Neural Network for melody extraction from humming/vocals.

Architecture:
- CNN Feature Extractor: 3 conv layers to extract spatial features from mel spectrograms
- LSTM Temporal Modeling: 2 bidirectional LSTM layers to capture temporal patterns
- Output Head: Linear layer for frame-level note predictions

Input: (batch, 1, time_frames, n_mels) mel spectrogram
Output: (batch, time_frames, num_notes) note activations
"""

import torch
import torch.nn as nn

class Hum2MelodyCRNN(nn.Module):
    """
    CRNN for melody tanscription from audio
    
    Args:
        n_mels (int): Number of mel frequency bands (default: 128)
        hidden_size (int): LSTM hidden state size (default: 256)
        num_notes (int): Number of MIDI notes to predict (default: 88, MIDI 21-108)
        dropout (float): Dropout probability (default: 0.3)
    """

    def __init__(self, n_mels: int=128, hidden_size: int=256, num_notes: int=88, dropout: float=0.3):
        super(Hum2MelodyCRNN, self).__init__()

        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_notes = num_notes

        # CNN Feature Extractor
        # INPUT: (batch, 1, time_frames, n_mels)
        # Reducing dimensions w/ maxpool
        self.cnn = nn.Sequential(
            # 1 to 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), #batch normalization is one of my favorite tricks
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #/2
            nn.Dropout2d(0.2),

            # 32 to 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /4
            nn.Dropout2d(0.2),

            # 64 to 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /8
            nn.Dropout2d(dropout),
        )

        # 3 pooling layers complete
        # time_frames has been divided by 8
        # along with n_mels
        # 128 channels

        # ouput dimensions
        self.cnn_output_height = n_mels // 8 # 128 to 16
        self.cnn_output_channels = 128
        self.lstm_input_size = self.cnn_output_channels * self.cnn_output_height # 2048

        # bidirectional lstm layers
        # long term short memory if you're wondering

        self.lstm1 = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0 # single layer
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,  # *2 for bidirectional
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_size * 2, num_notes)

        self._init_weights()
    
    def _init_weights(self):
        """
        initialize the network weights 
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input mel spectrogram
                Shape: (batch_size, 1, time_frames, n_mels)
        
        Returns:
            torch.Tensor: Frame-level note predictions (logits)
                Shape: (batch_size, time_frames//8, num_notes)
        """
        batch_size = x.size(0)

        # CNN feature extraction
        # Input: (batch, 1, 500, 128)
        # Output: (batch, 128, 62, 16)
        x = self.cnn(x)

        # Reshape for lstm
        # (batch, 128, 62, 16) -> (batch, 62, 2048)
        batch, channels, time_steps, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)
        
        # lstm modeling
        # Input: (batch, 62, 2048)
        # Output: (batch, 62, 512)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        # Input: (batch, 62, 512)
        # Output: (batch, 62, 512)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Output layer
        # Input: (batch, 62, 512)
        # Output: (batch, 62, 88)
        x = self.output_layer(x)
        
        return x
    
    def get_note_activations(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary note activations for inference.
        
        Args:
            x (torch.Tensor): Input mel spectrogram
            threshold (float): Activation threshold (default: 0.5)
        
        Returns:
            torch.Tensor: Binary note activations
                Shape: (batch_size, time_frames//8, num_notes)
        """
        logits = self.forward(x)
        probabilities = torch.sigmoid(logits)
        activations = (probabilities > threshold).float()
        return activations
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        return {
            'n_mels': self.n_mels,
            'hidden_size': self.hidden_size,
            'num_notes': self.num_notes,
            'lstm_input_size': self.lstm_input_size,
            'total_parameters': self.count_parameters(),
            'input_shape': '(batch, 1, 500, 128)',
            'output_shape': f'(batch, 62, {self.num_notes})'
        }

##############################################################

def test_model():
    """Test model with dummy data."""
    print("Testing Hum2MelodyCRNN...")
    
    # Create model
    model = Hum2MelodyCRNN()
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 500, 128)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        activations = model.get_note_activations(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Activations shape: {activations.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"Active notes per frame: {activations.sum(dim=2).mean():.2f}")
    
    print("\n✓ Model test passed!")


if __name__ == '__main__':
    test_model()