"""
Beatbox2drums CNN Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BeatBox2DrumsCNN(nn.Module):
    """
    CNN-only model for drum hit detection and classification

    3 CNN blocks (2 w/ pool)
    Conv1d for temporal processing
    Linear output head
    """
    def __init__(self, n_mels: int=128, num_drum_types: int = 4, dropout: float = 0.3):
        super().__init__()

        self.n_mels = n_mels
        self.num_drum_types = num_drum_types

        # Block 1: 1 → 32 channels
        # Input: (batch, 1, 500, 128)
        # Output: (batch, 32, 250, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2
            nn.Dropout2d(0.2)
        )

        # Block 2: 32 → 64 channels
        # Input: (batch, 32, 250, 64)
        # Output: (batch, 64, 125, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # /2
            nn.Dropout2d(0.2)
        )

        # Block 3: 64 → 128 channels (NO POOLING - preserve temporal resolution)
        # Input: (batch, 64, 125, 32)
        # Output: (batch, 128, 125, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        # Calculate flattened spatial dimension after CNN
        # After 2 pooling layers: time / 4, freq / 4
        self.time_frames = 125  # 500 / 4
        self.freq_bins = 32     # 128 / 4
        self.flattened_spatial = 128 * 32  # channels * freq = 4096

        # Conv1d for temporal processing
        # Captures rhythmic patterns across time
        self.temporal_conv = nn.Conv1d(
            in_channels=self.flattened_spatial,
            out_channels=256,
            kernel_size=5,
            padding=2
        )

        # Output head
        self.fc_out = nn.Linear(256, num_drum_types)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # Kaiming initialization for Conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, 1, 500, 128) - mel spectrogram
            
        Returns:
            (batch, 125, 4) - per-frame drum class logits
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv1(x)  # (batch, 32, 250, 64)
        x = self.conv2(x)  # (batch, 64, 125, 32)
        x = self.conv3(x)  # (batch, 128, 125, 32)
        
        # Reshape for temporal processing
        # (batch, channels, time, freq) → (batch, channels*freq, time)
        batch, channels, time_steps, freq = x.size()
        x = x.view(batch_size, channels * freq, time_steps)  # (batch, 4096, 125)
        
        # Conv1d across time dimension
        x = self.temporal_conv(x)  # (batch, 256, 125)
        x = F.relu(x)
        
        # Reshape for output layer
        # (batch, 256, 125) → (batch, 125, 256)
        x = x.permute(0, 2, 1)
        
        # Output layer
        x = self.fc_out(x)  # (batch, 125, 4)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def test_model():
    """Test the BeatBox2Drums model."""
    print("Testing BeatBox2DrumsCNN...")
    
    model = BeatBox2DrumsCNN()
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 500, 128)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 125, 4)")
    
    # Verify shapes
    assert output.shape == (batch_size, 125, 4), f"Output shape mismatch! Got {output.shape}"
    
    # Check forward pass works
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n✓ Model test passed!")
    
    # Calculate temporal reduction
    input_frames = 500
    output_frames = 125
    reduction = input_frames / output_frames
    print(f"Temporal reduction: {reduction:.1f}x (4x from 2 pooling layers)")
    
    return model


if __name__ == '__main__':
    test_model()
