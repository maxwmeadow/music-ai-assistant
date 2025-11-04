"""
Onset/Offset Detection Model

Lightweight temporal CNN focused on detecting note starts and ends.

Design principles:
- Temporal focus (detect edges in time, not harmonics)
- Lightweight (2-5M params vs 15M for pitch model)
- Different inductive bias from pitch model (good for model combination)
- TorchScript-compatible
"""

import torch
import torch.nn as nn
import numpy as np


class OnsetOffsetModel(nn.Module):
    """
    Lightweight onset/offset detection model.

    Architecture:
        Input (CQT 88 bins)
        → Lightweight CNN (temporal focus, minimal freq pooling)
           - 3 conv blocks (64→128→128 channels)
           - Focus on TIME dimension
        → Temporal LSTM (edge detection in time)
           - 128-256 hidden units
        → Dual binary heads: onset, offset

    Args:
        n_bins: Number of CQT bins (default: 88)
        hidden_size: LSTM hidden size (default: 128)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        n_bins: int = 88,
        hidden_size: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_size = hidden_size

        print(f"\n{'='*60}")
        print(f"ONSET/OFFSET MODEL INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  n_bins: {n_bins}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  dropout: {dropout}")

        # Lightweight CNN (temporal focus)
        # Downsampling: 500 → 125 frames (4x reduction)
        # Freq reduction: 88 → 22 bins (4x reduction)
        self.cnn = nn.Sequential(
            # Block 1: (500, 88) → (250, 44)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # Time /2, freq /2
            nn.Dropout2d(dropout * 0.5),

            # Block 2: (250, 44) → (125, 22)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # Time /2, freq /2
            nn.Dropout2d(dropout * 0.7),

            # Block 3: (125, 22) → (125, 22) [NO pooling - preserve time resolution]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # Calculate LSTM input size
        # After CNN: (batch, 128, 125, 22)
        # Reshape to: (batch, 125, 128*22) = (batch, 125, 2816)
        lstm_input_size = 128 * 22  # 2816

        print(f"  LSTM input size: {lstm_input_size}")

        # Temporal LSTM (edge detection)
        # Bidirectional to see both rising and falling edges
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Layer normalization
        self.ln = nn.LayerNorm(hidden_size * 2)

        # Shared FC layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Binary heads (sigmoid activation applied in loss)
        self.onset_head = nn.Linear(hidden_size, 1)
        self.offset_head = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

        total_params = self.count_parameters()
        print(f"  Total parameters: {total_params:,}")
        print(f"{'='*60}\n")

    def _init_weights(self):
        """Initialize weights with special handling for onset/offset heads."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        # CRITICAL: Initialize onset/offset heads with bias matching dataset density
        # Measured from full dataset:
        # - Onset density: 5.221% → logit = log(0.052207/0.947793) ≈ -2.8989
        # - Offset density: 5.213% → logit = log(0.052130/0.947870) ≈ -2.9005
        nn.init.constant_(self.onset_head.bias, -2.8989)
        nn.init.constant_(self.offset_head.bias, -2.9005)

        print(f"  ✅ Onset head initialized with bias -2.8989 (5.221% density)")
        print(f"  ✅ Offset head initialized with bias -2.9005 (5.213% density)")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: (batch, 1, time, n_bins) - CQT input

        Returns:
            dict with 'onset' and 'offset' tensors
            - onset: (batch, time, 1) - onset logits
            - offset: (batch, time, 1) - offset logits
        """
        batch_size = x.size(0)

        # CNN feature extraction
        x = self.cnn(x)  # (batch, 128, 125, 22)

        # Reshape for LSTM
        batch, channels, time_steps, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)

        # LSTM temporal processing
        x, _ = self.lstm(x)  # (batch, time, hidden*2)
        x = self.ln(x)

        # Shared FC
        x = self.fc(x)  # (batch, time, hidden)

        # Binary heads
        onset = self.onset_head(x)   # (batch, time, 1)
        offset = self.offset_head(x)  # (batch, time, 1)

        return {
            'onset': onset,
            'offset': offset
        }

    @torch.jit.export
    def get_outputs(self, x: torch.Tensor) -> tuple:
        """
        TorchScript-friendly version (returns tuple instead of dict).

        Args:
            x: (batch, 1, time, n_bins) - CQT input

        Returns:
            tuple: (onset_logits, offset_logits)
        """
        out = self.forward(x)
        return (out['onset'], out['offset'])

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING ONSET/OFFSET MODEL")
    print("="*70)

    # Test model initialization
    model = OnsetOffsetModel(n_bins=88, hidden_size=128)
    model.eval()  # Set to eval mode to disable dropout for deterministic testing

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 1, 500, 88)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Onset output shape: {output['onset'].shape}")
    print(f"Offset output shape: {output['offset'].shape}")

    # Verify shapes
    expected_time = 125  # 500 / 4
    assert output['onset'].shape == (2, expected_time, 1), f"Onset shape mismatch!"
    assert output['offset'].shape == (2, expected_time, 1), f"Offset shape mismatch!"

    print("\n✅ Forward pass successful!")
    print(f"   Input: (2, 1, 500, 88)")
    print(f"   Output: (2, 125, 1) for both onset and offset")

    # Test TorchScript-friendly version
    print("\nTesting TorchScript-friendly get_outputs()...")
    with torch.no_grad():
        tuple_output = model.get_outputs(dummy_input)
        onset_tuple, offset_tuple = tuple_output
        assert torch.allclose(onset_tuple, output['onset']), "Onset mismatch!"
        assert torch.allclose(offset_tuple, output['offset']), "Offset mismatch!"
    print("✅ TorchScript-friendly version matches!")

    # Test model size
    print(f"\nModel size: {model.count_parameters():,} parameters")
    target_size = 5_000_000  # 5M params
    if model.count_parameters() < target_size:
        print(f"✅ Model is lightweight (< {target_size:,} params)")
    else:
        print(f"⚠️  Model is larger than target ({target_size:,} params)")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
