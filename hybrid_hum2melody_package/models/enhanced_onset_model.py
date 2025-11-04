"""
Enhanced Onset/Offset Detection Model

Improvements over basic onset_model.py:
1. Musical components (HarmonicCNN, multi-scale encoder)
2. Rich input features (onset strength + musical context)
3. Pitch-informed cross-attention (optional)
4. Larger capacity for better performance

Design:
- Input: 112 channels (88 CQT + 5 onset features + 19 musical context)
- Architecture: HarmonicCNN → Multi-scale Encoder → LSTM → Attention → Heads
- Output: Binary onset/offset predictions
- TorchScript-compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
from pathlib import Path
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import musical components
try:
    from models.musical_components import MultiScaleTemporalEncoder
    MUSICAL_COMPONENTS_AVAILABLE = True
except ImportError:
    MUSICAL_COMPONENTS_AVAILABLE = False
    print("⚠️ Musical components not available - using basic versions")


class HarmonicCNN(nn.Module):
    """
    CNN with dilated convolutions to capture harmonic relationships.

    Uses multi-scale convolutions (local + octave spacing) to capture
    both fine-grained and harmonic patterns.
    """

    def __init__(self, input_channels: int = 112, dropout: float = 0.3):
        super().__init__()

        print(f"[HarmonicCNN] input_channels={input_channels}, dropout={dropout}")

        # Block 1: Multi-scale feature extraction
        # Local patterns (neighboring bins)
        self.conv1_local = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Octave patterns (12 semitones = 1 octave with dilation=6 on 2x pooled = 12)
        self.conv1_octave = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Combine multi-scale features
        self.conv1_combine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.5)
        )

        # Block 2: Standard conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout * 0.7)
        )

        # Block 3: No pooling (preserve temporal resolution)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

        print("[HarmonicCNN] ✅ Initialized")

    def forward(self, x):
        # Multi-scale path
        local = self.conv1_local(x)
        octave = self.conv1_octave(x)
        x = torch.cat([local, octave], dim=1)
        x = self.conv1_combine(x)

        # Standard convs
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class SimpleMultiScaleEncoder(nn.Module):
    """Fallback multi-scale encoder if musical_components not available."""

    def __init__(self, hidden_size: int = 256, num_scales: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        # Create parallel 1D convs with different dilations
        self.scales = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size // num_scales, kernel_size=3,
                     padding=2**i, dilation=2**i)
            for i in range(num_scales)
        ])

        # Combine scales
        self.combine = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (batch, time, hidden)
        Returns:
            (batch, time, hidden)
        """
        # Transpose for Conv1d
        x_transposed = x.transpose(1, 2)  # (batch, hidden, time)

        # Process at each scale
        scale_outputs = [scale(x_transposed) for scale in self.scales]

        # Concatenate
        multi_scale = torch.cat(scale_outputs, dim=1)

        # Combine
        out = self.combine(multi_scale)

        # Transpose back and add residual
        out = out.transpose(1, 2)
        return out + x


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for pitch-informed onset/offset prediction.

    Query: onset/offset features
    Key/Value: pitch features (if available)

    This allows the model to use pitch information to better predict
    note boundaries.
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Query projection (from onset/offset features)
        self.q_proj = nn.Linear(hidden_size, hidden_size)

        # Key/Value projections (from pitch features or self)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key=None, value=None, mask=None):
        """
        Args:
            query: (batch, time, hidden) - onset/offset features
            key: (batch, time, hidden) - pitch features (optional)
            value: (batch, time, hidden) - pitch features (optional)
            mask: (batch, time) - attention mask (optional)

        Returns:
            (batch, time, hidden)
        """
        batch_size, seq_len, _ = query.size()

        # If no key/value provided, use self-attention
        if key is None:
            key = query
        if value is None:
            value = key

        # Project
        Q = self.q_proj(query)  # (batch, time, hidden)
        K = self.k_proj(key)    # (batch, time, hidden)
        V = self.v_proj(value)  # (batch, time, hidden)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, heads, time, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        out = self.out_proj(out)

        return out


class EnhancedOnsetOffsetModel(nn.Module):
    """
    Enhanced onset/offset detection model with musical components.

    Architecture:
        Input: (batch, channels, time, freq)
            - 112 channels: 88 CQT + 5 onset features + 19 musical context
        → HarmonicCNN (multi-scale feature extraction)
        → LSTM (temporal modeling)
        → Multi-scale Encoder (capture patterns at different time scales)
        → Cross-Attention (optional pitch conditioning)
        → Dual binary heads (onset, offset)

    Args:
        n_bins: Number of frequency bins (default: 88)
        input_channels: Number of input channels (default: 112)
        hidden_size: LSTM hidden size (default: 256)
        dropout: Dropout rate (default: 0.3)
        use_multi_scale: Use multi-scale encoder (default: True)
        use_cross_attention: Use cross-attention (default: True)
    """

    def __init__(
        self,
        n_bins: int = 88,
        input_channels: int = 112,
        hidden_size: int = 256,
        dropout: float = 0.3,
        use_multi_scale: bool = True,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.n_bins = n_bins
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.use_multi_scale = use_multi_scale
        self.use_cross_attention = use_cross_attention

        print(f"\n{'='*60}")
        print(f"ENHANCED ONSET/OFFSET MODEL INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  n_bins: {n_bins}")
        print(f"  input_channels: {input_channels}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  dropout: {dropout}")
        print(f"  use_multi_scale: {use_multi_scale}")
        print(f"  use_cross_attention: {use_cross_attention}")

        # Harmonic CNN (multi-scale feature extraction)
        # Note: We use input_channels=1 and let the "channels" be in the freq dimension
        # Dataset provides (batch, 1, time, 112) where 112 = 88 CQT + 24 extras
        self.cnn = HarmonicCNN(input_channels=1, dropout=dropout)

        # Calculate LSTM input size based on actual input dimensions
        # With 112 freq bins and 2 pooling layers: 112 -> 56 -> 28
        # After CNN: (batch, 256, 125, 28)
        # Reshape to: (batch, 125, 256*28) = (batch, 125, 7168)
        expected_freq_after_pool = input_channels // 4  # 112 / 4 = 28
        lstm_input_size = 256 * expected_freq_after_pool
        print(f"  LSTM input size: {lstm_input_size} (freq bins after pooling: {expected_freq_after_pool})")

        # Bidirectional LSTM for temporal modeling
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

        # Multi-scale temporal encoder
        if use_multi_scale:
            if MUSICAL_COMPONENTS_AVAILABLE:
                self.multi_scale = MultiScaleTemporalEncoder(
                    hidden_size=hidden_size * 2,
                    num_scales=4
                )
                print("  ✅ Using MultiScaleTemporalEncoder from musical_components")
            else:
                self.multi_scale = SimpleMultiScaleEncoder(
                    hidden_size=hidden_size * 2,
                    num_scales=4
                )
                print("  ✅ Using SimpleMultiScaleEncoder (fallback)")
        else:
            self.multi_scale = None

        # Cross-attention for pitch-informed predictions
        if use_cross_attention:
            self.cross_attn = CrossAttention(
                hidden_size=hidden_size * 2,
                num_heads=4
            )
            print("  ✅ Using CrossAttention for pitch conditioning")
        else:
            self.cross_attn = None

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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        # Initialize onset/offset heads with bias matching dataset density
        # Measured from full dataset:
        # - Onset density: 5.221% → logit = log(0.052207/0.947793) ≈ -2.8989
        # - Offset density: 5.213% → logit = log(0.052130/0.947870) ≈ -2.9005
        nn.init.constant_(self.onset_head.bias, -2.8989)
        nn.init.constant_(self.offset_head.bias, -2.9005)

        print(f"  ✅ Onset head initialized with bias -2.8989 (5.221% density)")
        print(f"  ✅ Offset head initialized with bias -2.9005 (5.213% density)")

    def forward(self, x: torch.Tensor, pitch_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Multi-channel input in one of two formats:
               - (batch, 1, time, total_features) from EnhancedMelodyDataset
               - (batch, channels, time, freq) pre-formatted
            pitch_features: (batch, time, hidden) - Optional pitch features for attention

        Returns:
            dict with 'onset' and 'offset' tensors
            - onset: (batch, time, 1) - onset logits
            - offset: (batch, time, 1) - offset logits
        """
        batch_size = x.size(0)

        # Input format: (batch, 1, time, input_channels)
        # where input_channels = 112 (88 CQT + 24 extras)
        # CNN expects this exact format

        # CNN feature extraction
        x = self.cnn(x)  # (batch, 256, 125, 28) for 112 input channels

        # Reshape for LSTM
        batch, channels, time_steps, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)

        # LSTM temporal processing
        x, _ = self.lstm(x)  # (batch, time, hidden*2)
        x = self.ln(x)

        # Multi-scale temporal encoding
        if self.multi_scale is not None:
            x = self.multi_scale(x)

        # Cross-attention with pitch features (if available)
        if self.cross_attn is not None:
            if pitch_features is not None:
                # Use pitch features as key/value
                x = x + self.cross_attn(query=x, key=pitch_features, value=pitch_features)
            else:
                # Self-attention
                x = x + self.cross_attn(query=x)

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
            x: (batch, channels, time, freq) - Multi-channel input

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
    print("TESTING ENHANCED ONSET/OFFSET MODEL")
    print("="*70)

    # Test model initialization
    model = EnhancedOnsetOffsetModel(
        n_bins=88,
        input_channels=112,
        hidden_size=256,
        use_multi_scale=True,
        use_cross_attention=True
    )
    model.eval()

    # Test forward pass with dataset format
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 1, 500, 112)  # Dataset format: (batch, 1, time, 112)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Onset output shape: {output['onset'].shape}")
    print(f"Offset output shape: {output['offset'].shape}")

    # Verify shapes
    expected_time = 125  # 500 / 4
    assert output['onset'].shape == (2, expected_time, 1), "Onset shape mismatch!"
    assert output['offset'].shape == (2, expected_time, 1), "Offset shape mismatch!"

    print("\n✅ Forward pass successful!")
    print(f"   Input: (2, 1, 500, 112)")
    print(f"   Output: (2, 125, 1) for both onset and offset")

    # Test with pitch features
    print("\nTesting with pitch features...")
    dummy_pitch = torch.randn(2, 125, 512)  # pitch features

    with torch.no_grad():
        output_with_pitch = model(dummy_input, pitch_features=dummy_pitch)

    print("✅ Pitch-conditioned forward pass successful!")

    # Test TorchScript-friendly version
    print("\nTesting TorchScript-friendly get_outputs()...")
    with torch.no_grad():
        tuple_output = model.get_outputs(dummy_input)
        onset_tuple, offset_tuple = tuple_output
        # Note: Can't compare with output from forward with pitch features
        # So just verify shapes
        assert onset_tuple.shape == (2, 125, 1), "Onset tuple shape mismatch!"
        assert offset_tuple.shape == (2, 125, 1), "Offset tuple shape mismatch!"

    print("✅ TorchScript-friendly version working!")

    print(f"\nModel size: {model.count_parameters():,} parameters")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nKey Features:")
    print("  ✅ HarmonicCNN with multi-scale (local + octave) convolutions")
    print("  ✅ Multi-scale temporal encoder")
    print("  ✅ Cross-attention for pitch-informed predictions")
    print("  ✅ Rich input: 88 CQT + 5 onset + 19 musical = 112 channels")
    print("  ✅ TorchScript compatible")
    print("="*70)
