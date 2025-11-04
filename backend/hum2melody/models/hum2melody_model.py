"""
Hum2Melody Model - Complete Implementation (HPCC-FIXED)

Fixed for actual HPCC structure:
- /mnt/scratch/meadowm1/music-ai-training/models/musical_components.py
- /mnt/scratch/meadowm1/music-ai-training/models/hum2melody_model.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# HPCC-specific import fix
# Add parent directory to path so we can import from models/
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Try to import musical components
try:
    from models.musical_components import (
        MultiScaleTemporalEncoder,
        MusicalTransitionModel,
        ImprovedOnsetOffsetLoss
    )
    MUSICAL_COMPONENTS_AVAILABLE = True
    print("[hum2melody_model] ✅ Musical components available")
except ImportError as e:
    MUSICAL_COMPONENTS_AVAILABLE = False
    print(f"[hum2melody_model] ⚠️ Musical components not available: {e}")
    print("[hum2melody_model] Will use basic versions")


class HarmonicCNN(nn.Module):
    """CNN with dilated convolutions to capture harmonic relationships."""
    
    def __init__(self, input_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        
        print(f"[HarmonicCNN] Initializing with input_channels={input_channels}, dropout={dropout}")
        
        # Block 1: Multi-scale feature extraction
        self.conv1_local = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
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
        
        print("[HarmonicCNN] ✅ Initialized successfully")
    
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


class AdaptiveInputProjection(nn.Module):
    """Project variable-size input to fixed hidden size."""
    
    def __init__(self, input_channels: int, hidden_size: int = 256):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        
        print(f"[AdaptiveInputProjection] input_channels={input_channels}, hidden_size={hidden_size}")
        
        # Use 1D conv for efficiency
        self.projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print("[AdaptiveInputProjection] ✅ Initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, input_channels)
        Returns:
            (batch, time, hidden_size)
        """
        # Conv1d expects (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class ImprovedHum2MelodyWithOnsets(nn.Module):
    """Basic model with frame + onset heads."""
    
    def __init__(
        self,
        n_bins: int = 72,  # Changed default from 84 to 72
        hidden_size: int = 256,
        num_notes: int = 88,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.num_notes = num_notes
        self.use_attention = use_attention
        
        print(f"\n{'='*60}")
        print(f"IMPROVED MODEL INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  n_bins: {n_bins}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_notes: {num_notes}")
        print(f"  dropout: {dropout}")
        print(f"  use_attention: {use_attention}")
        
        # Harmonic CNN
        self.cnn = HarmonicCNN(input_channels=1, dropout=dropout)
        
        # Calculate LSTM input size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 500, n_bins)
            cnn_output = self.cnn(dummy_input)
            batch, channels, time_steps, freq = cnn_output.size()
            self.lstm_input_size = channels * freq
            print(f"  CNN output shape: {cnn_output.shape}")
            print(f"  LSTM input size: {self.lstm_input_size}")
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        
        # Optional attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.ln2 = nn.LayerNorm(hidden_size * 2)
        
        # Shared representation
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_notes)
        )
        
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self._init_weights()
        
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"{'='*60}\n")
    
    def _init_weights(self):
        """Initialize weights properly."""
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
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, time, n_bins) - CQT input
        
        Returns:
            dict with 'frame' and 'onset' outputs
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.cnn(x)
        
        # Reshape for LSTM
        batch, channels, time_steps, freq = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)
        
        # LSTM
        x, _ = self.lstm(x)
        x = self.ln1(x)
        
        # Attention
        if self.use_attention:
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out
            x = self.ln2(x)
        
        # Shared representation
        x = self.shared_fc(x)
        
        # Outputs
        return {
            'frame': self.frame_head(x),
            'onset': self.onset_head(x)
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedHum2MelodyModel(nn.Module):
    """Enhanced model with ALL improvements."""
    
    def __init__(
        self,
        n_bins: int = 72,  # Changed default from 84 to 72
        input_channels: int = None,
        hidden_size: int = 256,
        num_notes: int = 88,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_multi_scale: bool = True,
        use_transition_model: bool = True,
        transition_smoothing: float = 0.3,
        f0_min: float = 80.0,
        f0_max: float = 800.0
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.num_notes = num_notes
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale and MUSICAL_COMPONENTS_AVAILABLE
        self.use_transition_model = use_transition_model and MUSICAL_COMPONENTS_AVAILABLE
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        print(f"\n{'='*60}")
        print(f"ENHANCED MODEL INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  n_bins: {n_bins}")
        print(f"  input_channels: {input_channels if input_channels else 'auto-detect'}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_notes: {num_notes}")
        print(f"  dropout: {dropout}")
        print(f"  use_attention: {use_attention}")
        print(f"  use_multi_scale: {self.use_multi_scale}")
        print(f"  use_transition_model: {self.use_transition_model}")
        
        # Input handling
        self.input_channels = input_channels if input_channels else n_bins
        self.use_adaptive_input = (input_channels is None)
        
        if self.use_adaptive_input:
            print(f"  Using adaptive input projection")
            self.input_projection = None
        else:
            print(f"  Using fixed input channels: {self.input_channels}")
            if self.input_channels > n_bins:
                # Only project the EXTRA channels (not CQT)
                extra_channels = self.input_channels - n_bins
                print(f"  Creating projection for {extra_channels} extra channels")
                self.input_projection = AdaptiveInputProjection(
                    extra_channels,
                    hidden_size
                )
            else:
                self.input_projection = None
        
        # Shared feature extractor
        self.cnn = HarmonicCNN(input_channels=1, dropout=dropout)

        # Calculate LSTM input size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 500, n_bins)
            cnn_output = self.cnn(dummy_input)
            batch, channels, time_steps, freq = cnn_output.size()
            self.lstm_input_size = channels * freq
            print(f"  CNN output shape: {cnn_output.shape}")
            print(f"  LSTM input size: {self.lstm_input_size}")

        # Re-create input projection with correct output size (lstm_input_size)
        if self.input_projection is not None:
            extra_channels = self.input_channels - n_bins
            print(f"  Re-creating projection to match LSTM input size ({self.lstm_input_size})")
            self.input_projection = AdaptiveInputProjection(
                extra_channels,
                self.lstm_input_size  # Match CNN output size
            )
        
        # SHARED-LSTM ARCHITECTURE (like v1 - proven for pitch)
        # Single shared LSTM learns representations for ALL tasks
        # This preserves the spectral-temporal features that made v1's pitch work
        self.shared_lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Layer norm
        self.lstm_ln = nn.LayerNorm(hidden_size * 2)

        print(f"  ✅ SHARED-LSTM architecture (v1-style):")
        print(f"     - Shared LSTM: {hidden_size * 2} units (for ALL tasks)")
        print(f"     - Multiple heads read from same representation")
        print(f"     - Preserves pitch detection inductive bias from v1")

        # DISABLED: Attention complexity (can re-add later if needed)
        self.use_attention = False  # Force disable for simplicity
        
        # DISABLED: Multi-scale complexity (can re-add later if needed)
        self.use_multi_scale = False  # Force disable for simplicity

        # Single shared FC layer - all heads read from same representation
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        print(f"  ✅ Shared FC layer: All heads use same learned representation")

        # === OUTPUT HEADS ===
        
        # 1. Frame head (pitch classification)
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_notes)
        )
        
        # 2. Onset head (note starts)
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 3. Offset head (note ends)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 4. F0 head (continuous pitch + voicing)
        self.f0_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 2)
        )
        
        # Musical transition model
        if self.use_transition_model:
            print(f"  ✅ Adding musical transition model")
            self.transition_model = MusicalTransitionModel(
                num_notes=num_notes,
                smoothing_strength=transition_smoothing
            )
        
        self._init_weights()
        
        print(f"  Total parameters: {self.count_parameters():,}")
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
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        # CRITICAL: Initialize onset/offset heads with POSITIVE bias
        # This prevents them from learning to predict all zeros
        # Onsets are ~1-2% of frames, so starting with small positive bias
        # helps the model explore onset predictions early in training
        if hasattr(self, 'onset_head'):
            # Get the final layer of onset head
            final_layer = self.onset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                # Initialize with stronger positive bias (-1.0 in logits ≈ 0.27 in probability)
                # Stronger bias to help onset head maintain predictions during early training
                # Previous -2.0 bias wasn't strong enough to prevent suppression
                nn.init.constant_(final_layer.bias, -1.0)
                print(f"  ✅ Onset head initialized with stronger positive bias (-1.0)")

        if hasattr(self, 'offset_head'):
            final_layer = self.offset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, -2.5)  # Slightly lower than onset
                print(f"  ✅ Offset head initialized with positive bias")
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, time, channels)
               Can be CQT only or multi-channel input
        
        Returns:
            dict with 'frame', 'onset', 'offset', 'f0'
        """
        batch_size = x.size(0)
        
        # Handle adaptive input projection
        if self.use_adaptive_input and self.input_projection is None:
            input_channels = x.size(-1)
            print(f"[EnhancedModel] First forward - detected {input_channels} input channels")
            if input_channels > self.n_bins:
                # Only project the EXTRA channels (not CQT)
                extra_channels = input_channels - self.n_bins
                print(f"[EnhancedModel] Creating projection for {extra_channels} extra channels to {self.lstm_input_size}")
                self.input_projection = AdaptiveInputProjection(
                    extra_channels,
                    self.lstm_input_size  # Match CNN output size
                ).to(x.device)
        
        # Handle multi-channel input
        if x.size(-1) > self.n_bins:
            # Extract CQT and extra features
            cqt_input = x[:, :, :, :self.n_bins]
            extra_features = x[:, :, :, self.n_bins:]
            
            # Process CQT through CNN
            cnn_output = self.cnn(cqt_input)
            
            # Reshape CNN output
            batch, channels, time_steps, freq = cnn_output.size()
            cnn_flat = cnn_output.permute(0, 2, 1, 3).contiguous()
            cnn_flat = cnn_flat.view(batch_size, time_steps, channels * freq)
            
            # Process extra features
            if self.input_projection is not None and extra_features.size(-1) > 0:
                # AdaptiveInputProjection expects (batch, time, channels)
                extra_flat = extra_features.squeeze(1)  # (batch, time, channels)
                extra_projected = self.input_projection(extra_flat)  # Returns (batch, time, hidden)

                # Downsample to match CNN time dimension if needed
                if extra_projected.size(1) != time_steps:  # size(1) is time dimension
                    # Permute to (batch, hidden, time) for interpolation
                    extra_projected = extra_projected.permute(0, 2, 1)
                    extra_projected = F.interpolate(
                        extra_projected,
                        size=time_steps,
                        mode='linear',
                        align_corners=False
                    )
                    # Permute back to (batch, time, hidden)
                    extra_projected = extra_projected.permute(0, 2, 1)

                # Combine (both are now (batch, time, hidden))
                x = cnn_flat + extra_projected
            else:
                x = cnn_flat
        else:
            # Standard CQT input
            x = self.cnn(x)
            
            # Reshape for LSTM
            batch, channels, time_steps, freq = x.size()
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(batch_size, time_steps, channels * freq)
        
        # SHARED-LSTM ARCHITECTURE (v1-style): Single representation for all tasks
        # This preserves the pitch detection inductive bias that made v1 work

        # Single shared LSTM learns features for ALL tasks
        shared_features, _ = self.shared_lstm(x)
        shared_features = self.lstm_ln(shared_features)

        # Attention and multi-scale are DISABLED for simplicity (like v1)
        # Can re-enable later if needed after recovering pitch

        # Single shared FC layer - all heads read from same representation
        shared_repr = self.shared_fc(shared_features)

        # All output heads read from same shared representation
        # This allows multi-task learning without architectural separation
        frame_logits = self.frame_head(shared_repr)
        onset_logits = self.onset_head(shared_repr)
        offset_logits = self.offset_head(shared_repr)
        f0_logits = self.f0_head(shared_repr)

        # Apply musical transition smoothing to frame output
        if self.use_transition_model:
            frame_logits = self.transition_model(frame_logits)

        # Return all outputs
        return {
            'frame': frame_logits,
            'onset': onset_logits,
            'offset': offset_logits,
            'f0': f0_logits
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualPathHum2MelodyModel(nn.Module):
    """
    Dual-path architecture with independent harmonic and temporal processing.

    Architecture:
        Input → CNN Encoder (shared) → ┬→ Harmonic LSTM → Frame/F0 heads
                                        └→ Temporal LSTM → Onset/Offset heads

    This design eliminates gradient conflicts between pitch (harmonic) and timing (temporal)
    tasks by giving each path its own LSTM optimized for its specific task.

    Key features:
    - Harmonic path: Optimized for spectral smoothness (pitch detection)
    - Temporal path: Optimized for edge detection (onset/offset detection)
    - Can freeze harmonic path to preserve Stage-1 pitch performance
    - Loads Stage-1 weights into harmonic path via load_stage1_weights()
    """

    def __init__(
        self,
        n_bins: int = 88,
        input_channels: int = None,
        hidden_size: int = 256,
        num_notes: int = 88,
        dropout: float = 0.3,
        use_attention: bool = False,
        f0_min: float = 80.0,
        f0_max: float = 800.0
    ):
        super().__init__()

        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.num_notes = num_notes
        self.use_attention = use_attention
        self.f0_min = f0_min
        self.f0_max = f0_max

        print(f"\n{'='*60}")
        print(f"DUAL-PATH MODEL INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  n_bins: {n_bins}")
        print(f"  input_channels: {input_channels if input_channels else 'auto-detect'}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_notes: {num_notes}")
        print(f"  dropout: {dropout}")

        # Input handling
        self.input_channels = input_channels if input_channels else n_bins
        self.use_adaptive_input = (input_channels is None)

        if self.use_adaptive_input:
            print(f"  Using adaptive input projection")
            self.input_projection = None
        else:
            print(f"  Using fixed input channels: {self.input_channels}")
            if self.input_channels > n_bins:
                extra_channels = self.input_channels - n_bins
                print(f"  Creating projection for {extra_channels} extra channels")
                self.input_projection = AdaptiveInputProjection(
                    extra_channels,
                    hidden_size  # Will be recreated after CNN shape detection
                )
            else:
                self.input_projection = None

        # Shared CNN encoder (low-level feature extraction)
        self.cnn_encoder = HarmonicCNN(input_channels=1, dropout=dropout)

        # Calculate LSTM input size from CNN output
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 500, n_bins)
            cnn_output = self.cnn_encoder(dummy_input)
            batch, channels, time_steps, freq = cnn_output.size()
            self.lstm_input_size = channels * freq
            print(f"  CNN output shape: {cnn_output.shape}")
            print(f"  LSTM input size: {self.lstm_input_size}")

        # Re-create input projection with correct output size
        if self.input_projection is not None:
            extra_channels = self.input_channels - n_bins
            print(f"  Re-creating projection to match LSTM input size ({self.lstm_input_size})")
            self.input_projection = AdaptiveInputProjection(
                extra_channels,
                self.lstm_input_size
            )

        print(f"\n  ✅ DUAL-PATH ARCHITECTURE:")

        # === HARMONIC PATH (for pitch detection) ===
        print(f"     HARMONIC PATH (pitch detection):")
        self.harmonic_lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.harmonic_ln = nn.LayerNorm(hidden_size * 2)
        self.harmonic_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"       - Harmonic LSTM: {hidden_size * 2} units (bidirectional)")
        print(f"       - Optimized for: spectral smoothness, frequency correlation")

        # === TEMPORAL PATH (for onset/offset detection) ===
        print(f"     TEMPORAL PATH (onset/offset detection):")
        self.temporal_lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temporal_ln = nn.LayerNorm(hidden_size * 2)
        self.temporal_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"       - Temporal LSTM: {hidden_size * 2} units (bidirectional)")
        print(f"       - Optimized for: edge detection, temporal contrast")

        # === OUTPUT HEADS ===
        print(f"     OUTPUT HEADS:")

        # Harmonic path heads
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_notes)
        )
        print(f"       - Frame head: connected to harmonic path")

        self.f0_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 2)
        )
        print(f"       - F0 head: connected to harmonic path")

        # Temporal path heads
        self.onset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        print(f"       - Onset head: connected to temporal path")

        self.offset_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 4, 1)
        )
        print(f"       - Offset head: connected to temporal path")

        self._init_weights()

        total_params = self.count_parameters()
        harmonic_params = sum(p.numel() for n, p in self.named_parameters()
                             if p.requires_grad and ('harmonic' in n or 'frame' in n or 'f0' in n or 'cnn' in n))
        temporal_params = sum(p.numel() for n, p in self.named_parameters()
                             if p.requires_grad and ('temporal' in n or 'onset' in n or 'offset' in n))

        print(f"\n  Total parameters: {total_params:,}")
        print(f"    - Shared CNN: {total_params - harmonic_params - temporal_params:,}")
        print(f"    - Harmonic path: {harmonic_params:,}")
        print(f"    - Temporal path: {temporal_params:,}")
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
        # Measured from full dataset: onset=5.221%, offset=5.213%
        if hasattr(self, 'onset_head'):
            final_layer = self.onset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                # Measured: 5.221% density → logit = log(0.052207/0.947793) ≈ -2.8989
                nn.init.constant_(final_layer.bias, -2.8989)
                print(f"  ✅ Onset head initialized with bias -2.8989 (dataset density: 5.221%)")

        if hasattr(self, 'offset_head'):
            final_layer = self.offset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                # Measured: 5.213% density → logit = log(0.052130/0.947870) ≈ -2.9005
                nn.init.constant_(final_layer.bias, -2.9005)
                print(f"  ✅ Offset head initialized with bias -2.9005 (dataset density: 5.213%)")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through dual paths.

        Args:
            x: (batch, 1, time, channels) - CQT input or multi-channel

        Returns:
            dict with 'frame', 'onset', 'offset', 'f0'
        """
        batch_size = x.size(0)

        # Handle adaptive input projection
        if self.use_adaptive_input and self.input_projection is None:
            input_channels = x.size(-1)
            print(f"[DualPath] First forward - detected {input_channels} input channels")
            if input_channels > self.n_bins:
                extra_channels = input_channels - self.n_bins
                print(f"[DualPath] Creating projection for {extra_channels} extra channels to {self.lstm_input_size}")
                self.input_projection = AdaptiveInputProjection(
                    extra_channels,
                    self.lstm_input_size
                ).to(x.device)

        # === SHARED CNN ENCODER ===
        # Handle multi-channel input
        if x.size(-1) > self.n_bins:
            # Extract CQT and extra features
            cqt_input = x[:, :, :, :self.n_bins]
            extra_features = x[:, :, :, self.n_bins:]

            # Process CQT through CNN
            cnn_output = self.cnn_encoder(cqt_input)

            # Reshape CNN output
            batch, channels, time_steps, freq = cnn_output.size()
            cnn_flat = cnn_output.permute(0, 2, 1, 3).contiguous()
            cnn_flat = cnn_flat.view(batch_size, time_steps, channels * freq)

            # Process extra features
            if self.input_projection is not None and extra_features.size(-1) > 0:
                extra_flat = extra_features.squeeze(1)
                extra_projected = self.input_projection(extra_flat)

                # Downsample to match CNN time dimension if needed
                if extra_projected.size(1) != time_steps:
                    extra_projected = extra_projected.permute(0, 2, 1)
                    extra_projected = F.interpolate(
                        extra_projected,
                        size=time_steps,
                        mode='linear',
                        align_corners=False
                    )
                    extra_projected = extra_projected.permute(0, 2, 1)

                # Combine
                cnn_features = cnn_flat + extra_projected
            else:
                cnn_features = cnn_flat
        else:
            # Standard CQT input
            cnn_output = self.cnn_encoder(x)

            # Reshape for LSTM
            batch, channels, time_steps, freq = cnn_output.size()
            cnn_features = cnn_output.permute(0, 2, 1, 3)
            cnn_features = cnn_features.contiguous().view(batch_size, time_steps, channels * freq)

        # === DUAL PATH PROCESSING ===
        # Harmonic path: spectral smoothing for pitch
        harmonic_repr, _ = self.harmonic_lstm(cnn_features)
        harmonic_repr = self.harmonic_ln(harmonic_repr)
        harmonic_repr = self.harmonic_fc(harmonic_repr)

        # Temporal path: edge detection for onsets/offsets
        temporal_repr, _ = self.temporal_lstm(cnn_features)
        temporal_repr = self.temporal_ln(temporal_repr)
        temporal_repr = self.temporal_fc(temporal_repr)

        # === OUTPUT HEADS ===
        frame_logits = self.frame_head(harmonic_repr)
        f0_logits = self.f0_head(harmonic_repr)
        onset_logits = self.onset_head(temporal_repr)
        offset_logits = self.offset_head(temporal_repr)

        return {
            'frame': frame_logits,
            'onset': onset_logits,
            'offset': offset_logits,
            'f0': f0_logits
        }

    def freeze_harmonic_path(self):
        """
        Freeze all harmonic path parameters (CNN, input projection, harmonic LSTM, frame/f0 heads).

        CRITICAL: Also sets BatchNorm layers to eval() mode to prevent running statistics drift!
        """
        frozen_params = []
        for name, param in self.named_parameters():
            if any(key in name for key in ['cnn_encoder', 'input_projection', 'harmonic_lstm', 'harmonic_ln',
                                           'harmonic_fc', 'frame_head', 'f0_head']):
                param.requires_grad = False
                frozen_params.append(name)

        # CRITICAL: Set BatchNorm layers to eval() mode to freeze running statistics
        # Without this, BatchNorm running_mean/running_var will update during training,
        # causing model outputs to drift even with frozen weights (causes ~10% F1 drop!)
        # Store references so we can keep them in eval mode even after model.train()
        self._frozen_bn_modules = []
        frozen_bn_count = 0
        for name, module in self.named_modules():
            if any(key in name for key in ['cnn_encoder', 'input_projection', 'harmonic_lstm',
                                           'harmonic_fc', 'frame_head', 'f0_head']):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    module.eval()  # Force eval mode even during training
                    # Also freeze running statistics
                    if hasattr(module, 'track_running_stats'):
                        module.track_running_stats = False
                    self._frozen_bn_modules.append(module)
                    frozen_bn_count += 1

        print(f"\n🔒 FROZE HARMONIC PATH:")
        print(f"   Frozen {len(frozen_params)} parameter groups")
        print(f"   Frozen {frozen_bn_count} BatchNorm/LayerNorm layers (set to eval mode)")
        print(f"   Components: CNN encoder + Input projection + Harmonic LSTM + Frame/F0 heads")
        print(f"   ⚠️ This prevents BatchNorm drift that would degrade Frame F1!")
        return frozen_params

    def train(self, mode=True):
        """
        Override train() to keep frozen BatchNorm modules in eval mode.

        When model.train() is called, PyTorch recursively sets ALL modules to train mode,
        which would override our frozen BatchNorm eval() settings. We need to restore them.
        """
        super().train(mode)

        # Restore frozen BatchNorm modules to eval mode
        if mode and hasattr(self, '_frozen_bn_modules'):
            for module in self._frozen_bn_modules:
                module.eval()
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = False

        return self

    def freeze_temporal_path(self):
        """Freeze all temporal path parameters (temporal LSTM, onset/offset heads)."""
        frozen_params = []
        for name, param in self.named_parameters():
            if any(key in name for key in ['temporal_lstm', 'temporal_ln',
                                           'temporal_fc', 'onset_head', 'offset_head']):
                param.requires_grad = False
                frozen_params.append(name)

        print(f"\n🔒 FROZE TEMPORAL PATH:")
        print(f"   Frozen {len(frozen_params)} parameter groups")
        print(f"   Components: Temporal LSTM + Onset/Offset heads")
        return frozen_params

    def load_stage1_weights(self, checkpoint_path: str, device='cpu', strict_cnn=True):
        """
        Load Stage-1 (pitch-only) weights into harmonic path.

        Args:
            checkpoint_path: Path to Stage-1 checkpoint
            device: Device to load weights to
            strict_cnn: If True, freeze CNN exactly. If False, allow CNN to adapt.

        Returns:
            Dict with loading statistics
        """
        print(f"\n{'='*60}")
        print(f"LOADING STAGE-1 WEIGHTS INTO DUAL-PATH MODEL")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        stage1_state = ckpt['model_state_dict']
        model_state = self.state_dict()

        # Weight mapping: Stage-1 → Dual-Path
        # Stage-1 uses: cnn, shared_lstm, shared_fc, frame_head, f0_head
        # Dual-Path uses: cnn_encoder, harmonic_lstm, harmonic_fc, frame_head, f0_head
        weight_mapping = {
            'cnn.': 'cnn_encoder.',
            'shared_lstm.': 'harmonic_lstm.',
            'lstm_ln.': 'harmonic_ln.',
            'shared_fc.': 'harmonic_fc.',
            # frame_head and f0_head stay the same
        }

        copied = []
        skipped = []
        shape_mismatch = []

        for stage1_key, stage1_tensor in stage1_state.items():
            # Map Stage-1 key to Dual-Path key
            dual_key = stage1_key
            for old_prefix, new_prefix in weight_mapping.items():
                if dual_key.startswith(old_prefix):
                    dual_key = dual_key.replace(old_prefix, new_prefix, 1)
                    break

            # Copy if key exists and shapes match
            if dual_key in model_state:
                if model_state[dual_key].shape == stage1_tensor.shape:
                    model_state[dual_key].copy_(stage1_tensor)
                    copied.append((stage1_key, dual_key))
                else:
                    shape_mismatch.append((stage1_key, dual_key,
                                          model_state[dual_key].shape, stage1_tensor.shape))
            else:
                skipped.append(stage1_key)

        # Load the updated state dict
        self.load_state_dict(model_state)

        # Report
        print(f"\n✅ Weight Loading Summary:")
        print(f"   Copied: {len(copied)} parameters")
        print(f"   Skipped: {len(skipped)} parameters (onset/offset heads - will be trained)")
        if shape_mismatch:
            print(f"   ⚠️ Shape mismatch: {len(shape_mismatch)} parameters")

        if copied:
            print(f"\n   Sample copied weights:")
            for stage1_k, dual_k in copied[:5]:
                print(f"     {stage1_k} → {dual_k}")
            if len(copied) > 5:
                print(f"     ... and {len(copied) - 5} more")

        if shape_mismatch:
            print(f"\n   ⚠️ Shape mismatches:")
            for stage1_k, dual_k, model_shape, ckpt_shape in shape_mismatch[:3]:
                print(f"     {stage1_k}: model={model_shape}, ckpt={ckpt_shape}")

        # Verify by checking Frame F1 from checkpoint
        if 'val_metrics' in ckpt and 'frame_f1' in ckpt['val_metrics']:
            baseline_f1 = ckpt['val_metrics']['frame_f1']
            print(f"\n   Stage-1 baseline Frame F1: {baseline_f1:.4f}")
            print(f"   ⚠️ Validate on your data to confirm weights loaded correctly!")

        print(f"{'='*60}\n")

        return {
            'copied': len(copied),
            'skipped': len(skipped),
            'shape_mismatch': len(shape_mismatch),
            'baseline_frame_f1': ckpt.get('val_metrics', {}).get('frame_f1', None)
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Basic multi-task loss for frame + onset."""
    
    def __init__(
        self,
        frame_weight: float = 1.0,
        onset_weight: float = 0.5,
        mono_weight: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_monophonic: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        
        print(f"[MultiTaskLoss] Initializing:")
        print(f"  frame_weight: {frame_weight}")
        print(f"  onset_weight: {onset_weight}")
        print(f"  mono_weight: {mono_weight}")
        print(f"  use_monophonic: {use_monophonic}")
        
        self.frame_weight = frame_weight
        self.onset_weight = onset_weight
        self.mono_weight = mono_weight
        self.use_monophonic = use_monophonic
        
        # Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Onset loss
        self.onset_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(10.0, device=device)
        )
        
        print("[MultiTaskLoss] ✅ Initialized")
    
    def to(self, device):
        """Move loss to device."""
        super().to(device)
        self.onset_criterion.pos_weight = self.onset_criterion.pos_weight.to(device)
        return self
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for frame predictions."""
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()
    
    def monophonic_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Encourage only one pitch at a time."""
        probs = torch.sigmoid(predictions)
        
        eps = 1e-8
        probs_sum = probs.sum(dim=2, keepdim=True) + eps
        probs_norm = probs / probs_sum
        entropy = -(probs_norm * torch.log(probs_norm + eps)).sum(dim=2)
        
        return entropy.mean()
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        """Calculate multi-task loss."""
        losses = {}
        
        # Frame loss
        losses['frame'] = self.focal_loss(predictions['frame'], targets['frame'])
        
        # Onset loss
        onset_pred = predictions['onset'].squeeze(-1) if predictions['onset'].dim() == 3 else predictions['onset']
        losses['onset'] = self.onset_criterion(onset_pred, targets['onset'])
        
        # Total
        losses['total'] = (
            self.frame_weight * losses['frame'] +
            self.onset_weight * losses['onset']
        )
        
        # Optional monophonic constraint
        if self.use_monophonic:
            losses['mono'] = self.monophonic_loss(predictions['frame'])
            losses['total'] = losses['total'] + self.mono_weight * losses['mono']
        
        return losses


class EnhancedMultiTaskLoss(nn.Module):
    """Enhanced loss for all 4 heads with improved onset/offset handling."""

    def __init__(
        self,
        frame_weight: float = 1.0,  # Balanced with dual-LSTM architecture
        onset_weight: float = 1.0,  # REDUCED from 5.0 - dual-LSTM eliminates need for aggressive weighting
        offset_weight: float = 0.5,
        f0_weight: float = 1.0,
        mono_weight: float = 0.02,
        sparsity_weight: float = 0.02,  # Phase 2: prevent predict-everywhere-high
        temporal_loss_scale: float = 0.1,  # NEW: Direct gradient dampening for temporal path (onset/offset)
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_monophonic: bool = True,
        use_improved_onset_loss: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()

        print(f"[EnhancedMultiTaskLoss] Initializing:")
        print(f"  frame_weight: {frame_weight} (balanced for multi-task)")
        print(f"  onset_weight: {onset_weight} (increased to prevent onset head death)")
        print(f"  offset_weight: {offset_weight}")
        print(f"  f0_weight: {f0_weight}")
        print(f"  mono_weight: {mono_weight}")
        print(f"  sparsity_weight: {sparsity_weight} (Phase 2: prevent predict-everywhere-high)")
        print(f"  temporal_loss_scale: {temporal_loss_scale} (direct gradient dampening for temporal path)")
        print(f"  use_improved_onset_loss: {use_improved_onset_loss}")

        self.frame_weight = frame_weight
        self.onset_weight = onset_weight
        self.offset_weight = offset_weight
        self.f0_weight = f0_weight
        self.mono_weight = mono_weight
        self.temporal_loss_scale = temporal_loss_scale
        self.use_monophonic = use_monophonic
        self.use_improved_onset_loss = use_improved_onset_loss and MUSICAL_COMPONENTS_AVAILABLE

        # Store initial weights for warmup scheduler
        self.initial_frame_weight = frame_weight
        
        # Focal loss params
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if self.use_improved_onset_loss:
            print("  ✅ Using improved onset/offset loss with sparsity regularization")
            self.onset_offset_criterion = ImprovedOnsetOffsetLoss(
                # Use improved defaults from ImprovedOnsetOffsetLoss:
                # onset_weight=20.0, offset_weight=10.0,
                # consistency_weight=0.1, pairing_weight=0.05
                sparsity_weight=sparsity_weight,  # Phase 2: prevent predict-everywhere-high
                device=device
            )
        else:
            print("  ⚠️ Using basic BCE for onset/offset")
            self.onset_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(10.0, device=device)
            )
            self.offset_criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(10.0, device=device)
            )
        
        # F0 loss
        self.f0_criterion = nn.MSELoss()
        
        print("[EnhancedMultiTaskLoss] ✅ Initialized")
    
    def to(self, device):
        """Move loss to device."""
        super().to(device)
        if hasattr(self, 'onset_criterion'):
            self.onset_criterion.pos_weight = self.onset_criterion.pos_weight.to(device)
            self.offset_criterion.pos_weight = self.offset_criterion.pos_weight.to(device)
        return self
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for frame predictions."""
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_weight = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()
    
    def monophonic_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Encourage single pitch per frame."""
        probs = torch.sigmoid(predictions)
        
        eps = 1e-8
        probs_sum = probs.sum(dim=2, keepdim=True) + eps
        probs_norm = probs / probs_sum
        entropy = -(probs_norm * torch.log(probs_norm + eps)).sum(dim=2)
        
        return entropy.mean()
    
    def f0_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss for continuous f0 regression."""
        pred_log_f0 = predictions[:, :, 0]
        pred_voicing = predictions[:, :, 1]
        
        target_log_f0 = targets[:, :, 0]
        target_voicing = targets[:, :, 1]
        
        # Voicing loss
        voicing_loss = F.binary_cross_entropy_with_logits(
            pred_voicing,
            target_voicing,
            reduction='mean'
        )
        
        # F0 loss (only where voiced)
        voiced_mask = target_voicing > 0.5
        
        if voiced_mask.sum() > 0:
            f0_mse = F.mse_loss(
                pred_log_f0[voiced_mask],
                target_log_f0[voiced_mask],
                reduction='mean'
            )
        else:
            f0_mse = torch.tensor(0.0, device=predictions.device)
        
        return voicing_loss + f0_mse

    def set_frame_weight(self, weight: float):
        """
        Update frame_weight dynamically for warmup scheduler.

        Args:
            weight: New frame weight (typically 0.0 during warmup, then ramp to initial_frame_weight)
        """
        self.frame_weight = weight

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Calculate all losses."""
        losses = {}
        
        # Frame loss
        losses['frame'] = self.focal_loss(predictions['frame'], targets['frame'])
        
        # Onset/Offset loss
        if self.use_improved_onset_loss:
            onset_offset_losses = self.onset_offset_criterion(
                predictions['onset'],
                predictions['offset'],
                targets['onset'],
                targets['offset']
            )
            losses['onset'] = onset_offset_losses['total']
            losses['onset_consistency'] = onset_offset_losses['consistency']
            losses['onset_pairing'] = onset_offset_losses['pairing']
            # Phase 2: Propagate sparsity monitoring
            losses['sparsity'] = onset_offset_losses['sparsity']
            losses['mean_onset_pred'] = onset_offset_losses['mean_onset_pred']
        else:
            onset_pred = predictions['onset'].squeeze(-1) if predictions['onset'].dim() == 3 else predictions['onset']
            offset_pred = predictions['offset'].squeeze(-1) if predictions['offset'].dim() == 3 else predictions['offset']
            
            losses['onset'] = self.onset_criterion(onset_pred, targets['onset'])
            losses['offset'] = self.offset_criterion(offset_pred, targets['offset'])
        
        # F0 loss
        if 'f0' in predictions and 'f0' in targets:
            losses['f0'] = self.f0_loss(predictions['f0'], targets['f0'])
        else:
            losses['f0'] = torch.tensor(0.0, device=predictions['frame'].device)
        
        # Monophonic loss
        if self.use_monophonic:
            losses['mono'] = self.monophonic_loss(predictions['frame'])
        
        # Total
        # Apply temporal_loss_scale to onset/offset to dampen temporal path gradients
        losses['total'] = (
            self.frame_weight * losses['frame'] +
            self.temporal_loss_scale * self.onset_weight * losses['onset'] +
            self.f0_weight * losses['f0']
        )

        if 'offset' in losses and not self.use_improved_onset_loss:
            losses['total'] += self.temporal_loss_scale * self.offset_weight * losses['offset']

        if self.use_monophonic:
            losses['total'] += self.mono_weight * losses['mono']
        
        return losses


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING HUM2MELODY MODELS")
    print("="*70)
    
    # Test basic model
    print("\n1. Testing ImprovedHum2MelodyWithOnsets...")
    model1 = ImprovedHum2MelodyWithOnsets(n_bins=84)
    dummy_input = torch.randn(2, 1, 500, 84)
    output1 = model1(dummy_input)
    print(f"   Output shapes: frame={output1['frame'].shape}, onset={output1['onset'].shape}")
    print("   ✅ Basic model working")
    
    # Test enhanced model
    print("\n2. Testing EnhancedHum2MelodyModel...")
    model2 = EnhancedHum2MelodyModel(
        n_bins=84,
        input_channels=108,
        use_multi_scale=True,
        use_transition_model=True
    )
    dummy_input2 = torch.randn(2, 1, 500, 108)
    output2 = model2(dummy_input2)
    print(f"   Output shapes:")
    for k, v in output2.items():
        print(f"     {k}: {v.shape}")
    print("   ✅ Enhanced model working")
    
    # Test losses
    print("\n3. Testing MultiTaskLoss...")
    criterion1 = MultiTaskLoss()
    targets1 = {
        'frame': torch.zeros(2, 125, 88),
        'onset': torch.zeros(2, 125)
    }
    targets1['frame'][:, 10:20, 40:45] = 1.0
    targets1['onset'][:, [10, 50, 90]] = 1.0
    losses1 = criterion1(output1, targets1)
    print(f"   Losses: {list(losses1.keys())}")
    print("   ✅ Basic loss working")
    
    print("\n4. Testing EnhancedMultiTaskLoss...")
    criterion2 = EnhancedMultiTaskLoss(use_improved_onset_loss=True)
    targets2 = {
        'frame': torch.zeros(2, 125, 88),
        'onset': torch.zeros(2, 125),
        'offset': torch.zeros(2, 125),
        'f0': torch.zeros(2, 125, 2)
    }
    targets2['frame'][:, 10:20, 40:45] = 1.0
    targets2['onset'][:, [10, 50, 90]] = 1.0
    targets2['offset'][:, [20, 60, 100]] = 1.0
    targets2['f0'][:, 10:30, 0] = 5.0
    targets2['f0'][:, 10:30, 1] = 1.0
    losses2 = criterion2(output2, targets2)
    print(f"   Losses: {list(losses2.keys())}")
    print("   ✅ Enhanced loss working")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
