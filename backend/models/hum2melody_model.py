"""
Hum2Melody Model - Complete Implementation

Production model with frame, onset, offset, and f0 prediction heads.
This is the canonical melody transcription model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Try to import musical components
try:
    from .musical_components import (
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


class Hum2MelodyModel(nn.Module):
    """
    Canonical Hum2Melody model with all output heads.

    Outputs:
        - frame: Frame-level pitch activations (88 notes)
        - onset: Note onset detection
        - offset: Note offset detection
        - f0: Continuous F0 + voicing prediction
    """

    def __init__(
        self,
        n_bins: int = 88,  # CQT bins matching MIDI range
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
        print(f"HUM2MELODY MODEL INITIALIZATION")
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

        # SHARED-LSTM ARCHITECTURE
        # Single shared LSTM learns representations for ALL tasks
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

        print(f"  ✅ SHARED-LSTM architecture:")
        print(f"     - Shared LSTM: {hidden_size * 2} units (for ALL tasks)")
        print(f"     - Multiple heads read from same representation")

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
        if hasattr(self, 'onset_head'):
            final_layer = self.onset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, -1.0)
                print(f"  ✅ Onset head initialized with positive bias (-1.0)")

        if hasattr(self, 'offset_head'):
            final_layer = self.offset_head[-1]
            if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, -2.5)
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
            print(f"[Hum2MelodyModel] First forward - detected {input_channels} input channels")
            if input_channels > self.n_bins:
                # Only project the EXTRA channels (not CQT)
                extra_channels = input_channels - self.n_bins
                print(f"[Hum2MelodyModel] Creating projection for {extra_channels} extra channels to {self.lstm_input_size}")
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

        # SHARED-LSTM ARCHITECTURE: Single representation for all tasks

        # Single shared LSTM learns features for ALL tasks
        shared_features, _ = self.shared_lstm(x)
        shared_features = self.lstm_ln(shared_features)

        # Single shared FC layer - all heads read from same representation
        shared_repr = self.shared_fc(shared_features)

        # All output heads read from same shared representation
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
    """Full multi-task loss for all 4 output heads."""

    def __init__(
        self,
        frame_weight: float = 1.0,
        onset_weight: float = 1.0,
        offset_weight: float = 0.5,
        f0_weight: float = 1.0,
        mono_weight: float = 0.02,
        sparsity_weight: float = 0.02,
        temporal_loss_scale: float = 0.1,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_monophonic: bool = True,
        use_improved_onset_loss: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()

        print(f"[EnhancedMultiTaskLoss] Initializing:")
        print(f"  frame_weight: {frame_weight}")
        print(f"  onset_weight: {onset_weight}")
        print(f"  offset_weight: {offset_weight}")
        print(f"  f0_weight: {f0_weight}")
        print(f"  mono_weight: {mono_weight}")
        print(f"  sparsity_weight: {sparsity_weight}")
        print(f"  temporal_loss_scale: {temporal_loss_scale}")
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
                sparsity_weight=sparsity_weight,
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
        """Update frame_weight dynamically for warmup scheduler."""
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
    print("TESTING HUM2MELODY MODEL")
    print("="*70)

    # Test model
    print("\nTesting Hum2MelodyModel...")
    model = Hum2MelodyModel(
        n_bins=88,
        input_channels=None,
        use_multi_scale=False,
        use_transition_model=False
    )
    dummy_input = torch.randn(2, 1, 500, 88)
    output = model(dummy_input)
    print(f"   Output shapes:")
    for k, v in output.items():
        print(f"     {k}: {v.shape}")
    print("   ✅ Model working")

    # Test losses
    print("\nTesting EnhancedMultiTaskLoss...")
    criterion = EnhancedMultiTaskLoss(use_improved_onset_loss=False)
    targets = {
        'frame': torch.zeros(2, 125, 88),
        'onset': torch.zeros(2, 125),
        'offset': torch.zeros(2, 125),
        'f0': torch.zeros(2, 125, 2)
    }
    targets['frame'][:, 10:20, 40:45] = 1.0
    targets['onset'][:, [10, 50, 90]] = 1.0
    targets['offset'][:, [20, 60, 100]] = 1.0
    targets['f0'][:, 10:30, 0] = 5.0
    targets['f0'][:, 10:30, 1] = 1.0
    losses = criterion(output, targets)
    print(f"   Losses: {list(losses.keys())}")
    print("   ✅ Loss working")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
