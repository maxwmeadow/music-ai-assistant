"""
Combined Model Loader

Loads a combined checkpoint (created by create_combined_checkpoint.py)
and reconstructs the full combined model.

This provides a simple API for loading and using the single-file model.

Usage:
    from models.combined_model_loader import load_combined_model

    model = load_combined_model('combined_hum2melody_full.pth', device='cuda')
    model.eval()

    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt, extras)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from pathlib import Path

from .hum2melody_model import Hum2MelodyModel
from .onset_model import OnsetOffsetModel
from .enhanced_onset_model import EnhancedOnsetOffsetModel


class CombinedModelFromCheckpoint(nn.Module):
    """
    Combined model loaded from a single checkpoint file.

    This wraps both pitch and onset models and provides the same
    interface as CombinedHum2MelodyModel.
    """

    def __init__(
        self,
        checkpoint: Dict,
        device: str = 'cpu'
    ):
        super().__init__()

        self.device = torch.device(device)

        print(f"\n{'='*70}")
        print(f"LOADING COMBINED MODEL FROM SINGLE CHECKPOINT")
        print(f"{'='*70}")
        print(f"Combined model version: {checkpoint.get('combined_model_version', 'unknown')}")
        print(f"Device: {self.device}")

        # Store metadata
        self.preprocessing = checkpoint['preprocessing']
        self.architecture = checkpoint['architecture']

        # Load pitch model
        print(f"\nLoading pitch model...")
        self.pitch_model = self._load_pitch_model(checkpoint)
        self.pitch_model = self.pitch_model.to(self.device)
        self.pitch_model.eval()
        print(f"  ✅ Pitch model loaded")

        # Load onset model
        print(f"\nLoading onset model...")
        self.onset_model, self.onset_is_enhanced = self._load_onset_model(checkpoint)
        self.onset_model = self.onset_model.to(self.device)
        self.onset_model.eval()
        print(f"  ✅ Onset model loaded")

        # Freeze for inference
        for param in self.parameters():
            param.requires_grad = False

        # Count parameters
        pitch_params = sum(p.numel() for p in self.pitch_model.parameters())
        onset_params = sum(p.numel() for p in self.onset_model.parameters())
        total_params = pitch_params + onset_params

        print(f"\n✅ Combined model loaded successfully")
        print(f"   Pitch model: {pitch_params:,} params")
        print(f"   Onset model: {onset_params:,} params")
        print(f"   Total: {total_params:,} params")
        print(f"   Expected output frames: {self.preprocessing['expected_output_frames']}")
        print(f"{'='*70}\n")

    def _load_pitch_model(self, checkpoint):
        """Load pitch model from combined checkpoint."""
        arch = checkpoint['architecture']['pitch_model']

        # Create model
        model = Hum2MelodyModel(
            n_bins=arch['n_bins'],
            input_channels=arch['input_channels'],
            hidden_size=arch['hidden_size'],
            use_attention=True,
            use_multi_scale=False,
            use_transition_model=False
        )

        # Extract pitch model weights (remove 'pitch_model.' prefix)
        pitch_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('pitch_model.'):
                new_key = key.replace('pitch_model.', '')
                pitch_state_dict[new_key] = value

        # Load weights (allow missing keys for backward compatibility)
        missing, unexpected = model.load_state_dict(pitch_state_dict, strict=False)

        if missing:
            print(f"     ⚠️  Missing keys: {len(missing)} (likely newer model version)")
        if unexpected:
            print(f"     ⚠️  Unexpected keys: {len(unexpected)}")

        return model

    def _load_onset_model(self, checkpoint):
        """Load onset model from combined checkpoint."""
        arch = checkpoint['architecture']['onset_model']
        model_class = arch['model_class']

        # Detect model type from checkpoint
        if model_class == 'EnhancedOnsetOffsetModel':
            print(f"     Detected enhanced onset model with {arch['input_channels']} input channels")
            model = EnhancedOnsetOffsetModel(
                n_bins=arch['n_bins'],
                input_channels=arch['input_channels'],
                hidden_size=arch['hidden_size'],
                dropout=0.3,
                use_multi_scale=arch.get('use_multi_scale', True),
                use_cross_attention=arch.get('use_cross_attention', True)
            )
            is_enhanced = True
        else:
            print(f"     Loading basic onset model (CQT-only)")
            model = OnsetOffsetModel(
                n_bins=arch['n_bins'],
                hidden_size=arch['hidden_size'],
                dropout=0.3
            )
            is_enhanced = False

        # Extract onset model weights (remove 'onset_model.' prefix)
        onset_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('onset_model.'):
                new_key = key.replace('onset_model.', '')
                onset_state_dict[new_key] = value

        # Load weights
        model.load_state_dict(onset_state_dict)

        return model, is_enhanced

    def forward(
        self,
        cqt: torch.Tensor,
        extras: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            cqt: (batch, 1, time, 88) - CQT spectrogram
            extras: (batch, 1, time, 24) - onset features + musical context (optional)

        Returns:
            tuple: (frame, onset, offset, f0)
                - frame: (batch, time, 88) - pitch classification logits
                - onset: (batch, time, 1) - onset detection logits
                - offset: (batch, time, 1) - offset detection logits
                - f0: (batch, time, 2) - [f0_value, voicing]
        """
        # Build pitch model input (112 channels: 88 CQT + 24 extras)
        pitch_input = self._build_pitch_input(cqt, extras)

        # Run pitch model
        pitch_out = self.pitch_model(pitch_input)

        # Run onset model (use full input if enhanced, CQT-only if basic)
        if self.onset_is_enhanced:
            onset_out = self.onset_model(pitch_input)
        else:
            onset_out = self.onset_model(cqt)

        # Verify frame alignment
        pitch_frames = pitch_out['frame'].shape[1]
        onset_frames = onset_out['onset'].shape[1]

        if pitch_frames != onset_frames:
            raise RuntimeError(
                f"Frame alignment mismatch! "
                f"Pitch: {pitch_frames}, Onset: {onset_frames}"
            )

        # Return tuple
        return (
            pitch_out['frame'],
            onset_out['onset'],
            onset_out['offset'],
            pitch_out['f0']
        )

    def _build_pitch_input(
        self,
        cqt: torch.Tensor,
        extras: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Build 112-channel input for pitch model."""
        if extras is None:
            batch, c, time, freq = cqt.shape
            extras = torch.zeros(
                batch, c, time, 24,
                dtype=cqt.dtype,
                device=cqt.device
            )

        return torch.cat([cqt, extras], dim=-1)

    def count_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_preprocessing_info(self):
        """Get preprocessing parameters."""
        return self.preprocessing.copy()

    def get_architecture_info(self):
        """Get architecture information."""
        return self.architecture.copy()


def load_combined_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a combined model from a single checkpoint file.

    Args:
        checkpoint_path: Path to combined checkpoint (.pth file)
        device: Device to load on ('cpu' or 'cuda')

    Returns:
        CombinedModelFromCheckpoint instance ready for inference

    Example:
        >>> model = load_combined_model('combined_hum2melody_full.pth', device='cuda')
        >>> model.eval()
        >>> with torch.no_grad():
        ...     frame, onset, offset, f0 = model(cqt, extras)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Verify it's a combined checkpoint
    if 'combined_model_version' not in checkpoint:
        raise ValueError(
            f"Not a valid combined checkpoint file!\n"
            f"  File: {checkpoint_path}\n"
            f"  Expected 'combined_model_version' key"
        )

    model = CombinedModelFromCheckpoint(checkpoint, device=device)
    return model


if __name__ == '__main__':
    # Test loading
    print("\nTesting combined model loader...")

    checkpoint_path = 'combined_hum2melody_full.pth'
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Run: python scripts/create_combined_checkpoint.py")
        exit(1)

    # Load model
    model = load_combined_model(checkpoint_path, device='cpu')

    # Test forward pass
    print("\nTesting forward pass...")
    test_cqt = torch.randn(2, 1, 500, 88)
    test_extras = torch.randn(2, 1, 500, 24)

    with torch.no_grad():
        frame, onset, offset, f0 = model(test_cqt, test_extras)

    print(f"  Input CQT: {test_cqt.shape}")
    print(f"  Input extras: {test_extras.shape}")
    print(f"  Output frame: {frame.shape}")
    print(f"  Output onset: {onset.shape}")
    print(f"  Output offset: {offset.shape}")
    print(f"  Output f0: {f0.shape}")

    expected_time = 125
    assert frame.shape == (2, expected_time, 88), "Frame shape mismatch!"
    assert onset.shape == (2, expected_time, 1), "Onset shape mismatch!"
    assert offset.shape == (2, expected_time, 1), "Offset shape mismatch!"
    assert f0.shape == (2, expected_time, 2), "F0 shape mismatch!"

    print(f"\n✅ All tests passed!")
    print(f"   Combined model works correctly")
