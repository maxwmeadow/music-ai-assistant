"""
Combined Hum2Melody Model

Wrapper that combines pitch detection and onset/offset detection models
into a single deployable artifact.

Design:
- Loads two separate trained models (pitch + onset)
- Verifies preprocessing compatibility at load time
- TorchScript-compatible (tuple inputs/outputs, fixed control flow)
- Frozen for inference (no training)
- Single forward() returns all outputs (frame, onset, offset, f0)

Usage:
    # Create wrapper
    model = CombinedHum2MelodyModel(
        pitch_ckpt_path='checkpoints_fixed_data/best_model_with_metadata.pth',
        onset_ckpt_path='checkpoints_onset/best_model.pth'
    )

    # Run inference
    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt_input, extras_input)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

# Import model architectures
from models.hum2melody_model import EnhancedHum2MelodyModel
from models.onset_model import OnsetOffsetModel
from models.enhanced_onset_model import EnhancedOnsetOffsetModel


class CombinedHum2MelodyModel(nn.Module):
    """
    Combined model wrapping pitch + onset/offset models.

    TorchScript-compatible:
    - Tuple inputs/outputs (not dicts)
    - Fixed control flow
    - No tensor-dependent branching
    """

    def __init__(
        self,
        pitch_ckpt_path: str,
        onset_ckpt_path: str,
        device: str = 'cpu'
    ):
        super().__init__()

        print(f"\n{'='*70}")
        print(f"COMBINED HUM2MELODY MODEL INITIALIZATION")
        print(f"{'='*70}")
        print(f"Pitch checkpoint: {pitch_ckpt_path}")
        print(f"Onset checkpoint: {onset_ckpt_path}")
        print(f"Device: {device}")

        self.device = torch.device(device)

        # Load checkpoints
        pitch_ckpt = torch.load(pitch_ckpt_path, map_location=self.device, weights_only=False)
        onset_ckpt = torch.load(onset_ckpt_path, map_location=self.device, weights_only=False)

        # Verify preprocessing compatibility
        self._verify_preprocessing(pitch_ckpt, onset_ckpt)

        # Store metadata
        self.preprocessing = pitch_ckpt['preprocessing']
        self.architecture = {
            'pitch': pitch_ckpt['architecture'],
            'onset': onset_ckpt['architecture']
        }

        # Load models
        self.pitch_model = self._load_pitch_model(pitch_ckpt, self.device)
        self.onset_model = self._load_onset_model(onset_ckpt, self.device)

        # Freeze both models for inference
        self._freeze_model(self.pitch_model, "Pitch")
        self._freeze_model(self.onset_model, "Onset")

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

    def _verify_preprocessing(self, pitch_ckpt, onset_ckpt):
        """Verify preprocessing compatibility between models."""
        print("\nVerifying preprocessing compatibility...")

        pitch_prep = pitch_ckpt['preprocessing']
        onset_prep = onset_ckpt['preprocessing']

        # Critical checks
        checks = [
            ('sr', 'Sample rate'),
            ('hop_length', 'Hop length'),
            ('n_bins', 'CQT bins'),
            ('target_frames', 'Target frames'),
            ('expected_output_frames', 'Output frames'),
            ('cnn_downsample_factor', 'CNN downsample factor'),
        ]

        for key, name in checks:
            pitch_val = pitch_prep.get(key)
            onset_val = onset_prep.get(key)

            if pitch_val != onset_val:
                raise ValueError(
                    f"{name} mismatch!\n"
                    f"  Pitch model: {pitch_val}\n"
                    f"  Onset model: {onset_val}\n"
                    f"  These models were trained with incompatible preprocessing."
                )

            print(f"  ✅ {name}: {pitch_val}")

        print("✅ All preprocessing parameters match!")

    def _load_pitch_model(self, checkpoint, device):
        """Load pitch detection model from checkpoint."""
        print("\nLoading pitch model...")

        arch = checkpoint['architecture']

        # Create model instance
        model = EnhancedHum2MelodyModel(
            n_bins=arch['n_bins'],
            input_channels=arch['input_channels'],
            hidden_size=arch['hidden_size'],
            use_attention=True,
            use_multi_scale=False,
            use_transition_model=False
        )

        # Load weights (allow missing keys for backward compatibility)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if missing_keys:
            print(f"  ⚠️  Missing keys (likely from newer model version): {len(missing_keys)} keys")
            # This is expected for AdaptiveInputProjection in newer code versions
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {unexpected_keys}")

        model = model.to(device)

        print(f"  ✅ Pitch model loaded ({arch['model_class']})")
        return model

    def _load_onset_model(self, checkpoint, device):
        """Load onset/offset detection model from checkpoint."""
        print("\nLoading onset model...")

        arch = checkpoint['architecture']
        model_class = arch['model_class']

        # Detect model type and create appropriate instance
        if model_class == 'EnhancedOnsetOffsetModel':
            print(f"  Detected enhanced onset model with {arch['input_channels']} input channels")
            model = EnhancedOnsetOffsetModel(
                n_bins=arch['n_bins'],
                input_channels=arch['input_channels'],
                hidden_size=arch['hidden_size'],
                dropout=0.3,  # Not used in eval mode
                use_multi_scale=arch.get('use_multi_scale', True),
                use_cross_attention=arch.get('use_cross_attention', True)
            )
            self.onset_is_enhanced = True
        else:
            print(f"  Detected basic onset model (CQT-only)")
            model = OnsetOffsetModel(
                n_bins=arch['n_bins'],
                hidden_size=arch['hidden_size'],
                dropout=0.3  # Not used in eval mode
            )
            self.onset_is_enhanced = False

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        print(f"  ✅ Onset model loaded ({model_class})")
        return model

    def _freeze_model(self, model, name):
        """Freeze model for inference (no gradients, BN in eval mode)."""
        print(f"\nFreezing {name} model...")

        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # CRITICAL: Force all BN/LN layers to eval mode and disable tracking
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = False

        print(f"  ✅ {name} model frozen")

    @torch.jit.export
    def forward(
        self,
        cqt: torch.Tensor,
        extras: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TorchScript-compatible forward pass.

        Args:
            cqt: (batch, 1, time, 88) - CQT spectrogram
            extras: (batch, 1, time, 24) - onset features + musical context (optional)
                    If None, zeros will be used

        Returns:
            tuple: (frame, onset, offset, f0)
                - frame: (batch, time, 88) - pitch classification logits
                - onset: (batch, time, 1) - onset detection logits
                - offset: (batch, time, 1) - offset detection logits
                - f0: (batch, time, 2) - [f0_value, voicing] continuous pitch
        """
        # Build pitch model input (112 channels: 88 CQT + 24 extras)
        pitch_input = self._build_pitch_input(cqt, extras)

        # Run pitch model
        pitch_out = self.pitch_model(pitch_input)

        # Run onset model (use full input if enhanced, CQT-only if basic)
        if self.onset_is_enhanced:
            # Enhanced onset model needs same 112-channel input as pitch model
            onset_out = self.onset_model(pitch_input)
        else:
            # Basic onset model only uses CQT
            onset_out = self.onset_model(cqt)

        # Verify frame alignment (sanity check)
        # This should never fail if preprocessing metadata was verified correctly
        pitch_frames = pitch_out['frame'].shape[1]
        onset_frames = onset_out['onset'].shape[1]

        if pitch_frames != onset_frames:
            raise RuntimeError(
                f"Frame alignment mismatch! "
                f"Pitch: {pitch_frames}, Onset: {onset_frames}"
            )

        # Return tuple (TorchScript-friendly)
        return (
            pitch_out['frame'],     # (batch, time, 88)
            onset_out['onset'],     # (batch, time, 1)
            onset_out['offset'],    # (batch, time, 1)
            pitch_out['f0']         # (batch, time, 2)
        )

    def _build_pitch_input(
        self,
        cqt: torch.Tensor,
        extras: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Build 112-channel input for pitch model.

        Args:
            cqt: (batch, 1, time, 88)
            extras: (batch, 1, time, 24) or None

        Returns:
            pitch_input: (batch, 1, time, 112)
        """
        if extras is None:
            # Create zero-filled extras
            batch, c, time, freq = cqt.shape
            extras = torch.zeros(
                batch, c, time, 24,
                dtype=cqt.dtype,
                device=cqt.device
            )

        # Concatenate along feature dimension
        return torch.cat([cqt, extras], dim=-1)

    def count_parameters(self):
        """Count total parameters in combined model."""
        return (
            sum(p.numel() for p in self.pitch_model.parameters()) +
            sum(p.numel() for p in self.onset_model.parameters())
        )

    def get_preprocessing_info(self):
        """Get preprocessing parameters for deployment."""
        return self.preprocessing.copy()

    def get_architecture_info(self):
        """Get architecture information."""
        return self.architecture.copy()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING COMBINED MODEL")
    print("="*70)

    # Test model loading
    print("\n1. Testing model loading...")
    try:
        model = CombinedHum2MelodyModel(
            pitch_ckpt_path='checkpoints_fixed_data/best_model_with_metadata.pth',
            onset_ckpt_path='checkpoints_onset/best_model.pth',
            device='cpu'
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test forward pass
    print("\n2. Testing forward pass...")
    dummy_cqt = torch.randn(2, 1, 500, 88)
    dummy_extras = torch.randn(2, 1, 500, 24)

    try:
        with torch.no_grad():
            frame, onset, offset, f0 = model(dummy_cqt, dummy_extras)

        print(f"  Input CQT shape: {dummy_cqt.shape}")
        print(f"  Input extras shape: {dummy_extras.shape}")
        print(f"  Output frame shape: {frame.shape}")
        print(f"  Output onset shape: {onset.shape}")
        print(f"  Output offset shape: {offset.shape}")
        print(f"  Output f0 shape: {f0.shape}")

        # Verify shapes
        expected_time = 125  # 500 / 4
        assert frame.shape == (2, expected_time, 88), f"Frame shape mismatch!"
        assert onset.shape == (2, expected_time, 1), f"Onset shape mismatch!"
        assert offset.shape == (2, expected_time, 1), f"Offset shape mismatch!"
        assert f0.shape == (2, expected_time, 2), f"F0 shape mismatch!"

        print("✅ Forward pass successful with correct shapes")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test with None extras
    print("\n3. Testing forward pass with None extras...")
    try:
        with torch.no_grad():
            frame, onset, offset, f0 = model(dummy_cqt, None)
        print("✅ Forward pass with None extras successful")
    except Exception as e:
        print(f"❌ Forward pass with None extras failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Test preprocessing info
    print("\n4. Testing metadata retrieval...")
    prep_info = model.get_preprocessing_info()
    arch_info = model.get_architecture_info()

    print(f"  Preprocessing params: {len(prep_info)} fields")
    print(f"    - Sample rate: {prep_info['sr']}")
    print(f"    - Hop length: {prep_info['hop_length']}")
    print(f"    - Output frames: {prep_info['expected_output_frames']}")

    print(f"  Architecture info:")
    print(f"    - Pitch: {arch_info['pitch']['model_class']}")
    print(f"    - Onset: {arch_info['onset']['model_class']}")

    print("✅ Metadata retrieval successful")

    # Test parameter count
    print("\n5. Testing parameter count...")
    total_params = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print("✅ Parameter count successful")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
