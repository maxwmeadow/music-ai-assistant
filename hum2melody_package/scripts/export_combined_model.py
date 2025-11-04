"""
Export Combined Model to TorchScript

Exports the combined hum2melody model (pitch + onset) to a single
TorchScript artifact for deployment.

The exported model:
- Loads from a single .pt file
- Runs on CPU or GPU
- Has fixed preprocessing requirements
- Returns tuple of (frame, onset, offset, f0) outputs

Usage:
    python scripts/export_combined_model.py \\
        --output combined_hum2melody.pt \\
        --device cuda
"""

import torch
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.combined_model import CombinedHum2MelodyModel


def export_to_torchscript(
    pitch_ckpt_path: str,
    onset_ckpt_path: str,
    output_path: str,
    device: str = 'cpu',
    verify: bool = True
):
    """
    Export combined model to TorchScript.

    Args:
        pitch_ckpt_path: Path to pitch model checkpoint (with metadata)
        onset_ckpt_path: Path to onset model checkpoint
        output_path: Path to save exported TorchScript model
        device: Device to export on ('cpu' or 'cuda')
        verify: Whether to verify traced model matches original
    """
    print(f"\n{'='*70}")
    print(f"EXPORTING COMBINED MODEL TO TORCHSCRIPT")
    print(f"{'='*70}")
    print(f"Pitch checkpoint: {pitch_ckpt_path}")
    print(f"Onset checkpoint: {onset_ckpt_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")

    device = torch.device(device)

    # Load wrapper
    print("\n1. Loading wrapper model...")
    wrapper = CombinedHum2MelodyModel(
        pitch_ckpt_path=pitch_ckpt_path,
        onset_ckpt_path=onset_ckpt_path,
        device=str(device)
    )
    wrapper.eval()
    print("✅ Wrapper loaded")

    # Get preprocessing info for documentation
    prep_info = wrapper.get_preprocessing_info()
    arch_info = wrapper.get_architecture_info()

    print(f"\nModel Info:")
    print(f"  Sample rate: {prep_info['sr']} Hz")
    print(f"  Hop length: {prep_info['hop_length']}")
    print(f"  CQT bins: {prep_info['n_bins']}")
    print(f"  Input frames: {prep_info['target_frames']}")
    print(f"  Output frames: {prep_info['expected_output_frames']}")
    print(f"  Total parameters: {wrapper.count_parameters():,}")

    # Create example inputs for tracing
    print("\n2. Creating example inputs...")
    example_cqt = torch.randn(1, 1, 500, 88).to(device)
    example_extras = torch.randn(1, 1, 500, 24).to(device)
    print(f"  CQT shape: {example_cqt.shape}")
    print(f"  Extras shape: {example_extras.shape}")

    # Test wrapper before tracing
    print("\n3. Testing wrapper before tracing...")
    with torch.no_grad():
        orig_out = wrapper(example_cqt, example_extras)
        frame_orig, onset_orig, offset_orig, f0_orig = orig_out

    print(f"  Output shapes:")
    print(f"    Frame: {frame_orig.shape}")
    print(f"    Onset: {onset_orig.shape}")
    print(f"    Offset: {offset_orig.shape}")
    print(f"    F0: {f0_orig.shape}")
    print("✅ Wrapper forward pass successful")

    # Trace the model
    print("\n4. Tracing model with TorchScript...")
    try:
        traced = torch.jit.trace(
            wrapper,
            (example_cqt, example_extras),
            strict=False,  # Allow some flexibility
            check_trace=True  # Verify trace is correct
        )
        print("✅ Model traced successfully")
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        raise

    # Verify traced model produces identical outputs
    if verify:
        print("\n5. Verifying traced model...")
        with torch.no_grad():
            traced_out = traced(example_cqt, example_extras)
            frame_traced, onset_traced, offset_traced, f0_traced = traced_out

            # Check shapes
            assert frame_traced.shape == frame_orig.shape, "Frame shape mismatch!"
            assert onset_traced.shape == onset_orig.shape, "Onset shape mismatch!"
            assert offset_traced.shape == offset_orig.shape, "Offset shape mismatch!"
            assert f0_traced.shape == f0_orig.shape, "F0 shape mismatch!"

            # Check values
            max_diff_frame = (frame_orig - frame_traced).abs().max().item()
            max_diff_onset = (onset_orig - onset_traced).abs().max().item()
            max_diff_offset = (offset_orig - offset_traced).abs().max().item()
            max_diff_f0 = (f0_orig - f0_traced).abs().max().item()

            print(f"  Max absolute differences:")
            print(f"    Frame:  {max_diff_frame:.2e}")
            print(f"    Onset:  {max_diff_onset:.2e}")
            print(f"    Offset: {max_diff_offset:.2e}")
            print(f"    F0:     {max_diff_f0:.2e}")

            # Verify within tolerance
            tolerance = 1e-5
            if max(max_diff_frame, max_diff_onset, max_diff_offset, max_diff_f0) > tolerance:
                print(f"❌ Traced model outputs differ by more than {tolerance}")
                print(f"   This may indicate a tracing issue")
            else:
                print(f"✅ Traced model produces identical outputs (within {tolerance})")
    else:
        print("\n5. Skipping verification (--no-verify)")

    # Save traced model
    print(f"\n6. Saving traced model to {output_path}...")
    traced.save(output_path)

    # Check file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Model saved")
    print(f"   File size: {file_size_mb:.1f} MB")

    # Save metadata alongside model
    metadata_path = Path(output_path).with_suffix('.json')
    print(f"\n7. Saving metadata to {metadata_path}...")

    import json
    metadata = {
        'preprocessing': prep_info,
        'architecture': {
            'pitch_model': arch_info['pitch']['model_class'],
            'onset_model': arch_info['onset']['model_class'],
        },
        'total_parameters': wrapper.count_parameters(),
        'export_info': {
            'torch_version': torch.__version__,
            'device': str(device),
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved")

    # Print deployment instructions
    print(f"\n{'='*70}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDeployment files:")
    print(f"  Model: {output_path} ({file_size_mb:.1f} MB)")
    print(f"  Metadata: {metadata_path}")
    print(f"\nTo load and use:")
    print(f"  model = torch.jit.load('{output_path}')")
    print(f"  model.eval()")
    print(f"  frame, onset, offset, f0 = model(cqt_input, extras_input)")
    print(f"\nPreprocessing requirements:")
    print(f"  - Sample rate: {prep_info['sr']} Hz")
    print(f"  - Hop length: {prep_info['hop_length']}")
    print(f"  - CQT bins: {prep_info['n_bins']} (fmin={prep_info['fmin']} Hz)")
    print(f"  - Input shape: (batch, 1, {prep_info['target_frames']}, {prep_info['n_bins']})")
    print(f"  - Output shape: (batch, {prep_info['expected_output_frames']}, *)")
    print(f"{'='*70}\n")

    return traced


def main():
    parser = argparse.ArgumentParser(description="Export combined model to TorchScript")
    parser.add_argument(
        '--pitch-ckpt',
        type=str,
        default='checkpoints_fixed_data/best_model_with_metadata.pth',
        help='Path to pitch model checkpoint (with metadata)'
    )
    parser.add_argument(
        '--onset-ckpt',
        type=str,
        default='checkpoints_onset/best_model.pth',
        help='Path to onset model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_hum2melody.pt',
        help='Path to save exported TorchScript model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to export on'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification step'
    )

    args = parser.parse_args()

    export_to_torchscript(
        pitch_ckpt_path=args.pitch_ckpt,
        onset_ckpt_path=args.onset_ckpt,
        output_path=args.output,
        device=args.device,
        verify=not args.no_verify
    )


if __name__ == '__main__':
    main()
