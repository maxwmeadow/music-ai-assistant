"""
Create Combined Checkpoint

Merges pitch and onset/offset model checkpoints into a single deployable file.

The output file contains:
- Both models' state dicts (prefixed for clarity)
- All preprocessing metadata
- Architecture information
- Model type flags

Usage:
    python scripts/create_combined_checkpoint.py
"""

import torch
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_combined_checkpoint(
    pitch_ckpt_path: str,
    onset_ckpt_path: str,
    output_path: str
):
    """
    Merge two model checkpoints into a single file.

    Args:
        pitch_ckpt_path: Path to pitch model checkpoint
        onset_ckpt_path: Path to onset/offset model checkpoint
        output_path: Path to save combined checkpoint
    """
    print(f"\n{'='*70}")
    print(f"CREATING COMBINED CHECKPOINT")
    print(f"{'='*70}")
    print(f"Pitch model: {pitch_ckpt_path}")
    print(f"Onset model: {onset_ckpt_path}")
    print(f"Output: {output_path}")

    # Load both checkpoints
    print(f"\n1. Loading checkpoints...")
    pitch_ckpt = torch.load(pitch_ckpt_path, map_location='cpu', weights_only=False)
    onset_ckpt = torch.load(onset_ckpt_path, map_location='cpu', weights_only=False)
    print(f"   ✅ Both checkpoints loaded")

    # Verify preprocessing compatibility
    print(f"\n2. Verifying preprocessing compatibility...")
    pitch_prep = pitch_ckpt['preprocessing']
    onset_prep = onset_ckpt['preprocessing']

    critical_keys = ['sr', 'hop_length', 'n_bins', 'target_frames',
                     'expected_output_frames', 'cnn_downsample_factor']

    for key in critical_keys:
        pitch_val = pitch_prep.get(key)
        onset_val = onset_prep.get(key)
        if pitch_val != onset_val:
            raise ValueError(
                f"Preprocessing mismatch for '{key}'!\n"
                f"  Pitch: {pitch_val}\n"
                f"  Onset: {onset_val}"
            )
        print(f"   ✅ {key}: {pitch_val}")

    # Create combined state dict with prefixes
    print(f"\n3. Merging state dicts...")
    combined_state_dict = {}

    # Add pitch model weights with prefix
    for key, value in pitch_ckpt['model_state_dict'].items():
        combined_state_dict[f'pitch_model.{key}'] = value

    # Add onset model weights with prefix
    for key, value in onset_ckpt['model_state_dict'].items():
        combined_state_dict[f'onset_model.{key}'] = value

    pitch_params = len(pitch_ckpt['model_state_dict'])
    onset_params = len(onset_ckpt['model_state_dict'])
    total_params = len(combined_state_dict)

    print(f"   Pitch model keys: {pitch_params}")
    print(f"   Onset model keys: {onset_params}")
    print(f"   Combined keys: {total_params}")
    print(f"   ✅ State dicts merged")

    # Create combined checkpoint
    print(f"\n4. Creating combined checkpoint...")
    combined_ckpt = {
        'combined_model_version': '1.0',
        'model_state_dict': combined_state_dict,

        # Preprocessing (use pitch model's as reference)
        'preprocessing': pitch_prep,

        # Architecture info for both models
        'architecture': {
            'pitch_model': pitch_ckpt['architecture'],
            'onset_model': onset_ckpt['architecture'],
        },

        # Training info (for reference)
        'training_info': {
            'pitch_model': {
                'epoch': pitch_ckpt.get('epoch', 'unknown'),
                'best_f1': pitch_ckpt.get('best_f1', None),
                'train_metrics': pitch_ckpt.get('train_metrics', {}),
                'val_metrics': pitch_ckpt.get('val_metrics', {}),
            },
            'onset_model': {
                'epoch': onset_ckpt.get('epoch', 'unknown'),
                'best_f1': onset_ckpt.get('best_f1', None),
                'train_metrics': onset_ckpt.get('train_metrics', {}),
                'val_metrics': onset_ckpt.get('val_metrics', {}),
            }
        },

        # Source checkpoint paths
        'source_checkpoints': {
            'pitch': pitch_ckpt_path,
            'onset': onset_ckpt_path,
        }
    }

    # Save combined checkpoint
    print(f"\n5. Saving combined checkpoint...")
    torch.save(combined_ckpt, output_path)

    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✅ Saved to: {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")

    # Save metadata as JSON for easy reference
    metadata_path = Path(output_path).with_suffix('.json')
    print(f"\n6. Saving metadata...")
    metadata = {
        'version': '1.0',
        'preprocessing': pitch_prep,
        'architecture': {
            'pitch_model': {
                'class': pitch_ckpt['architecture']['model_class'],
                'n_bins': pitch_ckpt['architecture']['n_bins'],
                'hidden_size': pitch_ckpt['architecture']['hidden_size'],
            },
            'onset_model': {
                'class': onset_ckpt['architecture']['model_class'],
                'n_bins': onset_ckpt['architecture']['n_bins'],
                'hidden_size': onset_ckpt['architecture']['hidden_size'],
            }
        },
        'training_info': combined_ckpt['training_info'],
        'source_checkpoints': combined_ckpt['source_checkpoints'],
        'file_size_mb': file_size_mb
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Metadata saved to: {metadata_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"COMBINED CHECKPOINT CREATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nDeployment file: {output_path} ({file_size_mb:.1f} MB)")
    print(f"Metadata: {metadata_path}")
    print(f"\nTo load and use:")
    print(f"  from models.combined_model_loader import load_combined_model")
    print(f"  model = load_combined_model('{output_path}')")
    print(f"  frame, onset, offset, f0 = model(cqt, extras)")
    print(f"\nYour original checkpoints are safe:")
    print(f"  - {pitch_ckpt_path}")
    print(f"  - {onset_ckpt_path}")
    print(f"{'='*70}\n")

    return combined_ckpt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create combined checkpoint")
    parser.add_argument(
        '--pitch-ckpt',
        type=str,
        default='checkpoints_fixed_data/best_model_with_metadata.pth',
        help='Path to pitch model checkpoint'
    )
    parser.add_argument(
        '--onset-ckpt',
        type=str,
        default='checkpoints_enhanced_onset/best_model.pth',
        help='Path to onset model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_hum2melody_full.pth',
        help='Path to save combined checkpoint'
    )

    args = parser.parse_args()

    create_combined_checkpoint(
        pitch_ckpt_path=args.pitch_ckpt,
        onset_ckpt_path=args.onset_ckpt,
        output_path=args.output
    )
