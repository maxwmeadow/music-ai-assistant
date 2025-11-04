#!/usr/bin/env python3
"""
Simple Inference Example for Hum2Melody Model

This script demonstrates how to load and use the combined hum2melody model
for pitch and onset/offset detection.

Usage:
    python simple_inference.py --audio path/to/audio.wav --checkpoint ../checkpoints/combined_hum2melody_full.pth
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path so we can import from models/data
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import librosa
from models.combined_model_loader import load_combined_model


def preprocess_audio(audio_path: str, sr: int = 16000, target_frames: int = 500):
    """
    Preprocess audio file to model input format.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate (must be 16000)
        target_frames: Number of frames to process (500 = ~16 seconds)

    Returns:
        cqt_tensor: (1, 1, 500, 88) - CQT input
        extras_tensor: (1, 1, 500, 24) - Onset features (zeros for simple inference)
    """
    print(f"\n{'='*70}")
    print(f"PREPROCESSING AUDIO")
    print(f"{'='*70}")
    print(f"Audio file: {audio_path}")

    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    print(f"Loaded audio: {len(audio)} samples ({len(audio)/sr:.2f} seconds)")

    # Extract CQT
    cqt = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=512,
        n_bins=88,
        bins_per_octave=12,
        fmin=27.5  # A0 (MIDI 21)
    )
    print(f"CQT shape: {cqt.shape}")

    # Convert to dB and normalize
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_normalized = (cqt_db + 80) / 80  # Normalize to [0, 1]
    cqt_normalized = np.clip(cqt_normalized, 0, 1)

    # Pad or truncate to target_frames
    if cqt_normalized.shape[1] < target_frames:
        pad_width = target_frames - cqt_normalized.shape[1]
        cqt_normalized = np.pad(
            cqt_normalized,
            ((0, 0), (0, pad_width)),
            mode='constant'
        )
        print(f"Padded CQT to {target_frames} frames")
    elif cqt_normalized.shape[1] > target_frames:
        cqt_normalized = cqt_normalized[:, :target_frames]
        print(f"Truncated CQT to {target_frames} frames")

    # Transpose and add batch/channel dims: (88, 500) → (1, 1, 500, 88)
    cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)

    # Create zero-filled extras tensor
    # In production, you may want to extract actual onset features
    extras_tensor = torch.zeros(1, 1, target_frames, 24)

    print(f"Final CQT tensor: {cqt_tensor.shape}")
    print(f"Final extras tensor: {extras_tensor.shape}")
    print(f"{'='*70}\n")

    return cqt_tensor, extras_tensor


def run_inference(model, cqt, extras, device='cpu'):
    """
    Run inference on preprocessed audio.

    Args:
        model: Loaded hum2melody model
        cqt: CQT tensor (1, 1, 500, 88)
        extras: Extras tensor (1, 1, 500, 24)
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        dict: Model outputs with probabilities
    """
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE")
    print(f"{'='*70}")
    print(f"Device: {device}")

    # Move to device
    cqt = cqt.to(device)
    extras = extras.to(device)

    # Run inference
    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt, extras)

    print(f"Raw outputs:")
    print(f"  Frame: {frame.shape} (pitch classification logits)")
    print(f"  Onset: {onset.shape} (onset detection logits)")
    print(f"  Offset: {offset.shape} (offset detection logits)")
    print(f"  F0: {f0.shape} (continuous F0 [value, voicing])")

    # Convert logits to probabilities
    frame_probs = torch.sigmoid(frame)  # (1, 125, 88)
    onset_probs = torch.sigmoid(onset)  # (1, 125, 1)
    offset_probs = torch.sigmoid(offset)  # (1, 125, 1)
    f0_value = f0[:, :, 0]  # (1, 125) - normalized F0
    voicing = torch.sigmoid(f0[:, :, 1])  # (1, 125) - voicing probability

    print(f"\nConverted to probabilities:")
    print(f"  Frame probs: min={frame_probs.min():.3f}, max={frame_probs.max():.3f}")
    print(f"  Onset probs: min={onset_probs.min():.3f}, max={onset_probs.max():.3f}")
    print(f"  Offset probs: min={offset_probs.min():.3f}, max={offset_probs.max():.3f}")
    print(f"  Voicing: min={voicing.min():.3f}, max={voicing.max():.3f}")
    print(f"{'='*70}\n")

    return {
        'frame_probs': frame_probs.cpu(),
        'onset_probs': onset_probs.cpu(),
        'offset_probs': offset_probs.cpu(),
        'f0_value': f0_value.cpu(),
        'voicing': voicing.cpu(),
    }


def extract_notes(outputs, onset_threshold=0.15, voicing_threshold=0.5, min_midi=21):
    """
    Extract note sequence from model outputs.

    Args:
        outputs: Dict of model outputs (from run_inference)
        onset_threshold: Threshold for onset detection (0.05-0.3)
        voicing_threshold: Threshold for voicing detection
        min_midi: Minimum MIDI note (21 = A0)

    Returns:
        list: List of (start_time, duration, midi_note, confidence) tuples
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING NOTES")
    print(f"{'='*70}")
    print(f"Onset threshold: {onset_threshold}")
    print(f"Voicing threshold: {voicing_threshold}")

    frame_probs = outputs['frame_probs'][0]  # (125, 88)
    onset_probs = outputs['onset_probs'][0]  # (125, 1)
    voicing = outputs['voicing'][0]  # (125,)

    frame_rate = 31.25  # Hz (16000 / 512 / 4)  # CNN downsamples by 4

    # Detect onsets
    onset_mask = (onset_probs.squeeze() > onset_threshold).numpy()
    voiced_mask = (voicing > voicing_threshold).numpy()

    # Combine onset and voicing
    valid_onsets = onset_mask & voiced_mask
    onset_frames = np.where(valid_onsets)[0]

    print(f"Found {len(onset_frames)} onsets")

    # Extract notes
    notes = []
    for i, onset_frame in enumerate(onset_frames):
        # Get pitch (highest probability note)
        pitch_idx = frame_probs[onset_frame].argmax().item()
        midi_note = pitch_idx + min_midi
        confidence = frame_probs[onset_frame, pitch_idx].item()

        # Find offset (next onset or end of sequence)
        if i + 1 < len(onset_frames):
            offset_frame = onset_frames[i + 1]
        else:
            offset_frame = len(onset_mask) - 1

        # Convert frames to time
        start_time = onset_frame / frame_rate
        end_time = offset_frame / frame_rate
        duration = end_time - start_time

        notes.append((start_time, duration, midi_note, confidence))

    print(f"Extracted {len(notes)} notes")
    print(f"{'='*70}\n")

    return notes


def print_notes(notes):
    """Print note sequence in a readable format."""
    print(f"\n{'='*70}")
    print(f"NOTE SEQUENCE")
    print(f"{'='*70}")

    if not notes:
        print("No notes detected!")
        print(f"{'='*70}\n")
        return

    print(f"{'Time (s)':<10} {'Duration (s)':<12} {'MIDI':<6} {'Note':<8} {'Confidence':<10}")
    print(f"{'-'*70}")

    for start, duration, midi, confidence in notes:
        note_name = librosa.midi_to_note(midi)
        print(f"{start:<10.2f} {duration:<12.2f} {midi:<6} {note_name:<8} {confidence:<10.3f}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simple inference example for Hum2Melody model"
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file (.wav, .mp3, etc.)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='../checkpoints/combined_hum2melody_full.pth',
        help='Path to combined checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cpu or cuda)'
    )
    parser.add_argument(
        '--onset-threshold',
        type=float,
        default=0.15,
        help='Onset detection threshold (0.05-0.3)'
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_combined_model(args.checkpoint, device=args.device)
    model.eval()

    # Preprocess audio
    cqt, extras = preprocess_audio(args.audio)

    # Run inference
    outputs = run_inference(model, cqt, extras, device=args.device)

    # Extract notes
    notes = extract_notes(outputs, onset_threshold=args.onset_threshold)

    # Print results
    print_notes(notes)

    print("\nDone! You can now use these notes in your application.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
