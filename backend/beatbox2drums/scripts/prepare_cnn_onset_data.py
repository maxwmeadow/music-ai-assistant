#!/usr/bin/env python3
"""
Prepare training data for CNN-based onset detection.

Creates mel spectrogram windows labeled as onset/no-onset based on ground truth.
"""

import sys
import json
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ground_truth_onsets(label_path: Path) -> np.ndarray:
    """Load ground truth onset times from label file."""
    with open(label_path, 'r') as f:
        label = json.load(f)

    onset_times = []
    for drum_type in ['kick', 'snare', 'hihat']:
        hits = label.get('drum_hits', {}).get(drum_type, [])
        for hit in hits:
            onset_times.append(hit['time'])

    return np.array(sorted(onset_times))


def extract_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 441,
    fmax: float = 8000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mel spectrogram from audio file.

    Returns:
        mel_db: Mel spectrogram in dB (shape: [n_mels, n_frames])
        times: Time stamps for each frame (shape: [n_frames])
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Harmonic-percussive separation (isolate percussion)
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))

    # Compute mel spectrogram on percussive component
    mel_spec = librosa.feature.melspectrogram(
        y=y_percussive,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax
    )

    # Convert to dB
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Get time stamps
    times = librosa.frames_to_time(
        np.arange(mel_db.shape[1]),
        sr=sr,
        hop_length=hop_length
    )

    return mel_db, times


def create_spectrogram_windows(
    mel_spec: np.ndarray,
    times: np.ndarray,
    window_frames: int = 12
) -> List[Tuple[np.ndarray, float]]:
    """
    Create sliding windows over mel spectrogram.

    Args:
        mel_spec: Mel spectrogram [n_mels, n_frames]
        times: Time stamps [n_frames]
        window_frames: Number of frames per window

    Returns:
        List of (window, center_time) tuples
    """
    n_mels, n_frames = mel_spec.shape
    windows = []

    for i in range(n_frames - window_frames + 1):
        window = mel_spec[:, i:i + window_frames]
        center_time = times[i + window_frames // 2]
        windows.append((window, center_time))

    return windows


def label_windows(
    windows: List[Tuple[np.ndarray, float]],
    ground_truth_onsets: np.ndarray,
    tolerance: float = 0.030
) -> np.ndarray:
    """
    Label windows as onset (1) or no-onset (0).

    A window is labeled as onset if its center time is within
    tolerance of any ground truth onset.

    Args:
        windows: List of (window, center_time)
        ground_truth_onsets: Array of ground truth onset times
        tolerance: Time tolerance in seconds (default: 30ms)

    Returns:
        Array of labels [0 or 1] for each window
    """
    labels = []

    for window, center_time in windows:
        # Check if center time is within tolerance of any onset
        is_onset = False

        for onset_time in ground_truth_onsets:
            if abs(center_time - onset_time) <= tolerance:
                is_onset = True
                break

        labels.append(1 if is_onset else 0)

    return np.array(labels)


def process_single_file(
    label_path: Path,
    sample_rate: int = 16000,
    n_mels: int = 80,
    window_frames: int = 12,
    tolerance: float = 0.030,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process single audio file into windows and labels.

    Returns:
        windows: Array of shape [n_windows, n_mels, window_frames]
        labels: Array of shape [n_windows] with 0/1 labels
    """
    # Load label
    with open(label_path, 'r') as f:
        label = json.load(f)

    # Get audio path
    audio_path = Path(label['audio_path'])

    # Resolve relative paths
    if not audio_path.is_absolute():
        search_dir = label_path.parent
        while search_dir.name != 'beatbox2drums' and search_dir != search_dir.parent:
            search_dir = search_dir.parent
        if search_dir.name == 'beatbox2drums':
            audio_path = search_dir / audio_path

    if not audio_path.exists():
        if verbose:
            print(f"⚠️  Audio file not found: {audio_path}")
        return None, None

    # Load ground truth onsets
    ground_truth = load_ground_truth_onsets(label_path)

    if len(ground_truth) == 0:
        if verbose:
            print(f"⚠️  No ground truth onsets in {label_path}")
        return None, None

    try:
        # Extract mel spectrogram
        mel_spec, times = extract_mel_spectrogram(
            str(audio_path),
            sample_rate=sample_rate,
            n_mels=n_mels
        )

        # Create windows
        windows_with_times = create_spectrogram_windows(
            mel_spec,
            times,
            window_frames=window_frames
        )

        # Label windows
        labels = label_windows(
            windows_with_times,
            ground_truth,
            tolerance=tolerance
        )

        # Extract just the windows
        windows = np.array([w for w, t in windows_with_times])

        # Expand dimensions for CNN input [n_windows, n_mels, window_frames, 1]
        windows = np.expand_dims(windows, axis=-1)

        if verbose:
            onset_count = np.sum(labels)
            print(f"✓ {audio_path.name}: {len(windows)} windows, {onset_count} onset, {len(windows)-onset_count} no-onset")

        return windows, labels

    except Exception as e:
        if verbose:
            print(f"⚠️  Error processing {audio_path}: {e}")
        return None, None


def prepare_dataset(
    manifest_path: Path,
    output_dir: Path,
    split: str = 'train',
    sample_rate: int = 16000,
    n_mels: int = 80,
    window_frames: int = 12,
    tolerance: float = 0.030,
    max_files: int = None,
    verbose: bool = False
):
    """
    Prepare CNN onset detection dataset from manifest.

    Args:
        manifest_path: Path to combined manifest
        output_dir: Directory to save prepared data
        split: Dataset split ('train', 'val', 'test')
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        window_frames: Number of frames per window
        tolerance: Onset labeling tolerance (seconds)
        max_files: Maximum files to process (None = all)
        verbose: Print progress
    """
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    label_files = manifest.get(split, [])

    if max_files is not None:
        label_files = label_files[:max_files]

    print(f"\n{'='*70}")
    print(f"Preparing CNN Onset Detection Data: {split} split")
    print(f"{'='*70}")
    print(f"Files: {len(label_files)}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Mel bands: {n_mels}")
    print(f"Window frames: {window_frames}")
    print(f"Tolerance: {tolerance*1000:.0f}ms")
    print(f"{'='*70}\n")

    # Process all files
    all_windows = []
    all_labels = []

    for label_info in tqdm(label_files, desc=f"Processing {split} files"):
        label_path = Path(label_info['label_path'])

        windows, labels = process_single_file(
            label_path,
            sample_rate=sample_rate,
            n_mels=n_mels,
            window_frames=window_frames,
            tolerance=tolerance,
            verbose=False
        )

        if windows is not None and labels is not None:
            all_windows.append(windows)
            all_labels.append(labels)

    # Concatenate all data
    if len(all_windows) == 0:
        print("⚠️  No files successfully processed!")
        return

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Print statistics
    n_onset = np.sum(all_labels == 1)
    n_no_onset = np.sum(all_labels == 0)

    print(f"\n{'='*70}")
    print(f"Dataset Statistics: {split}")
    print(f"{'='*70}")
    print(f"Total windows: {len(all_labels):,}")
    print(f"Onset windows: {n_onset:,} ({n_onset/len(all_labels)*100:.1f}%)")
    print(f"No-onset windows: {n_no_onset:,} ({n_no_onset/len(all_labels)*100:.1f}%)")
    print(f"Class ratio: 1:{n_no_onset/n_onset:.1f}")
    print(f"Window shape: {all_windows.shape}")
    print(f"{'='*70}\n")

    # Save data
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_file = output_dir / f'{split}_windows.npy'
    labels_file = output_dir / f'{split}_labels.npy'

    np.save(windows_file, all_windows)
    np.save(labels_file, all_labels)

    print(f"✓ Saved to:")
    print(f"  {windows_file}")
    print(f"  {labels_file}")
    print()

    # Save metadata
    metadata = {
        'split': split,
        'sample_rate': sample_rate,
        'n_mels': n_mels,
        'window_frames': window_frames,
        'tolerance': tolerance,
        'n_files': len(label_files),
        'n_windows': len(all_labels),
        'n_onset': int(n_onset),
        'n_no_onset': int(n_no_onset),
        'window_shape': list(all_windows.shape),
    }

    metadata_file = output_dir / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CNN onset detection training data'
    )

    parser.add_argument('--manifest', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json',
                       help='Path to combined manifest')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_data',
                       help='Output directory for prepared data')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to prepare')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--n-mels', type=int, default=80,
                       help='Number of mel bands (default: 80)')
    parser.add_argument('--window-frames', type=int, default=12,
                       help='Number of frames per window (default: 12)')
    parser.add_argument('--tolerance', type=float, default=0.030,
                       help='Onset labeling tolerance in seconds (default: 0.030)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files to process (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')

    args = parser.parse_args()

    prepare_dataset(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        split=args.split,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        window_frames=args.window_frames,
        tolerance=args.tolerance,
        max_files=args.max_files,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
