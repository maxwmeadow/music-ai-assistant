#!/usr/bin/env python3
"""
Generate classifier training data using CNN onset detector.

This ensures train/test consistency: both use CNN-detected onsets rather
than ground truth, eliminating distribution mismatch.
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
import librosa
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.cnn_onset_detector import CNNOnsetDetector


def load_ground_truth(label_path: Path) -> dict:
    """Load ground truth labels."""
    with open(label_path, 'r') as f:
        return json.load(f)


def match_onset_to_drum_type(onset_time: float, label: dict, tolerance: float = 0.05) -> str:
    """
    Match a CNN-detected onset to the closest ground truth drum type.

    Args:
        onset_time: CNN-detected onset time
        label: Ground truth label dict
        tolerance: Tolerance window for matching (50ms default)

    Returns:
        Drum type ('kick', 'snare', 'hihat') or None if no match
    """
    drum_hits = label.get('drum_hits', {})

    best_match = None
    best_distance = np.inf

    for drum_type in ['kick', 'snare', 'hihat']:
        hits = drum_hits.get(drum_type, [])
        for hit in hits:
            gt_time = hit['time']
            distance = abs(onset_time - gt_time)

            if distance < best_distance and distance <= tolerance:
                best_distance = distance
                best_match = drum_type

    return best_match


def extract_window_at_onset(
    mel_spec: np.ndarray,
    times: np.ndarray,
    onset_time: float,
    window_frames: int = 12
) -> np.ndarray:
    """
    Extract mel spectrogram window centered at onset time.

    Args:
        mel_spec: Mel spectrogram [n_mels, n_frames]
        times: Time array [n_frames]
        onset_time: Center time for extraction
        window_frames: Window size in frames

    Returns:
        Window [n_mels, window_frames] or None if out of bounds
    """
    # Find frame closest to onset time
    center_frame = np.argmin(np.abs(times - onset_time))

    # Calculate window bounds
    half_window = window_frames // 2
    start_frame = center_frame - half_window
    end_frame = start_frame + window_frames

    # Check bounds
    if start_frame < 0 or end_frame > mel_spec.shape[1]:
        return None

    return mel_spec[:, start_frame:end_frame]


def process_audio_file(
    audio_path: Path,
    label_path: Path,
    detector: CNNOnsetDetector,
    n_mels: int = 128
) -> dict:
    """
    Process a single audio file: detect onsets with CNN, extract windows, label them.

    Returns:
        Dictionary with windows and labels
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Load ground truth
        label = load_ground_truth(label_path)

        # Detect onsets with CNN
        cnn_onsets = detector.detect_from_audio(y)

        if len(cnn_onsets) == 0:
            return {
                'success': True,
                'windows': [],
                'labels': [],
                'onset_times': [],
                'n_cnn_onsets': 0,
                'n_matched': 0,
                'n_unmatched': 0
            }

        # Extract mel spectrogram for windowing (higher resolution for classifier)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=441,
            fmax=8000.0
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Get time array
        times = librosa.frames_to_time(
            np.arange(mel_db.shape[1]),
            sr=sr,
            hop_length=441
        )

        # Process each CNN-detected onset
        windows = []
        labels = []
        onset_times_out = []
        matched_count = 0

        for onset_time in cnn_onsets:
            # Match to ground truth drum type
            drum_type = match_onset_to_drum_type(onset_time, label, tolerance=0.05)

            if drum_type is None:
                # CNN detected something that's not in ground truth
                # Could be a false positive, but we still extract it
                # The classifier will learn to reject these
                continue  # Skip unmatched for now

            # Extract window
            window = extract_window_at_onset(mel_db, times, onset_time, window_frames=12)

            if window is not None:
                windows.append(window)
                labels.append(drum_type)
                onset_times_out.append(onset_time)
                matched_count += 1

        return {
            'success': True,
            'windows': windows,
            'labels': labels,
            'onset_times': onset_times_out,
            'n_cnn_onsets': len(cnn_onsets),
            'n_matched': matched_count,
            'n_unmatched': len(cnn_onsets) - matched_count
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Generate classifier training data using CNN onset detector'
    )
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to dataset manifest.json')
    parser.add_argument('--cnn-model', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model/best_onset_model.h5',
                       help='Path to trained CNN onset model')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for classifier training data')
    parser.add_argument('--n-mels', type=int, default=128,
                       help='Number of mel bands for classifier')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'],
                       help='Which split to process')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"CLASSIFIER DATA GENERATION (CNN-BASED) - {args.split.upper()} SPLIT")
    print("="*70 + "\n")

    # Load CNN onset detector
    print("Loading CNN onset detector...")
    detector = CNNOnsetDetector(
        model_path=args.cnn_model,
        peak_delta=0.05,  # 50ms NMS window
        verbose=False
    )
    print("✓ CNN detector loaded\n")

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    files = manifest.get(args.split, [])
    print(f"Found {len(files)} files in {args.split} split\n")

    # Process all files
    all_windows = []
    all_labels = []

    stats = {
        'processed': 0,
        'failed': 0,
        'total_cnn_onsets': 0,
        'total_matched': 0,
        'total_unmatched': 0,
        'drum_counts': {'kick': 0, 'snare': 0, 'hihat': 0}
    }

    print(f"Processing {len(files)} audio files...")

    for file_info in tqdm(files, desc=f"Processing {args.split}"):
        label_path = Path(file_info['label_path'])

        # Get audio path
        with open(label_path, 'r') as f:
            label = json.load(f)
        audio_path = Path(label['audio_path'])

        # Make path absolute if needed
        if not audio_path.is_absolute():
            search_dir = label_path.parent
            while search_dir.name != 'beatbox2drums' and search_dir != search_dir.parent:
                search_dir = search_dir.parent
            if search_dir.name == 'beatbox2drums':
                audio_path = search_dir / audio_path

        if not audio_path.exists():
            stats['failed'] += 1
            continue

        # Process file
        result = process_audio_file(audio_path, label_path, detector, n_mels=args.n_mels)

        if not result['success']:
            stats['failed'] += 1
            continue

        # Collect windows and labels
        all_windows.extend(result['windows'])
        all_labels.extend(result['labels'])

        # Update stats
        stats['processed'] += 1
        stats['total_cnn_onsets'] += result['n_cnn_onsets']
        stats['total_matched'] += result['n_matched']
        stats['total_unmatched'] += result['n_unmatched']

        for label in result['labels']:
            stats['drum_counts'][label] += 1

    print(f"\n✓ Processing complete!\n")

    # Convert to arrays
    X = np.array(all_windows)  # [n_samples, n_mels, window_frames]
    y = np.array(all_labels)  # [n_samples]

    # Save data
    output_file = output_dir / f'{args.split}_data_cnn.npz'
    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Files processed: {stats['processed']}")
    print(f"Files failed: {stats['failed']}")
    print(f"Total CNN-detected onsets: {stats['total_cnn_onsets']}")
    print(f"Matched to ground truth: {stats['total_matched']} ({stats['total_matched']/stats['total_cnn_onsets']*100:.1f}%)")
    print(f"Unmatched (potential FPs): {stats['total_unmatched']} ({stats['total_unmatched']/stats['total_cnn_onsets']*100:.1f}%)")
    print(f"")
    print(f"Total training windows: {len(X)}")
    print(f"  Kick: {stats['drum_counts']['kick']} ({stats['drum_counts']['kick']/len(X)*100:.1f}%)")
    print(f"  Snare: {stats['drum_counts']['snare']} ({stats['drum_counts']['snare']/len(X)*100:.1f}%)")
    print(f"  Hihat: {stats['drum_counts']['hihat']} ({stats['drum_counts']['hihat']/len(X)*100:.1f}%)")
    print(f"")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"")
    print(f"✓ Saved to {output_file}")
    print("="*70 + "\n")

    # Save metadata
    metadata = {
        'split': args.split,
        'n_samples': len(X),
        'n_mels': args.n_mels,
        'window_frames': 12,
        'cnn_model': args.cnn_model,
        'stats': stats
    }

    metadata_file = output_dir / f'{args.split}_metadata_cnn.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to {metadata_file}\n")


if __name__ == '__main__':
    main()
