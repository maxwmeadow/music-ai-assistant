#!/usr/bin/env python3
"""
Visualize CNN onset detector predictions on test examples.

Shows where the model predicts onsets vs ground truth, including
analysis of false positive timing relative to true onsets.
"""

import sys
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.cnn_onset_detector import CNNOnsetDetector


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


def analyze_false_positives(
    predicted_onsets: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: float = 0.030
) -> dict:
    """
    Analyze false positives: how far are they from nearest true onset?

    Returns:
        dict with FP analysis including distances to nearest GT onset
    """
    # Match predictions to ground truth
    tp_indices = []
    fp_indices = []

    for i, pred_time in enumerate(predicted_onsets):
        # Check if within tolerance of any GT onset
        distances = np.abs(ground_truth - pred_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf

        if min_distance <= tolerance:
            tp_indices.append(i)
        else:
            fp_indices.append(i)

    # For false positives, compute distance to nearest GT onset
    fp_distances = []
    for i in fp_indices:
        pred_time = predicted_onsets[i]
        if len(ground_truth) > 0:
            distances = np.abs(ground_truth - pred_time)
            min_distance = np.min(distances)
            fp_distances.append(min_distance * 1000)  # Convert to ms

    # Find false negatives
    fn_count = 0
    for gt_time in ground_truth:
        distances = np.abs(predicted_onsets - gt_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        if min_distance > tolerance:
            fn_count += 1

    return {
        'n_tp': len(tp_indices),
        'n_fp': len(fp_indices),
        'n_fn': fn_count,
        'fp_distances_ms': fp_distances,
        'fp_mean_distance': np.mean(fp_distances) if fp_distances else 0,
        'fp_median_distance': np.median(fp_distances) if fp_distances else 0,
        'fp_within_50ms': sum(1 for d in fp_distances if d <= 50),
        'fp_within_100ms': sum(1 for d in fp_distances if d <= 100),
    }


def visualize_predictions(
    audio_path: Path,
    label_path: Path,
    detector: CNNOnsetDetector,
    output_path: Path,
    tolerance: float = 0.030
):
    """
    Visualize CNN predictions vs ground truth for a single audio file.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr

    # Get ground truth
    ground_truth = load_ground_truth_onsets(label_path)

    # Get CNN predictions WITH PROBABILITIES
    # We need to extract probabilities manually
    mel_spec, times = detector.extract_mel_spectrogram(y)
    windows = detector.create_sliding_windows(mel_spec)
    onset_probs = detector.predict_onset_probabilities(windows)

    # Get final predictions after post-processing
    predicted_onsets = detector.post_process_onsets(onset_probs, times)

    # Match predictions to their probabilities
    # Predictions are at window CENTERS
    # Window i has center at times[i + window_frames//2]
    window_frames = detector.window_frames
    half_window = window_frames // 2
    window_center_times = times[half_window : half_window + len(onset_probs)]

    pred_probs = {}
    pred_indices = {}
    for pred_time in predicted_onsets:
        # Find the window whose center is closest to this prediction
        idx = np.argmin(np.abs(window_center_times - pred_time))
        pred_probs[pred_time] = onset_probs[idx]
        pred_indices[pred_time] = idx

    # Analyze false positives
    fp_analysis = analyze_false_positives(predicted_onsets, ground_truth, tolerance)

    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 14))

    # 1. Waveform with onset markers and probabilities
    ax = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6, color='gray')

    # Ground truth (green solid lines)
    ax.vlines(ground_truth, -1, 1, colors='lime', alpha=0.8, linewidth=2.5,
              label='Ground Truth Onsets', linestyle='-', zorder=3)

    # CNN predictions (red lines with probability labels)
    for pred_time in predicted_onsets:
        prob = pred_probs[pred_time]
        ax.axvline(pred_time, color='red', alpha=0.6, linewidth=1.5, linestyle='--', zorder=2)
        # Add probability label above the line
        ax.text(pred_time, 0.9, f'{prob:.2f}', rotation=90,
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=8, color='red', weight='bold')

    # Manual legend entry for predictions
    ax.plot([], [], 'r--', linewidth=1.5, label='CNN Predictions (with probability)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Waveform with Onset Predictions\n{audio_path.name}\nGREEN = Ground Truth | RED = CNN Predictions | Numbers = Model Confidence')
    ax.legend(loc='upper right')
    ax.set_xlim(0, duration)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    # 2. Onset Probability Curve (KEY PLOT FOR UNDERSTANDING MODEL CONFIDENCE)
    ax = axes[1]
    # Plot probability curve
    prob_times = times[:len(onset_probs)]
    ax.plot(prob_times, onset_probs, 'b-', linewidth=1, alpha=0.7, label='Onset Probability')
    ax.fill_between(prob_times, 0, onset_probs, alpha=0.2, color='blue')

    # Mark the threshold
    ax.axhline(detector.onset_threshold, color='purple', linestyle='--', linewidth=2,
              label=f'Threshold = {detector.onset_threshold}', zorder=3)

    # Mark ground truth onsets
    ax.vlines(ground_truth, 0, 1, colors='lime', alpha=0.5, linewidth=2,
             linestyle='-', label='Ground Truth', zorder=2)

    # Mark predictions with their probabilities
    for pred_time in predicted_onsets:
        prob = pred_probs[pred_time]
        # Check if TP or FP
        distances = np.abs(ground_truth - pred_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        is_tp = min_distance <= tolerance

        color = 'green' if is_tp else 'red'
        marker = 'o' if is_tp else 'x'
        ax.plot(pred_time, prob, marker, color=color, markersize=8,
               markeredgewidth=2, zorder=4)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Onset Probability')
    ax.set_title('CNN Onset Probability Over Time\nGREEN DOT = True Positive | RED X = False Positive | PURPLE LINE = Detection Threshold')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Mel spectrogram with onset markers
    ax = axes[2]
    hop_length = 441
    mel_spec_viz = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel_spec_viz, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis='time',
                                    y_axis='mel', ax=ax, cmap='magma')
    ax.vlines(ground_truth, 0, sr/2, colors='lime', alpha=0.8, linewidth=2, label='Ground Truth')
    ax.vlines(predicted_onsets, 0, sr/2, colors='red', alpha=0.6, linewidth=1.5, label='Predictions')
    ax.set_title('Mel Spectrogram with Onsets')
    ax.legend(loc='upper right')
    ax.set_xlim(0, duration)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # 4. Onset classification details
    ax = axes[3]
    # Show classification: TP (green), FP (red), FN (orange)
    for gt_time in ground_truth:
        # Check if this GT has a matching prediction
        distances = np.abs(predicted_onsets - gt_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        if min_distance <= tolerance:
            # True positive
            ax.axvline(gt_time, color='green', alpha=0.7, linewidth=2, label='TP' if gt_time == ground_truth[0] else '')
        else:
            # False negative
            ax.axvline(gt_time, color='orange', alpha=0.7, linewidth=2, linestyle='--', label='FN (missed)' if gt_time == ground_truth[0] else '')

    for pred_time in predicted_onsets:
        # Check if this prediction matches a GT
        distances = np.abs(ground_truth - pred_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        if min_distance > tolerance:
            # False positive
            ax.axvline(pred_time, color='red', alpha=0.5, linewidth=1.5, linestyle=':',
                      label='FP (false alarm)' if pred_time == predicted_onsets[0] else '')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Onset Type')
    ax.set_title('Onset Classification (TP=green, FP=red, FN=orange)')
    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 5. False positive distance distribution
    ax = axes[4]
    if fp_analysis['fp_distances_ms']:
        ax.hist(fp_analysis['fp_distances_ms'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(fp_analysis['fp_mean_distance'], color='blue', linestyle='--',
                  linewidth=2, label=f"Mean: {fp_analysis['fp_mean_distance']:.1f}ms")
        ax.axvline(fp_analysis['fp_median_distance'], color='green', linestyle='--',
                  linewidth=2, label=f"Median: {fp_analysis['fp_median_distance']:.1f}ms")
        ax.axvline(50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='50ms')
        ax.axvline(100, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label='100ms')
        ax.set_xlabel('Distance to Nearest GT Onset (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'False Positive Proximity Analysis ({len(fp_analysis["fp_distances_ms"])} FPs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No false positives!', ha='center', va='center', fontsize=16)
        ax.set_title('False Positive Analysis')

    # Add summary statistics text
    stats_text = f"""
    Performance Summary:
    - True Positives: {fp_analysis['n_tp']}
    - False Positives: {fp_analysis['n_fp']}
    - False Negatives: {fp_analysis['n_fn']}
    - Precision: {fp_analysis['n_tp']/(fp_analysis['n_tp']+fp_analysis['n_fp']):.1%}
    - Recall: {fp_analysis['n_tp']/(fp_analysis['n_tp']+fp_analysis['n_fn']):.1%}

    FP within 50ms of GT: {fp_analysis['fp_within_50ms']}/{fp_analysis['n_fp']} ({fp_analysis['fp_within_50ms']/max(fp_analysis['n_fp'],1)*100:.1f}%)
    FP within 100ms of GT: {fp_analysis['fp_within_100ms']}/{fp_analysis['n_fp']} ({fp_analysis['fp_within_100ms']/max(fp_analysis['n_fp'],1)*100:.1f}%)
    """

    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to {output_path}")

    return fp_analysis


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CNN onset predictions'
    )

    parser.add_argument('--model', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model/best_onset_model.h5',
                       help='Path to trained CNN model')
    parser.add_argument('--manifest', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json',
                       help='Path to dataset manifest')
    parser.add_argument('--output-dir', type=str,
                       default='/tmp/cnn_onset_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--n-examples', type=int, default=5,
                       help='Number of validation examples to visualize')
    parser.add_argument('--tolerance', type=float, default=0.030,
                       help='Tolerance for onset matching (seconds)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CNN Onset Detector Visualization")
    print(f"{'='*70}\n")

    # Load CNN detector
    print("Loading CNN model...")
    detector = CNNOnsetDetector(model_path=args.model)
    print(f"✓ Model loaded from {args.model}")
    print()

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    val_files = manifest.get('val', [])

    if len(val_files) == 0:
        print("⚠️  No validation files found in manifest!")
        return

    print(f"Found {len(val_files)} validation files")
    print(f"Visualizing {min(args.n_examples, len(val_files))} examples...")
    print()

    # Select random examples
    import random
    random.seed(42)
    selected_files = random.sample(val_files, min(args.n_examples, len(val_files)))

    # Collect overall FP statistics
    all_fp_distances = []
    all_stats = []

    # Visualize each example
    for i, file_info in enumerate(selected_files, 1):
        label_path = Path(file_info['label_path'])

        # Get audio path
        with open(label_path, 'r') as f:
            label = json.load(f)

        audio_path = Path(label['audio_path'])

        # Resolve relative paths
        if not audio_path.is_absolute():
            search_dir = label_path.parent
            while search_dir.name != 'beatbox2drums' and search_dir != search_dir.parent:
                search_dir = search_dir.parent
            if search_dir.name == 'beatbox2drums':
                audio_path = search_dir / audio_path

        if not audio_path.exists():
            print(f"⚠️  Audio file not found: {audio_path}")
            continue

        print(f"[{i}/{args.n_examples}] Processing {audio_path.name}...")

        output_path = output_dir / f"example_{i:02d}_{audio_path.stem}.png"

        fp_analysis = visualize_predictions(
            audio_path=audio_path,
            label_path=label_path,
            detector=detector,
            output_path=output_path,
            tolerance=args.tolerance
        )

        all_fp_distances.extend(fp_analysis['fp_distances_ms'])
        all_stats.append(fp_analysis)
        print()

    # Print overall summary
    print(f"\n{'='*70}")
    print("Overall False Positive Analysis")
    print(f"{'='*70}\n")

    total_tp = sum(s['n_tp'] for s in all_stats)
    total_fp = sum(s['n_fp'] for s in all_stats)
    total_fn = sum(s['n_fn'] for s in all_stats)

    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Overall Precision: {total_tp/(total_tp+total_fp):.1%}")
    print(f"Overall Recall: {total_tp/(total_tp+total_fn):.1%}")
    print()

    if all_fp_distances:
        print(f"False Positive Distance Statistics:")
        print(f"  Mean: {np.mean(all_fp_distances):.1f}ms")
        print(f"  Median: {np.median(all_fp_distances):.1f}ms")
        print(f"  Std Dev: {np.std(all_fp_distances):.1f}ms")
        print(f"  Min: {np.min(all_fp_distances):.1f}ms")
        print(f"  Max: {np.max(all_fp_distances):.1f}ms")
        print()
        print(f"FP within 50ms of true onset: {sum(1 for d in all_fp_distances if d <= 50)}/{total_fp} ({sum(1 for d in all_fp_distances if d <= 50)/max(total_fp,1)*100:.1f}%)")
        print(f"FP within 100ms of true onset: {sum(1 for d in all_fp_distances if d <= 100)}/{total_fp} ({sum(1 for d in all_fp_distances if d <= 100)/max(total_fp,1)*100:.1f}%)")
        print(f"FP within 150ms of true onset: {sum(1 for d in all_fp_distances if d <= 150)}/{total_fp} ({sum(1 for d in all_fp_distances if d <= 150)/max(total_fp,1)*100:.1f}%)")
        print()

    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
