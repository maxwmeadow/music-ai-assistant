#!/usr/bin/env python3
"""
Analyze CNN onset detector outputs to recommend post-processing strategies.

Examines probability distributions, timing errors, and false positive patterns
to suggest optimal post-processing parameters.
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.cnn_onset_detector import CNNOnsetDetector
import librosa


def load_ground_truth_onsets(label_path: Path) -> np.ndarray:
    """Load ground truth onset times."""
    with open(label_path, 'r') as f:
        label = json.load(f)

    onset_times = []
    for drum_type in ['kick', 'snare', 'hihat']:
        hits = label.get('drum_hits', {}).get(drum_type, [])
        for hit in hits:
            onset_times.append(hit['time'])

    return np.array(sorted(onset_times))


def analyze_example(audio_path, label_path, detector, tolerance=0.030):
    """Analyze a single example in detail."""
    # Load audio
    y, _ = librosa.load(audio_path, sr=16000, mono=True)

    # Get ground truth
    ground_truth = load_ground_truth_onsets(label_path)

    # Get predictions WITH raw probabilities
    mel_spec, times = detector.extract_mel_spectrogram(y)
    windows = detector.create_sliding_windows(mel_spec)
    onset_probs = detector.predict_onset_probabilities(windows)
    predicted_onsets = detector.post_process_onsets(onset_probs, times)

    # Compute window center times for matching
    window_frames = detector.window_frames
    half_window = window_frames // 2
    window_center_times = times[half_window : half_window + len(onset_probs)]

    # Match predictions to probabilities
    pred_probs = {}
    for pred_time in predicted_onsets:
        idx = np.argmin(np.abs(window_center_times - pred_time))
        pred_probs[pred_time] = onset_probs[idx]

    # Classify each prediction
    tp_probs = []
    fp_probs = []
    fp_distances = []

    for pred_time in predicted_onsets:
        distances = np.abs(ground_truth - pred_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf

        if min_distance <= tolerance:
            tp_probs.append(pred_probs[pred_time])
        else:
            fp_probs.append(pred_probs[pred_time])
            fp_distances.append(min_distance * 1000)  # ms

    # Find false negatives (missed onsets)
    fn_gt_probs = []
    for gt_time in ground_truth:
        distances = np.abs(predicted_onsets - gt_time)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf

        if min_distance > tolerance:
            # Find what probability the model assigned at this location
            idx = np.argmin(np.abs(window_center_times - gt_time))
            fn_gt_probs.append(onset_probs[idx])

    return {
        'n_gt': len(ground_truth),
        'n_pred': len(predicted_onsets),
        'n_tp': len(tp_probs),
        'n_fp': len(fp_probs),
        'n_fn': len(fn_gt_probs),
        'tp_probs': tp_probs,
        'fp_probs': fp_probs,
        'fn_probs': fn_gt_probs,
        'fp_distances': fp_distances,
        'all_probs': onset_probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model/best_onset_model.h5')
    parser.add_argument('--manifest', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json')
    parser.add_argument('--n-examples', type=int, default=20)
    parser.add_argument('--output', type=str,
                       default='/tmp/postprocessing_analysis.txt')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CNN Onset Detector Post-Processing Analysis")
    print("="*70 + "\n")

    # Load detector
    print("Loading CNN model...")
    detector = CNNOnsetDetector(model_path=args.model)
    print(f"✓ Loaded\n")

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    val_files = manifest.get('val', [])
    random.seed(42)
    selected_files = random.sample(val_files, min(args.n_examples, len(val_files)))

    # Collect statistics
    all_tp_probs = []
    all_fp_probs = []
    all_fn_probs = []
    all_fp_distances = []
    total_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0}

    print(f"Analyzing {len(selected_files)} examples...\n")

    for file_info in selected_files:
        label_path = Path(file_info['label_path'])
        with open(label_path, 'r') as f:
            label = json.load(f)
        audio_path = Path(label['audio_path'])

        if not audio_path.is_absolute():
            search_dir = label_path.parent
            while search_dir.name != 'beatbox2drums' and search_dir != search_dir.parent:
                search_dir = search_dir.parent
            if search_dir.name == 'beatbox2drums':
                audio_path = search_dir / audio_path

        if not audio_path.exists():
            continue

        stats = analyze_example(audio_path, label_path, detector)

        all_tp_probs.extend(stats['tp_probs'])
        all_fp_probs.extend(stats['fp_probs'])
        all_fn_probs.extend(stats['fn_probs'])
        all_fp_distances.extend(stats['fp_distances'])

        total_stats['tp'] += stats['n_tp']
        total_stats['fp'] += stats['n_fp']
        total_stats['fn'] += stats['n_fn']
        total_stats['gt'] += stats['n_gt']

    # Compute metrics
    precision = total_stats['tp'] / (total_stats['tp'] + total_stats['fp']) if total_stats['tp'] + total_stats['fp'] > 0 else 0
    recall = total_stats['tp'] / (total_stats['tp'] + total_stats['fn']) if total_stats['tp'] + total_stats['fn'] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Generate report
    report = []
    report.append("="*70)
    report.append("POST-PROCESSING ANALYSIS REPORT")
    report.append("="*70)
    report.append("")

    # Overall Performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 70)
    report.append(f"Total Ground Truth Onsets: {total_stats['gt']}")
    report.append(f"Total Predictions: {total_stats['tp'] + total_stats['fp']}")
    report.append(f"True Positives: {total_stats['tp']}")
    report.append(f"False Positives: {total_stats['fp']}")
    report.append(f"False Negatives: {total_stats['fn']}")
    report.append(f"Precision: {precision:.1%}")
    report.append(f"Recall: {recall:.1%}")
    report.append(f"F1 Score: {f1:.1%}")
    report.append("")

    # Probability Distribution Analysis
    report.append("PROBABILITY DISTRIBUTION ANALYSIS")
    report.append("-" * 70)

    if all_tp_probs:
        report.append(f"True Positives (n={len(all_tp_probs)}):")
        report.append(f"  Mean probability: {np.mean(all_tp_probs):.3f}")
        report.append(f"  Median probability: {np.median(all_tp_probs):.3f}")
        report.append(f"  Std deviation: {np.std(all_tp_probs):.3f}")
        report.append(f"  Min: {np.min(all_tp_probs):.3f}")
        report.append(f"  25th percentile: {np.percentile(all_tp_probs, 25):.3f}")
        report.append(f"  75th percentile: {np.percentile(all_tp_probs, 75):.3f}")
        report.append(f"  Max: {np.max(all_tp_probs):.3f}")

    report.append("")

    if all_fp_probs:
        report.append(f"False Positives (n={len(all_fp_probs)}):")
        report.append(f"  Mean probability: {np.mean(all_fp_probs):.3f}")
        report.append(f"  Median probability: {np.median(all_fp_probs):.3f}")
        report.append(f"  Std deviation: {np.std(all_fp_probs):.3f}")
        report.append(f"  Min: {np.min(all_fp_probs):.3f}")
        report.append(f"  Max: {np.max(all_fp_probs):.3f}")
    else:
        report.append("False Positives: NONE!")

    report.append("")

    if all_fn_probs:
        report.append(f"False Negatives (Missed Onsets) (n={len(all_fn_probs)}):")
        report.append(f"  Mean probability at GT location: {np.mean(all_fn_probs):.3f}")
        report.append(f"  Median probability: {np.median(all_fn_probs):.3f}")
        report.append(f"  Max: {np.max(all_fn_probs):.3f}")
        report.append(f"  Below threshold (0.5): {sum(1 for p in all_fn_probs if p < 0.5)}/{len(all_fn_probs)}")

    report.append("")

    # False Positive Timing Analysis
    if all_fp_distances:
        report.append("FALSE POSITIVE TIMING ANALYSIS")
        report.append("-" * 70)
        report.append(f"Distance to nearest true onset:")
        report.append(f"  Mean: {np.mean(all_fp_distances):.1f}ms")
        report.append(f"  Median: {np.median(all_fp_distances):.1f}ms")
        report.append(f"  Min: {np.min(all_fp_distances):.1f}ms")
        report.append(f"  Max: {np.max(all_fp_distances):.1f}ms")
        report.append(f"  Within 50ms: {sum(1 for d in all_fp_distances if d <= 50)}/{len(all_fp_distances)} ({sum(1 for d in all_fp_distances if d <= 50)/len(all_fp_distances)*100:.1f}%)")
        report.append(f"  Within 100ms: {sum(1 for d in all_fp_distances if d <= 100)}/{len(all_fp_distances)} ({sum(1 for d in all_fp_distances if d <= 100)/len(all_fp_distances)*100:.1f}%)")
        report.append("")

    # Recommendations
    report.append("="*70)
    report.append("POST-PROCESSING RECOMMENDATIONS")
    report.append("="*70)
    report.append("")

    # Threshold analysis
    if all_tp_probs and all_fp_probs:
        tp_mean = np.mean(all_tp_probs)
        fp_mean = np.mean(all_fp_probs)

        if fp_mean < tp_mean - 0.1:
            new_threshold = (tp_mean + fp_mean) / 2
            report.append(f"1. THRESHOLD ADJUSTMENT")
            report.append(f"   Current: 0.5")
            report.append(f"   Recommended: {new_threshold:.2f}")
            report.append(f"   Reason: FP probabilities ({fp_mean:.3f}) are significantly lower than TP ({tp_mean:.3f})")
            report.append("")
        else:
            report.append(f"1. THRESHOLD: Keep at 0.5")
            report.append(f"   Reason: FP and TP probability distributions overlap significantly")
            report.append("")

    # Peak delta analysis
    if all_fp_distances:
        median_fp_dist = np.median(all_fp_distances)
        recommended_delta = max(40, int(median_fp_dist * 1.2))

        report.append(f"2. PEAK_DELTA (Minimum Inter-Onset Interval)")
        report.append(f"   Current: 20ms")
        report.append(f"   Recommended: {recommended_delta}ms")
        report.append(f"   Reason: FPs are clustered {median_fp_dist:.1f}ms from true onsets (timing jitter)")
        report.append(f"   This will merge duplicate detections of the same onset")
        report.append("")

    # Overall recommendation
    report.append(f"3. OVERALL STRATEGY")
    if len(all_fp_distances) > 0 and np.median(all_fp_distances) < 50:
        report.append(f"   ✓ Model performance is EXCELLENT")
        report.append(f"   ✓ False positives are timing jitter, not wrong detections")
        report.append(f"   ✓ Simple fix: Increase peak_delta to {recommended_delta}ms")
        report.append(f"   ✓ For classifier training: FPs are close enough to not hurt performance")
        report.append(f"   ✓ Consider using AS-IS for classifier training, filtering is optional")
    else:
        report.append(f"   - Review individual examples to understand FP patterns")
        report.append(f"   - May need more sophisticated post-processing")

    report.append("")
    report.append("="*70)

    # Print and save
    report_text = "\n".join(report)
    print(report_text)

    with open(args.output, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Report saved to {args.output}\n")


if __name__ == '__main__':
    main()
