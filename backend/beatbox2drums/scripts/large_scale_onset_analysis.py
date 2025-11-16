#!/usr/bin/env python3
"""
Large-scale CNN onset detector analysis.

Processes 1000-2000 validation examples to understand:
- True positive vs false positive probability distributions
- Distance patterns for false positives
- Overall performance characteristics
"""

import sys
import json
import numpy as np
from pathlib import Path
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    """Analyze a single example and collect all prediction data."""
    try:
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

        # Collect data for each prediction
        predictions = []
        for pred_time in predicted_onsets:
            idx = np.argmin(np.abs(window_center_times - pred_time))
            prob = onset_probs[idx]

            # Calculate distance to nearest ground truth
            distances = np.abs(ground_truth - pred_time)
            min_distance = np.min(distances) if len(distances) > 0 else np.inf

            is_tp = min_distance <= tolerance

            predictions.append({
                'time': pred_time,
                'probability': prob,
                'distance_ms': min_distance * 1000,
                'is_tp': is_tp
            })

        # Collect data for false negatives
        false_negatives = []
        for gt_time in ground_truth:
            distances = np.abs(predicted_onsets - gt_time)
            min_distance = np.min(distances) if len(distances) > 0 else np.inf

            if min_distance > tolerance:
                # Find what probability the model assigned at this location
                idx = np.argmin(np.abs(window_center_times - gt_time))
                prob = onset_probs[idx]
                false_negatives.append({
                    'time': gt_time,
                    'probability': prob
                })

        return {
            'predictions': predictions,
            'false_negatives': false_negatives,
            'n_gt': len(ground_truth),
            'success': True
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def plot_analysis(all_predictions, all_fn, output_path):
    """Create comprehensive visualization of all predictions."""

    # Separate TP and FP
    tp_data = [p for p in all_predictions if p['is_tp']]
    fp_data = [p for p in all_predictions if not p['is_tp']]

    tp_probs = [p['probability'] for p in tp_data]
    fp_probs = [p['probability'] for p in fp_data]
    fp_distances = [p['distance_ms'] for p in fp_data]
    fn_probs = [fn['probability'] for fn in all_fn]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main scatter plot: Distance vs Probability (all predictions)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    # Plot FP
    if fp_data:
        fp_x = [p['probability'] for p in fp_data]
        fp_y = [p['distance_ms'] for p in fp_data]
        ax1.scatter(fp_x, fp_y, c='red', alpha=0.3, s=20, label=f'False Positives (n={len(fp_data)})')

    # Plot TP
    if tp_data:
        tp_x = [p['probability'] for p in tp_data]
        tp_y = [p['distance_ms'] for p in tp_data]
        ax1.scatter(tp_x, tp_y, c='green', alpha=0.3, s=20, label=f'True Positives (n={len(tp_data)})')

    ax1.axhline(30, color='orange', linestyle='--', alpha=0.5, label='30ms tolerance')
    ax1.axvline(0.5, color='purple', linestyle='--', alpha=0.5, label='Threshold=0.5')
    ax1.set_xlabel('Model Probability', fontsize=12)
    ax1.set_ylabel('Distance to Nearest Ground Truth (ms)', fontsize=12)
    ax1.set_title('All Predictions: Distance vs Probability', fontsize=14, weight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.4, 1.0)

    # 2. TP Probability Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    if tp_probs:
        ax2.hist(tp_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(tp_probs), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(tp_probs):.3f}')
        ax2.axvline(np.median(tp_probs), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(tp_probs):.3f}')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Count')
        ax2.set_title(f'TP Probability Distribution\n(n={len(tp_probs)})')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    # 3. FP Probability Distribution
    ax3 = fig.add_subplot(gs[1, 2])
    if fp_probs:
        ax3.hist(fp_probs, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(np.mean(fp_probs), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(fp_probs):.3f}')
        ax3.axvline(np.median(fp_probs), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(fp_probs):.3f}')
        ax3.set_xlabel('Probability')
        ax3.set_ylabel('Count')
        ax3.set_title(f'FP Probability Distribution\n(n={len(fp_probs)})')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # 4. FP Distance Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    if fp_distances:
        ax4.hist(fp_distances, bins=100, alpha=0.7, color='red', edgecolor='black')
        ax4.axvline(np.mean(fp_distances), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(fp_distances):.1f}ms')
        ax4.axvline(np.median(fp_distances), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(fp_distances):.1f}ms')
        ax4.axvline(50, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='50ms')
        ax4.axvline(100, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label='100ms')
        ax4.set_xlabel('Distance to Nearest GT (ms)')
        ax4.set_ylabel('Count')
        ax4.set_title(f'FP Distance Distribution\n(n={len(fp_distances)})')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    # 5. FN Probability Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    if fn_probs:
        ax5.hist(fn_probs, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(np.mean(fn_probs), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(fn_probs):.3f}')
        ax5.axvline(0.5, color='purple', linestyle='--', linewidth=2, label='Threshold')
        below_thresh = sum(1 for p in fn_probs if p < 0.5)
        ax5.set_xlabel('Model Probability at GT Location')
        ax5.set_ylabel('Count')
        ax5.set_title(f'FN Probability Distribution\n(n={len(fn_probs)}, {below_thresh} below threshold)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # 6. Distance ranges breakdown
    ax6 = fig.add_subplot(gs[2, 2])
    if fp_distances:
        ranges = {
            '0-30ms': sum(1 for d in fp_distances if d <= 30),
            '30-50ms': sum(1 for d in fp_distances if 30 < d <= 50),
            '50-100ms': sum(1 for d in fp_distances if 50 < d <= 100),
            '100-500ms': sum(1 for d in fp_distances if 100 < d <= 500),
            '500ms+': sum(1 for d in fp_distances if d > 500)
        }

        labels = list(ranges.keys())
        values = list(ranges.values())
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']

        bars = ax6.bar(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha='right')
        ax6.set_ylabel('Count')
        ax6.set_title('FP Distance Ranges')
        ax6.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}\n({val/len(fp_distances)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/cnn_onset_model/best_onset_model.h5')
    parser.add_argument('--manifest', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json')
    parser.add_argument('--n-examples', type=int, default=1000,
                       help='Number of examples to analyze (default: 1000)')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/large_scale_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("LARGE-SCALE CNN ONSET DETECTOR ANALYSIS")
    print("="*70 + "\n")

    # Load detector
    print("Loading CNN model...")
    detector = CNNOnsetDetector(model_path=args.model)
    print(f"✓ Loaded\n")

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    val_files = manifest.get('val', [])
    n_to_analyze = min(args.n_examples, len(val_files))

    random.seed(42)
    selected_files = random.sample(val_files, n_to_analyze)

    print(f"Analyzing {n_to_analyze} validation examples...")
    print(f"(out of {len(val_files)} total validation files)\n")

    # Collect all data
    all_predictions = []
    all_false_negatives = []
    total_gt = 0
    successful = 0
    failed = 0

    for file_info in tqdm(selected_files, desc="Processing examples"):
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
            failed += 1
            continue

        result = analyze_example(audio_path, label_path, detector)

        if result['success']:
            # Convert numpy types to Python native types for JSON serialization
            for pred in result['predictions']:
                pred['time'] = float(pred['time'])
                pred['probability'] = float(pred['probability'])
                pred['distance_ms'] = float(pred['distance_ms'])
                pred['is_tp'] = bool(pred['is_tp'])
            for fn in result['false_negatives']:
                fn['time'] = float(fn['time'])
                fn['probability'] = float(fn['probability'])

            all_predictions.extend(result['predictions'])
            all_false_negatives.extend(result['false_negatives'])
            total_gt += result['n_gt']
            successful += 1
        else:
            failed += 1

    print(f"\n✓ Analysis complete!")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total ground truth onsets: {total_gt}")
    print(f"  Total predictions: {len(all_predictions)}\n")

    # Compute statistics
    tp_data = [p for p in all_predictions if p['is_tp']]
    fp_data = [p for p in all_predictions if not p['is_tp']]

    n_tp = len(tp_data)
    n_fp = len(fp_data)
    n_fn = len(all_false_negatives)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Generate detailed report
    report = []
    report.append("="*70)
    report.append("LARGE-SCALE CNN ONSET DETECTOR ANALYSIS REPORT")
    report.append("="*70)
    report.append(f"Analyzed: {successful} examples")
    report.append(f"Total ground truth onsets: {total_gt}")
    report.append("")

    report.append("OVERALL PERFORMANCE")
    report.append("-" * 70)
    report.append(f"True Positives: {n_tp}")
    report.append(f"False Positives: {n_fp}")
    report.append(f"False Negatives: {n_fn}")
    report.append(f"Total Predictions: {len(all_predictions)}")
    report.append(f"")
    report.append(f"Precision: {precision:.1%}")
    report.append(f"Recall: {recall:.1%}")
    report.append(f"F1 Score: {f1:.1%}")
    report.append("")

    # TP Statistics
    if tp_data:
        tp_probs = [p['probability'] for p in tp_data]
        tp_distances = [p['distance_ms'] for p in tp_data]

        report.append("TRUE POSITIVE STATISTICS")
        report.append("-" * 70)
        report.append(f"Count: {n_tp}")
        report.append(f"Probability Statistics:")
        report.append(f"  Mean: {np.mean(tp_probs):.3f}")
        report.append(f"  Median: {np.median(tp_probs):.3f}")
        report.append(f"  Std Dev: {np.std(tp_probs):.3f}")
        report.append(f"  Min: {np.min(tp_probs):.3f}")
        report.append(f"  25th percentile: {np.percentile(tp_probs, 25):.3f}")
        report.append(f"  75th percentile: {np.percentile(tp_probs, 75):.3f}")
        report.append(f"  Max: {np.max(tp_probs):.3f}")
        report.append(f"Distance from Ground Truth:")
        report.append(f"  Mean: {np.mean(tp_distances):.2f}ms")
        report.append(f"  Median: {np.median(tp_distances):.2f}ms")
        report.append(f"  Max: {np.max(tp_distances):.2f}ms")
        report.append("")

    # FP Statistics
    if fp_data:
        fp_probs = [p['probability'] for p in fp_data]
        fp_distances = [p['distance_ms'] for p in fp_data]

        report.append("FALSE POSITIVE STATISTICS")
        report.append("-" * 70)
        report.append(f"Count: {n_fp}")
        report.append(f"Probability Statistics:")
        report.append(f"  Mean: {np.mean(fp_probs):.3f}")
        report.append(f"  Median: {np.median(fp_probs):.3f}")
        report.append(f"  Std Dev: {np.std(fp_probs):.3f}")
        report.append(f"  Min: {np.min(fp_probs):.3f}")
        report.append(f"  Max: {np.max(fp_probs):.3f}")
        report.append(f"")
        report.append(f"Distance to Nearest Ground Truth:")
        report.append(f"  Mean: {np.mean(fp_distances):.1f}ms")
        report.append(f"  Median: {np.median(fp_distances):.1f}ms")
        report.append(f"  Std Dev: {np.std(fp_distances):.1f}ms")
        report.append(f"  Min: {np.min(fp_distances):.1f}ms")
        report.append(f"  Max: {np.max(fp_distances):.1f}ms")
        report.append(f"")
        report.append(f"Distance Distribution:")
        report.append(f"  Within 30ms: {sum(1 for d in fp_distances if d <= 30)} ({sum(1 for d in fp_distances if d <= 30)/n_fp*100:.1f}%)")
        report.append(f"  Within 50ms: {sum(1 for d in fp_distances if d <= 50)} ({sum(1 for d in fp_distances if d <= 50)/n_fp*100:.1f}%)")
        report.append(f"  Within 100ms: {sum(1 for d in fp_distances if d <= 100)} ({sum(1 for d in fp_distances if d <= 100)/n_fp*100:.1f}%)")
        report.append(f"  Within 500ms: {sum(1 for d in fp_distances if d <= 500)} ({sum(1 for d in fp_distances if d <= 500)/n_fp*100:.1f}%)")
        report.append(f"  Beyond 500ms: {sum(1 for d in fp_distances if d > 500)} ({sum(1 for d in fp_distances if d > 500)/n_fp*100:.1f}%)")
        report.append("")

    # FN Statistics
    if all_false_negatives:
        fn_probs = [fn['probability'] for fn in all_false_negatives]

        report.append("FALSE NEGATIVE STATISTICS")
        report.append("-" * 70)
        report.append(f"Count: {n_fn}")
        report.append(f"Model Probability at GT Location:")
        report.append(f"  Mean: {np.mean(fn_probs):.3f}")
        report.append(f"  Median: {np.median(fn_probs):.3f}")
        report.append(f"  Min: {np.min(fn_probs):.3f}")
        report.append(f"  Max: {np.max(fn_probs):.3f}")
        report.append(f"  Below threshold (0.5): {sum(1 for p in fn_probs if p < 0.5)}/{n_fn} ({sum(1 for p in fn_probs if p < 0.5)/n_fn*100:.1f}%)")
        report.append(f"  Above threshold (0.5): {sum(1 for p in fn_probs if p >= 0.5)}/{n_fn} ({sum(1 for p in fn_probs if p >= 0.5)/n_fn*100:.1f}%)")
        report.append("")

    # Probability overlap analysis
    if tp_data and fp_data:
        tp_probs = [p['probability'] for p in tp_data]
        fp_probs = [p['probability'] for p in fp_data]

        report.append("PROBABILITY OVERLAP ANALYSIS")
        report.append("-" * 70)
        report.append(f"TP mean: {np.mean(tp_probs):.3f} ± {np.std(tp_probs):.3f}")
        report.append(f"FP mean: {np.mean(fp_probs):.3f} ± {np.std(fp_probs):.3f}")
        report.append(f"Difference: {np.mean(tp_probs) - np.mean(fp_probs):.3f}")

        if np.mean(tp_probs) - np.mean(fp_probs) > 0.05:
            suggested_threshold = (np.mean(tp_probs) + np.mean(fp_probs)) / 2
            report.append(f"")
            report.append(f"SUGGESTION: Raise threshold to {suggested_threshold:.2f}")
            report.append(f"  This could filter some FPs while preserving most TPs")
        else:
            report.append(f"")
            report.append(f"NOTE: Significant probability overlap - threshold adjustment unlikely to help")
        report.append("")

    report.append("="*70)

    # Print and save report
    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {report_path}")

    # Save raw data
    data_path = output_dir / "analysis_data.json"
    with open(data_path, 'w') as f:
        json.dump({
            'metadata': {
                'n_examples': successful,
                'total_gt': total_gt,
                'n_tp': n_tp,
                'n_fp': n_fp,
                'n_fn': n_fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'predictions': all_predictions,
            'false_negatives': all_false_negatives
        }, f, indent=2)
    print(f"✓ Raw data saved to {data_path}")

    # Generate visualization
    print("\nGenerating comprehensive visualization...")
    plot_path = output_dir / "large_scale_analysis.png"
    plot_analysis(all_predictions, all_false_negatives, plot_path)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print("")


if __name__ == '__main__':
    main()
