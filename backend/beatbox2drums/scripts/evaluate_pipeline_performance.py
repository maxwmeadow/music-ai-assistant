#!/usr/bin/env python3
"""
Large-Scale Pipeline Performance Evaluation

Evaluates the complete pipeline (onset detection + classification) on the full dataset.
Generates comprehensive statistics and visualizations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline


# Color mapping
DRUM_COLORS = {
    'kick': '#FF6B6B',
    'snare': '#4ECDC4',
    'hihat': '#FFD93D',
}


def load_ground_truth(label_path: str) -> Dict:
    """Load ground truth drum hits."""
    with open(label_path, 'r') as f:
        label = json.load(f)
    return label.get('drum_hits', {})


def match_predictions_to_gt(
    predictions: List,
    ground_truth: List,
    tolerance: float = 0.05
) -> Tuple[List, List, List]:
    """
    Match predictions to ground truth within tolerance.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    tp = []  # True positives: (pred_time, gt_time, confidence, distance)
    fp = []  # False positives: (pred_time, confidence)
    fn = []  # False negatives: (gt_time,)

    # Track which GTs have been matched
    matched_gt = set()

    # For each prediction, find closest GT within tolerance
    for pred in predictions:
        pred_time = pred.time
        confidence = pred.confidence

        best_match = None
        best_distance = tolerance

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            gt_time = gt['time']
            distance = abs(pred_time - gt_time)

            if distance < best_distance:
                best_distance = distance
                best_match = gt_idx

        if best_match is not None:
            matched_gt.add(best_match)
            gt_time = ground_truth[best_match]['time']
            tp.append((pred_time, gt_time, confidence, best_distance))
        else:
            fp.append((pred_time, confidence))

    # Find unmatched ground truths
    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in matched_gt:
            fn.append((gt['time'],))

    return tp, fp, fn


def evaluate_pipeline(
    pipeline: Beatbox2DrumsPipeline,
    manifest_path: str,
    split: str = 'val',
    tolerance: float = 0.05
) -> Dict:
    """
    Evaluate pipeline on entire dataset.

    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Get samples
    samples = [s for s in manifest['samples'] if s['split'] == split]

    print(f"\nEvaluating on {len(samples)} samples from {split} split...")
    print(f"Matching tolerance: {tolerance*1000:.0f}ms")
    print("="*70)

    # Initialize metrics
    metrics = {
        'per_type': {
            'kick': {'tp': [], 'fp': [], 'fn': []},
            'snare': {'tp': [], 'fp': [], 'fn': []},
            'hihat': {'tp': [], 'fp': [], 'fn': []},
        },
        'overall': {
            'total_samples': len(samples),
            'total_predictions': 0,
            'total_gt_drums': 0,
            'avg_confidence': [],
            'onset_detection_count': 0,
            'rejected_count': 0
        },
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'confidence_distribution': {'kick': [], 'snare': [], 'hihat': []},
        'timing_errors': {'kick': [], 'snare': [], 'hihat': []},
    }

    # Process each sample
    for sample in tqdm(samples, desc="Processing"):
        audio_path = sample['audio_path']
        label_path = sample['label_path']

        try:
            # Get predictions
            results = pipeline.predict(audio_path, return_details=True)
            drum_hits = results['drum_hits']

            # Load ground truth
            gt_drum_hits = load_ground_truth(label_path)

            # Update global counts
            metrics['overall']['total_predictions'] += len(drum_hits)
            metrics['overall']['onset_detection_count'] += results['total_onsets']
            metrics['overall']['rejected_count'] += results['rejected_count']

            # Evaluate per drum type
            for drum_type in ['kick', 'snare', 'hihat']:
                # Get predictions and ground truth for this type
                preds = [h for h in drum_hits if h.drum_type == drum_type]
                gts = gt_drum_hits.get(drum_type, [])

                metrics['overall']['total_gt_drums'] += len(gts)

                # Match predictions to ground truth
                tp, fp, fn = match_predictions_to_gt(preds, gts, tolerance)

                # Store results
                metrics['per_type'][drum_type]['tp'].extend(tp)
                metrics['per_type'][drum_type]['fp'].extend(fp)
                metrics['per_type'][drum_type]['fn'].extend(fn)

                # Collect confidence and timing errors
                for _, _, conf, timing_err in tp:
                    metrics['confidence_distribution'][drum_type].append(conf)
                    metrics['timing_errors'][drum_type].append(timing_err * 1000)  # Convert to ms

                # Build confusion matrix (from TPs)
                for pred_time, gt_time, _, _ in tp:
                    metrics['confusion_matrix'][drum_type][drum_type] += 1

                # FPs count as wrong predictions (confused as what?)
                # We don't know, so just count them separately
                for pred_time, _ in fp:
                    # Check if there's a GT of different type nearby
                    for other_type in ['kick', 'snare', 'hihat']:
                        if other_type == drum_type:
                            continue

                        other_gts = gt_drum_hits.get(other_type, [])
                        for gt in other_gts:
                            if abs(pred_time - gt['time']) < tolerance:
                                metrics['confusion_matrix'][drum_type][other_type] += 1
                                break

        except Exception as e:
            print(f"\nError processing {Path(audio_path).name}: {e}")
            continue

    # Calculate aggregate metrics
    for drum_type in ['kick', 'snare', 'hihat']:
        tp_count = len(metrics['per_type'][drum_type]['tp'])
        fp_count = len(metrics['per_type'][drum_type]['fp'])
        fn_count = len(metrics['per_type'][drum_type]['fn'])

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['per_type'][drum_type]['precision'] = precision
        metrics['per_type'][drum_type]['recall'] = recall
        metrics['per_type'][drum_type]['f1'] = f1

    # Overall metrics
    all_tp = sum(len(metrics['per_type'][t]['tp']) for t in ['kick', 'snare', 'hihat'])
    all_fp = sum(len(metrics['per_type'][t]['fp']) for t in ['kick', 'snare', 'hihat'])
    all_fn = sum(len(metrics['per_type'][t]['fn']) for t in ['kick', 'snare', 'hihat'])

    metrics['overall']['precision'] = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    metrics['overall']['recall'] = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    metrics['overall']['f1'] = (
        2 * metrics['overall']['precision'] * metrics['overall']['recall'] /
        (metrics['overall']['precision'] + metrics['overall']['recall'])
        if (metrics['overall']['precision'] + metrics['overall']['recall']) > 0 else 0
    )

    return metrics


def print_metrics(metrics: Dict):
    """Print evaluation metrics in a readable format."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print("\nOverall Performance:")
    print(f"  Precision: {metrics['overall']['precision']*100:.2f}%")
    print(f"  Recall: {metrics['overall']['recall']*100:.2f}%")
    print(f"  F1-Score: {metrics['overall']['f1']*100:.2f}%")

    print(f"\n  Total Samples: {metrics['overall']['total_samples']}")
    print(f"  Total GT Drums: {metrics['overall']['total_gt_drums']}")
    print(f"  Total Predictions: {metrics['overall']['total_predictions']}")
    print(f"  Onset Detections: {metrics['overall']['onset_detection_count']}")
    print(f"  Rejected (low conf): {metrics['overall']['rejected_count']}")

    print("\nPer-Class Performance:")
    print(f"{'Type':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 76)

    for drum_type in ['kick', 'snare', 'hihat']:
        m = metrics['per_type'][drum_type]
        tp = len(m['tp'])
        fp = len(m['fp'])
        fn = len(m['fn'])

        print(f"{drum_type:<10} "
              f"{m['precision']*100:>10.2f}%  "
              f"{m['recall']*100:>10.2f}%  "
              f"{m['f1']*100:>10.2f}%  "
              f"{tp:>6}  "
              f"{fp:>6}  "
              f"{fn:>6}")

    print("\nTiming Accuracy (for True Positives):")
    for drum_type in ['kick', 'snare', 'hihat']:
        timing_errors = metrics['timing_errors'][drum_type]
        if len(timing_errors) > 0:
            print(f"  {drum_type}: "
                  f"mean={np.mean(timing_errors):.1f}ms, "
                  f"median={np.median(timing_errors):.1f}ms, "
                  f"std={np.std(timing_errors):.1f}ms")

    print("\nConfidence Distribution (for True Positives):")
    for drum_type in ['kick', 'snare', 'hihat']:
        confidences = metrics['confidence_distribution'][drum_type]
        if len(confidences) > 0:
            print(f"  {drum_type}: "
                  f"mean={np.mean(confidences):.3f}, "
                  f"median={np.median(confidences):.3f}, "
                  f"min={np.min(confidences):.3f}")

    print("\n" + "="*70)


def create_visualizations(metrics: Dict, output_dir: Path):
    """Create comprehensive visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pipeline Performance Evaluation', fontsize=16, fontweight='bold')

    # Precision/Recall/F1 per class
    ax = axes[0, 0]
    drum_types = ['kick', 'snare', 'hihat']
    x = np.arange(len(drum_types))
    width = 0.25

    precision = [metrics['per_type'][t]['precision'] * 100 for t in drum_types]
    recall = [metrics['per_type'][t]['recall'] * 100 for t in drum_types]
    f1 = [metrics['per_type'][t]['f1'] * 100 for t in drum_types]

    ax.bar(x - width, precision, width, label='Precision', color='#FF6B6B', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', color='#4ECDC4', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', color='#FFD93D', alpha=0.8)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Precision, Recall, F1-Score by Drum Type')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in drum_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    # TP/FP/FN counts
    ax = axes[0, 1]
    tp_counts = [len(metrics['per_type'][t]['tp']) for t in drum_types]
    fp_counts = [len(metrics['per_type'][t]['fp']) for t in drum_types]
    fn_counts = [len(metrics['per_type'][t]['fn']) for t in drum_types]

    ax.bar(x - width, tp_counts, width, label='True Positive', color='green', alpha=0.7)
    ax.bar(x, fp_counts, width, label='False Positive', color='red', alpha=0.7)
    ax.bar(x + width, fn_counts, width, label='False Negative', color='orange', alpha=0.7)

    ax.set_ylabel('Count')
    ax.set_title('Detection Counts by Drum Type')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in drum_types])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Timing error distribution
    ax = axes[1, 0]
    timing_data = [metrics['timing_errors'][t] for t in drum_types]
    bp = ax.boxplot(timing_data, labels=[t.capitalize() for t in drum_types],
                    patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], [DRUM_COLORS[t] for t in drum_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Timing Error (ms)')
    ax.set_title('Timing Error Distribution (True Positives)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Confidence distribution
    ax = axes[1, 1]
    conf_data = [metrics['confidence_distribution'][t] for t in drum_types]
    bp = ax.boxplot(conf_data, labels=[t.capitalize() for t in drum_types],
                    patch_artist=True, showmeans=True)

    for patch, color in zip(bp['boxes'], [DRUM_COLORS[t] for t in drum_types]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Confidence')
    ax.set_title('Classification Confidence Distribution (True Positives)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'pipeline_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate complete pipeline performance')
    parser.add_argument('--onset-checkpoint', type=str,
                       default='beatbox2drums/cnn_onset_checkpoints/best_model.pth',
                       help='Path to onset detector checkpoint')
    parser.add_argument('--classifier-checkpoint', type=str,
                       default='beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth',
                       help='Path to classifier checkpoint')
    parser.add_argument('--manifest', type=str,
                       default='beatbox2drums/dataset/combined/manifest.json',
                       help='Path to dataset manifest')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Which split to evaluate')
    parser.add_argument('--tolerance', type=float, default=0.05,
                       help='Matching tolerance in seconds')
    parser.add_argument('--onset-threshold', type=float, default=0.5)
    parser.add_argument('--classifier-threshold', type=float, default=0.3)
    parser.add_argument('--output-dir', type=str,
                       default='beatbox2drums/evaluation_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create pipeline
    pipeline = Beatbox2DrumsPipeline(
        onset_checkpoint_path=args.onset_checkpoint,
        classifier_checkpoint_path=args.classifier_checkpoint,
        onset_threshold=args.onset_threshold,
        classifier_confidence_threshold=args.classifier_threshold
    )

    # Run evaluation
    metrics = evaluate_pipeline(
        pipeline=pipeline,
        manifest_path=args.manifest,
        split=args.split,
        tolerance=args.tolerance
    )

    # Print results
    print_metrics(metrics)

    # Create visualizations
    output_dir = Path(args.output_dir)
    create_visualizations(metrics, output_dir)

    # Save metrics to JSON
    metrics_file = output_dir / f'pipeline_metrics_{args.split}.json'

    # Convert metrics to JSON-serializable format
    json_metrics = {
        'overall': metrics['overall'],
        'per_type': {}
    }

    for drum_type in ['kick', 'snare', 'hihat']:
        json_metrics['per_type'][drum_type] = {
            'precision': float(metrics['per_type'][drum_type]['precision']),
            'recall': float(metrics['per_type'][drum_type]['recall']),
            'f1': float(metrics['per_type'][drum_type]['f1']),
            'tp_count': len(metrics['per_type'][drum_type]['tp']),
            'fp_count': len(metrics['per_type'][drum_type]['fp']),
            'fn_count': len(metrics['per_type'][drum_type]['fn']),
        }

    with open(metrics_file, 'w') as f:
        json.dump(json_metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_file}")
    print("="*70)


if __name__ == '__main__':
    main()
