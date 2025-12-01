#!/usr/bin/env python3
"""
Visualize Pipeline Predictions (Individual Samples)

Shows:
1. Waveform
2. CNN onset detection probabilities
3. Detected onsets
4. Classified drum hits with types and confidence
5. Ground truth comparison
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import argparse
from typing import Dict, List

from inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline


# Color mapping for drum types
DRUM_COLORS = {
    'kick': '#FF6B6B',    # Red
    'snare': '#4ECDC4',   # Cyan
    'hihat': '#FFD93D',   # Yellow
}


def load_ground_truth(label_path: str) -> Dict:
    """Load ground truth drum hits from label file."""
    with open(label_path, 'r') as f:
        label = json.load(f)
    return label.get('drum_hits', {})


def visualize_predictions(
    audio_path: str,
    label_path: str,
    pipeline: Beatbox2DrumsPipeline,
    output_path: str = None,
    show: bool = True
):
    """
    Create a comprehensive visualization of pipeline predictions.

    Args:
        audio_path: Path to audio file
        label_path: Path to ground truth label file
        pipeline: Beatbox2Drums pipeline instance
        output_path: Where to save the plot (optional)
        show: Whether to display the plot
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=pipeline.sample_rate, mono=True)
    duration = len(y) / sr

    # Get predictions with details
    results = pipeline.predict_from_audio(y, return_details=True)
    drum_hits = results['drum_hits']
    onset_times = results['onset_times']

    # Load ground truth
    gt_drum_hits = load_ground_truth(label_path)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'Pipeline Predictions: {Path(audio_path).name}', fontsize=14, fontweight='bold')

    # Plot 1: Waveform with onset markers
    ax = axes[0]
    time_axis = np.linspace(0, duration, len(y))
    ax.plot(time_axis, y, color='gray', alpha=0.5, linewidth=0.5)

    # Mark onset detections
    for onset_time in onset_times:
        ax.axvline(onset_time, color='orange', alpha=0.3, linewidth=1, linestyle='--')

    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform with CNN Onset Detections')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mel Spectrogram
    ax = axes[1]
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=441, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=441, x_axis='time', y_axis='mel',
        ax=ax, cmap='viridis', fmax=8000
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Mark drum hits with colored lines
    for hit in drum_hits:
        ax.axvline(hit.time, color=DRUM_COLORS[hit.drum_type], alpha=0.6, linewidth=2)

    ax.set_title('Mel Spectrogram with Classified Drum Hits')

    # Plot 3: Predicted Drum Hits Timeline
    ax = axes[2]
    ax.set_ylim(0, 4)
    ax.set_xlim(0, duration)

    # Plot predicted hits
    for drum_idx, drum_type in enumerate(['kick', 'snare', 'hihat']):
        y_pos = drum_idx + 1
        hits = [h for h in drum_hits if h.drum_type == drum_type]

        for hit in hits:
            # Circle size based on confidence
            size = 100 + 200 * hit.confidence
            ax.scatter(
                hit.time, y_pos,
                s=size,
                color=DRUM_COLORS[drum_type],
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5,
                label=f'{drum_type} (pred)' if hit == hits[0] else ''
            )
            # Add confidence text
            ax.text(
                hit.time, y_pos + 0.15,
                f'{hit.confidence:.2f}',
                ha='center', va='bottom', fontsize=7, alpha=0.8
            )

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Kick', 'Snare', 'Hihat'])
    ax.set_xlabel('Time (s)')
    ax.set_title('Predicted Drum Hits (size = confidence)')
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: Ground Truth Comparison
    ax = axes[3]
    ax.set_ylim(0, 4)
    ax.set_xlim(0, duration)

    # Plot ground truth
    for drum_idx, drum_type in enumerate(['kick', 'snare', 'hihat']):
        y_pos = drum_idx + 1
        gt_hits = gt_drum_hits.get(drum_type, [])

        for hit in gt_hits:
            ax.scatter(
                hit['time'], y_pos,
                s=150,
                color=DRUM_COLORS[drum_type],
                alpha=0.5,
                marker='s',  # Square for ground truth
                edgecolors='black',
                linewidths=1.5
            )

    # Overlay predicted hits (smaller)
    for drum_idx, drum_type in enumerate(['kick', 'snare', 'hihat']):
        y_pos = drum_idx + 1
        pred_hits = [h for h in drum_hits if h.drum_type == drum_type]

        for hit in pred_hits:
            ax.scatter(
                hit.time, y_pos,
                s=80,
                color=DRUM_COLORS[drum_type],
                alpha=0.9,
                marker='o',  # Circle for prediction
                edgecolors='white',
                linewidths=1.5
            )

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Kick', 'Snare', 'Hihat'])
    ax.set_xlabel('Time (s)')
    ax.set_title('Ground Truth (squares) vs Predictions (circles)')
    ax.grid(True, alpha=0.3, axis='x')

    # Create custom legend
    gt_patch = mpatches.Patch(color='gray', label='Ground Truth (square)', alpha=0.5)
    pred_patch = mpatches.Patch(color='gray', label='Prediction (circle)', alpha=0.9)
    ax.legend(handles=[gt_patch, pred_patch], loc='upper right')

    # Add statistics text
    stats_text = f"""
Statistics:
  Total onsets detected: {results['total_onsets']}
  Drum hits after filtering: {len(drum_hits)}
  Rejected (low conf): {results['rejected_count']}

  Predictions by type:
    Kick: {sum(1 for h in drum_hits if h.drum_type == 'kick')}
    Snare: {sum(1 for h in drum_hits if h.drum_type == 'snare')}
    Hihat: {sum(1 for h in drum_hits if h.drum_type == 'hihat')}

  Ground truth by type:
    Kick: {len(gt_drum_hits.get('kick', []))}
    Snare: {len(gt_drum_hits.get('snare', []))}
    Hihat: {len(gt_drum_hits.get('hihat', []))}
"""

    fig.text(0.98, 0.5, stats_text,
             transform=fig.transFigure,
             fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize pipeline predictions on individual samples')
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
                       help='Which split to visualize')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str,
                       default='beatbox2drums/visualizations/pipeline',
                       help='Output directory for visualizations')
    parser.add_argument('--onset-threshold', type=float, default=0.5)
    parser.add_argument('--classifier-threshold', type=float, default=0.3)
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots, only save them')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    # Get samples from requested split
    samples = [s for s in manifest['samples'] if s['split'] == args.split]

    if len(samples) == 0:
        print(f"No samples found for split: {args.split}")
        return

    # Limit number of samples
    samples = samples[:args.num_samples]

    print(f"\nVisualizing {len(samples)} samples from {args.split} split...")
    print("="*70)

    # Create pipeline
    pipeline = Beatbox2DrumsPipeline(
        onset_checkpoint_path=args.onset_checkpoint,
        classifier_checkpoint_path=args.classifier_checkpoint,
        onset_threshold=args.onset_threshold,
        classifier_confidence_threshold=args.classifier_threshold
    )

    # Process each sample
    for idx, sample in enumerate(samples, 1):
        audio_path = sample['audio_path']
        label_path = sample['label_path']

        print(f"\n[{idx}/{len(samples)}] Processing: {Path(audio_path).name}")

        # Create output path
        output_path = output_dir / f"{Path(audio_path).stem}_visualization.png"

        try:
            visualize_predictions(
                audio_path=audio_path,
                label_path=label_path,
                pipeline=pipeline,
                output_path=str(output_path),
                show=(not args.no_show)
            )
            print(f"✓ Saved to: {output_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"Visualization complete! Saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
