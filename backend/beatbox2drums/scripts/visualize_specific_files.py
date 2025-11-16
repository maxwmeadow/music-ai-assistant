#!/usr/bin/env python3
"""
Generate visualizations for specific audio files by name.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import json
import torch
import librosa

from generate_visualizations_standalone import (
    load_test_sample, classify_onset, plot_drum_classification_results, compute_accuracy
)
from models.drum_classifier import DrumClassifierCNN


def run_inference(model, audio_path, ground_truth, device='cpu', min_confidence=0.3):
    """Run inference on all ground truth onsets."""
    audio, sr = librosa.load(audio_path, sr=None)

    predictions = []
    for gt in ground_truth:
        onset_time = gt['onset_time']
        result = classify_onset(model, audio, sr, onset_time, device=device, min_confidence=min_confidence)
        if result:
            predictions.append(result)

    return predictions


def find_label_for_audio(manifest_path, audio_filename):
    """Find label file path for given audio filename (e.g., '3999.wav')."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Search in test split
    for entry in manifest['test']:
        label_path = Path(entry['label_path'])
        # Check if this label corresponds to the audio file
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        audio_path = label_data['audio_path']
        if Path(audio_path).name == audio_filename or Path(audio_path).stem == Path(audio_filename).stem:
            return label_path

    return None


def main():
    import argparse

    script_dir = Path(__file__).parent.absolute()
    package_dir = script_dir.parent

    default_checkpoint = package_dir / 'checkpoints' / 'beatbox2drums_best.pth'
    default_manifest = package_dir.parent / 'beatbox2drums' / 'dataset' / 'combined' / 'manifest.json'
    default_output = package_dir / 'examples' / 'visualizations' / 'high_accuracy'

    parser = argparse.ArgumentParser(description='Generate visualizations for specific audio files')
    parser.add_argument('audio_files', nargs='+',
                       help='Audio filenames (e.g., 3999.wav 1225.wav 943.wav)')
    parser.add_argument('--manifest', default=str(default_manifest),
                       help='Path to dataset manifest')
    parser.add_argument('--checkpoint', default=str(default_checkpoint),
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', default=str(default_output),
                       help='Output directory for images')

    args = parser.parse_args()

    print("="*70)
    print("Generating Visualizations for Specific Files")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    model = DrumClassifierCNN(num_classes=3, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✓ Model loaded successfully\n")

    # Process each audio file
    for i, audio_filename in enumerate(args.audio_files, 1):
        print(f"\n[{i}/{len(args.audio_files)}] Processing: {audio_filename}")

        # Find label file
        label_path = find_label_for_audio(args.manifest, audio_filename)

        if not label_path:
            print(f"  ✗ Could not find label file for: {audio_filename}")
            continue

        print(f"  Found label: {label_path.name}")

        try:
            # Load sample
            audio_path, ground_truth = load_test_sample(str(label_path))

            # Run inference
            predictions = run_inference(model, audio_path, ground_truth)

            # Compute accuracy
            metrics = compute_accuracy(ground_truth, predictions)
            accuracy = metrics['overall_accuracy']

            # Generate filename
            audio_name = Path(audio_path).stem
            accuracy_pct = int(accuracy * 100)
            output_path = output_dir / f"{audio_name}_acc{accuracy_pct:03d}.png"

            # Create visualization
            plot_drum_classification_results(
                audio_path,
                ground_truth,
                predictions,
                output_path=str(output_path)
            )

            print(f"  ✓ Accuracy: {accuracy*100:.1f}%")
            print(f"  ✓ Saved: {output_path.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("Done!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
