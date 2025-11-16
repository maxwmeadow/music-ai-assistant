#!/usr/bin/env python3
"""
Simple example of using the Beatbox2Drums pipeline.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline


def main():
    print("="*70)
    print("Beatbox2Drums Pipeline Example")
    print("="*70)
    print()

    # Paths to checkpoints
    onset_checkpoint = Path("beatbox2drums/cnn_onset_checkpoints/best_model.pth")
    classifier_checkpoint = Path("beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth")

    # Check if checkpoints exist
    if not onset_checkpoint.exists():
        print(f"Error: Onset checkpoint not found: {onset_checkpoint}")
        print("Please train the onset detector first.")
        return

    if not classifier_checkpoint.exists():
        print(f"Error: Classifier checkpoint not found: {classifier_checkpoint}")
        print("Please train the classifier first.")
        return

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = Beatbox2DrumsPipeline(
        onset_checkpoint_path=str(onset_checkpoint),
        classifier_checkpoint_path=str(classifier_checkpoint),
        onset_threshold=0.5,
        onset_peak_delta=0.05,  # 50ms NMS window
        classifier_confidence_threshold=0.3
    )

    # Find a test audio file
    test_files = list(Path("beatbox2drums/dataset/combined/audio").glob("*.wav"))

    if not test_files:
        print("Error: No test audio files found in beatbox2drums/dataset/combined/audio/")
        return

    # Use the first file
    test_file = test_files[0]
    print(f"\nTesting with: {test_file.name}")
    print("-"*70)

    # Run prediction with details
    results = pipeline.predict(str(test_file), return_details=True)
    drum_hits = results['drum_hits']

    # Print results
    print("\nResults:")
    print(f"  Total onsets detected: {results['total_onsets']}")
    print(f"  Drum hits after filtering: {len(drum_hits)}")
    print(f"  Rejected (low confidence): {results['rejected_count']}")
    print()

    # Get statistics
    stats = pipeline.get_statistics(drum_hits)

    print("Detected Drum Hits by Type:")
    for drum_type in ['kick', 'snare', 'hihat']:
        count = stats['by_type'][drum_type]
        avg_conf = stats['confidence_by_type'][drum_type]
        print(f"  {drum_type.capitalize()}: {count} hits (avg confidence: {avg_conf:.3f})")

    print(f"\nOverall average confidence: {stats['avg_confidence']:.3f}")
    print()

    # Show first 15 drum hits
    print("First 15 drum hits:")
    print(f"{'Time (s)':<12} {'Type':<10} {'Confidence':<12}")
    print("-"*40)
    for hit in drum_hits[:15]:
        print(f"{hit.time:<12.3f} {hit.drum_type:<10} {hit.confidence:<12.3f}")

    print()
    print("="*70)
    print("Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()
