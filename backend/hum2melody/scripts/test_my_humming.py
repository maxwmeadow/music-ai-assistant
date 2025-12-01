#!/usr/bin/env python3
"""
Simple script for testing the hybrid system on your humming recordings.

Usage:
    # Single file
    python test_my_humming.py my_hum.wav

    # Multiple files
    python test_my_humming.py hum1.wav hum2.wav hum3.wav

    # All WAV files in a directory
    python test_my_humming.py humming_samples/*.wav
"""

import sys
import argparse
from pathlib import Path
import json

# Add to path
script_dir = Path(__file__).parent.resolve()
package_dir = script_dir.parent  # backend/hum2melody
sys.path.insert(0, str(package_dir))
sys.path.insert(0, str(package_dir / 'inference'))

from inference.hybrid_inference_chunked import ChunkedHybridHum2Melody


def test_single_file(audio_path, predictor, visualize=False, save_json=False):
    """Test on a single humming recording."""
    print(f"\n{'='*70}")
    print(f"TESTING: {audio_path}")
    print(f"{'='*70}")

    try:
        # Run prediction (uses chunking for long audio)
        notes = predictor.predict_chunked(audio_path)

        # Print summary
        print(f"\n✅ Detected {len(notes)} notes:")
        if notes:
            for note in notes:
                print(f"   {note['note']:4s} at {note['start']:.2f}s "
                      f"(duration: {note['duration']:.2f}s, "
                      f"confidence: {note['confidence']:.3f})")
        else:
            print("   ⚠️  No notes detected!")
            print("   Try lowering thresholds with:")
            print("      --onset-high 0.20 --onset-low 0.05 --min-confidence 0.05")

        # Save JSON if requested
        if save_json and notes:
            output_path = Path(audio_path).stem + '_notes.json'
            with open(output_path, 'w') as f:
                json.dump({
                    'audio_file': str(audio_path),
                    'num_notes': len(notes),
                    'notes': notes
                }, f, indent=2)
            print(f"\n📝 Saved results to: {output_path}")

        # Visualize if requested
        if visualize and notes:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig, ax = plt.subplots(figsize=(12, 4))

                # Plot each note as a colored rectangle
                for i, note in enumerate(notes):
                    color = plt.cm.viridis(note['confidence'])
                    rect = patches.Rectangle(
                        (note['start'], note['midi']-0.4),
                        note['duration'], 0.8,
                        linewidth=1, edgecolor='black',
                        facecolor=color, alpha=0.8
                    )
                    ax.add_patch(rect)
                    ax.text(note['start'] + note['duration']/2, note['midi'],
                           note['note'], ha='center', va='center',
                           fontsize=8, fontweight='bold')

                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('MIDI Note')
                ax.set_title(f"Detected Notes: {Path(audio_path).name}")
                ax.grid(True, alpha=0.3)

                # Set y-axis to show note names
                import librosa
                midi_min = min(n['midi'] for n in notes) - 2
                midi_max = max(n['midi'] for n in notes) + 2
                ax.set_ylim(midi_min, midi_max)

                # Set x-axis limits to audio duration
                time_max = max(n['start'] + n['duration'] for n in notes) + 0.5
                ax.set_xlim(0, time_max)

                plt.tight_layout()
                output_img = Path(audio_path).stem + '_notes.png'
                plt.savefig(output_img, dpi=150, bbox_inches='tight')
                print(f"📊 Saved visualization to: {output_img}")
                plt.close()

            except ImportError:
                print("⚠️  matplotlib not available, skipping visualization")

        return notes

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test hybrid hum2melody on your humming recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single file
  python test_my_humming.py my_hum.wav

  # Test with visualization
  python test_my_humming.py my_hum.wav --visualize

  # Test multiple files with JSON output
  python test_my_humming.py hum*.wav --save-json

  # Lower thresholds for more sensitive detection
  python test_my_humming.py my_hum.wav --onset-high 0.20 --min-confidence 0.05
        """
    )

    parser.add_argument(
        'audio_files',
        nargs='+',
        help='Audio file(s) to test (.wav, .mp3, etc.)'
    )
    parser.add_argument(
        '--checkpoint',
        default='../checkpoints/combined_hum2melody_full.pth',
        help='Path to combined model checkpoint (relative to scripts dir)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--onset-high',
        type=float,
        default=0.30,
        help='High threshold for onset detection (default: 0.30)'
    )
    parser.add_argument(
        '--onset-low',
        type=float,
        default=0.10,
        help='Low threshold for onset detection (default: 0.10)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.10,
        help='Minimum confidence to keep a note (default: 0.10)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create piano roll visualization (requires matplotlib)'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to JSON files'
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    # Initialize predictor (once for all files)
    print("\n🎵 Initializing Hybrid Hum2Melody...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Device: {args.device}")
    print(f"   Onset thresholds: high={args.onset_high}, low={args.onset_low}")
    print(f"   Min confidence: {args.min_confidence}")

    predictor = ChunkedHybridHum2Melody(
        checkpoint_path=args.checkpoint,
        device=args.device,
        onset_high=args.onset_high,
        onset_low=args.onset_low,
        chunk_duration=15.0,  # Process 15s chunks
        overlap=1.0  # 1s overlap
    )

    # Test each file
    results = {}
    for audio_file in args.audio_files:
        if not Path(audio_file).exists():
            print(f"\n⚠️  File not found: {audio_file}")
            continue

        notes = test_single_file(
            audio_file,
            predictor,
            visualize=args.visualize,
            save_json=args.save_json
        )

        if notes is not None:
            results[audio_file] = len(notes)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Tested {len(results)} file(s):")
    for audio_file, num_notes in results.items():
        print(f"  {Path(audio_file).name}: {num_notes} notes")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
