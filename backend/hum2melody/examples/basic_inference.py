#!/usr/bin/env python3
"""
Basic inference example - simplest way to use Hybrid Hum2Melody

Usage:
    python basic_inference.py my_humming.wav
    python basic_inference.py my_humming.wav --visualize
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.hybrid_inference_chunked import ChunkedHybridHum2Melody


def main():
    parser = argparse.ArgumentParser(description="Basic Hybrid Hum2Melody inference")
    parser.add_argument("audio_file", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--checkpoint", default="checkpoints/combined_hum2melody_full.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization (requires matplotlib)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to use")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Check audio file exists
    if not Path(args.audio_file).exists():
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        return 1

    # Initialize model
    print(f"\n🎵 Initializing Hybrid Hum2Melody...")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Device: {args.device}")

    model = ChunkedHybridHum2Melody(
        checkpoint_path=args.checkpoint,
        device=args.device,
        min_confidence=0.25,  # Recommended for production
    )

    # Run inference
    print(f"\n🎤 Processing: {args.audio_file}")
    notes = model.predict_chunked(args.audio_file)

    # Print results
    print(f"\n✅ Detected {len(notes)} notes:")
    if notes:
        for i, note in enumerate(notes, 1):
            print(f"   {i:2d}. {note['note']:4s} at {note['start']:6.2f}s "
                  f"(duration: {note['duration']:.2f}s, "
                  f"confidence: {note['confidence']:.3f})")
    else:
        print("   ⚠️  No notes detected!")
        print("   Try lowering --min-confidence or checking audio quality")

    # Save to JSON if requested
    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'audio_file': str(args.audio_file),
                'num_notes': len(notes),
                'notes': notes
            }, f, indent=2)
        print(f"\n💾 Saved results to: {output_path}")

    # Visualize if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            if not notes:
                print("\n⚠️  Cannot visualize: No notes detected")
                return 0

            fig, ax = plt.subplots(figsize=(12, 4))

            for note in notes:
                color = plt.cm.viridis(note['confidence'])
                rect = patches.Rectangle(
                    (note['start'], note['midi'] - 0.4),
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
            ax.set_title(f"Detected Notes: {Path(args.audio_file).name}")
            ax.grid(True, alpha=0.3)

            midi_min = min(n['midi'] for n in notes) - 2
            midi_max = max(n['midi'] for n in notes) + 2
            ax.set_ylim(midi_min, midi_max)

            time_max = max(n['start'] + n['duration'] for n in notes) + 0.5
            ax.set_xlim(0, time_max)

            plt.tight_layout()

            output_img = Path(args.audio_file).stem + '_notes.png'
            plt.savefig(output_img, dpi=150, bbox_inches='tight')
            print(f"\n📊 Saved visualization to: {output_img}")
            plt.close()

        except ImportError:
            print("\n⚠️  matplotlib not available, skipping visualization")
            print("   Install with: pip install matplotlib")

    return 0


if __name__ == '__main__':
    sys.exit(main())
