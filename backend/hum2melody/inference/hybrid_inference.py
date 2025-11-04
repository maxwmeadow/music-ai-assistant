#!/usr/bin/env python3
"""
Hybrid Hum2Melody Inference

Combines the best of both worlds:
- Combined model's pitch predictions (98% accuracy)
- Multi-band onset detector (88% precision)

Usage:
    python hybrid_inference.py --audio my_hum.wav --output notes.json
    python hybrid_inference.py --audio my_hum.wav --visualize  # Show piano roll
"""

import sys
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir / 'hum2melody_package'))
sys.path.insert(0, str(script_dir))

import argparse
import json
import numpy as np
import torch
import librosa
from typing import List, Tuple

# Import combined model
from models.combined_model_loader import load_combined_model

# Import onset detector (from parent directory)
try:
    from data.onset_offset_detector import detect_onsets_offsets
except ImportError:
    # Try absolute import
    import importlib.util
    onset_path = script_dir / 'data' / 'onset_offset_detector.py'
    spec = importlib.util.spec_from_file_location("onset_offset_detector", onset_path)
    onset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(onset_module)
    detect_onsets_offsets = onset_module.detect_onsets_offsets


class HybridHum2Melody:
    """
    Hybrid inference combining:
    - Combined model's pitch predictions
    - Multi-band spectral flux onset detection
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 onset_high: float = 0.30, onset_low: float = 0.10,
                 onset_offset_high: float = 0.30, onset_offset_low: float = 0.10):
        """
        Initialize hybrid predictor.

        Args:
            checkpoint_path: Path to combined model checkpoint
            device: 'cpu' or 'cuda'
            onset_high: High threshold for onset detection
            onset_low: Low threshold for onset detection (hysteresis)
        """
        print(f"\n{'='*70}")
        print("INITIALIZING HYBRID HUM2MELODY")
        print(f"{'='*70}")

        # Load combined model
        print(f"Loading combined model from: {checkpoint_path}")
        self.model = load_combined_model(checkpoint_path, device=device)
        self.model.eval()
        self.device = device
        print(f"✓ Model loaded successfully")

        # Onset detection parameters
        self.onset_high = onset_high
        self.onset_low = onset_low
        self.offset_high = onset_offset_high
        self.offset_low = onset_offset_low

        print(f"\nOnset Detection Settings:")
        print(f"  High threshold: {onset_high}")
        print(f"  Low threshold: {onset_low}")
        print(f"{'='*70}\n")

    def preprocess_audio(self, audio_path: str, target_frames: int = 500):
        """
        Preprocess audio for model input.

        Args:
            audio_path: Path to audio file
            target_frames: Number of frames for model (500 = ~16 seconds)

        Returns:
            audio: Raw audio (for onset detection)
            cqt_tensor: (1, 1, 500, 88) - CQT input
            extras_tensor: (1, 1, 500, 24) - Zero-filled extras
        """
        print(f"Preprocessing: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"  Loaded: {len(audio)} samples ({len(audio)/16000:.2f}s)")

        # Normalize
        audio = audio / np.max(np.abs(audio))

        # Extract CQT for model
        cqt = librosa.cqt(
            y=audio,
            sr=16000,
            hop_length=512,
            n_bins=88,
            bins_per_octave=12,
            fmin=27.5
        )

        # Normalize CQT
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_normalized = (cqt_db + 80) / 80
        cqt_normalized = np.clip(cqt_normalized, 0, 1)

        # Pad or truncate
        if cqt_normalized.shape[1] < target_frames:
            pad = target_frames - cqt_normalized.shape[1]
            cqt_normalized = np.pad(cqt_normalized, ((0, 0), (0, pad)))
        else:
            cqt_normalized = cqt_normalized[:, :target_frames]

        # Convert to tensors
        cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)
        extras_tensor = torch.zeros(1, 1, target_frames, 24)

        print(f"  CQT shape: {cqt_tensor.shape}")

        return audio, cqt_tensor, extras_tensor

    def detect_segments(self, audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
        """
        Detect note segments using multi-band onset detector.

        Args:
            audio: Raw audio signal
            sr: Sample rate

        Returns:
            List of (start_time, end_time) tuples
        """
        print(f"\nDetecting segments with multi-band onset detector...")

        segments = detect_onsets_offsets(
            audio,
            sr=sr,
            hop_length=512,
            onset_high=self.onset_high,
            onset_low=self.onset_low,
            offset_high=self.offset_high,
            offset_low=self.offset_low,
            min_note_len=0.05  # Minimum 50ms note
        )

        print(f"  Detected {len(segments)} segments")
        return segments

    def get_pitch_predictions(self, cqt: torch.Tensor, extras: torch.Tensor):
        """
        Get frame-level pitch predictions from combined model.

        Args:
            cqt: CQT tensor (1, 1, 500, 88)
            extras: Extras tensor (1, 1, 500, 24)

        Returns:
            frame_probs: (125, 88) pitch probabilities
            onset_probs: (125, 1) onset probabilities (not used)
            voicing: (125,) voicing probabilities
        """
        print(f"\nRunning pitch model inference...")

        cqt = cqt.to(self.device)
        extras = extras.to(self.device)

        with torch.no_grad():
            frame, onset, offset, f0 = self.model(cqt, extras)

        # Convert to probabilities
        frame_probs = torch.sigmoid(frame)[0].cpu().numpy()  # (125, 88)
        onset_probs = torch.sigmoid(onset)[0].cpu().numpy()  # (125, 1)
        voicing = torch.sigmoid(f0[:, :, 1])[0].cpu().numpy()  # (125,)

        print(f"  Frame probs shape: {frame_probs.shape}")
        print(f"  Voicing range: {voicing.min():.3f} - {voicing.max():.3f}")

        return frame_probs, onset_probs, voicing

    def aggregate_segment_pitch(self, frame_probs: np.ndarray, voicing: np.ndarray,
                                start_time: float, end_time: float,
                                output_frame_rate: float = 7.8125) -> Tuple[int, float]:
        """
        Aggregate pitch predictions within a segment.

        Args:
            frame_probs: (125, 88) pitch probabilities
            voicing: (125,) voicing probabilities
            start_time: Segment start (seconds)
            end_time: Segment end (seconds)
            output_frame_rate: Model output frame rate (Hz)
                             = input_frame_rate / downsample_factor
                             = 31.25 / 4 = 7.8125 Hz

        Returns:
            midi_note: Predicted MIDI note (21-108)
            confidence: Confidence score
        """
        # Convert times to model output frames
        start_frame = int(start_time * output_frame_rate)
        end_frame = int(end_time * output_frame_rate)

        # Clip to valid range
        start_frame = max(0, min(start_frame, len(frame_probs) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(frame_probs)))

        # Get frames in this segment
        segment_probs = frame_probs[start_frame:end_frame]  # (n_frames, 88)
        segment_voicing = voicing[start_frame:end_frame]  # (n_frames,)

        if len(segment_probs) == 0:
            return 0, 0.0  # Silence

        # Weight by voicing
        weighted_probs = segment_probs * segment_voicing[:, np.newaxis]

        # Average over time
        avg_probs = weighted_probs.mean(axis=0)  # (88,)

        # Get pitch with highest probability
        pitch_idx = avg_probs.argmax()
        confidence = float(avg_probs[pitch_idx])

        # Convert to MIDI
        midi_note = pitch_idx + 21

        return midi_note, confidence

    def predict(self, audio_path: str, min_confidence: float = 0.1) -> List[dict]:
        """
        Predict notes from audio file.

        Args:
            audio_path: Path to audio file
            min_confidence: Minimum confidence to keep note

        Returns:
            List of note dictionaries with:
                - start: Start time (seconds)
                - end: End time (seconds)
                - duration: Duration (seconds)
                - midi: MIDI note number (21-108)
                - note: Note name (e.g., 'C4')
                - confidence: Confidence score (0-1)
        """
        print(f"\n{'='*70}")
        print(f"PREDICTING NOTES FROM: {audio_path}")
        print(f"{'='*70}")

        # 1. Preprocess
        audio, cqt, extras = self.preprocess_audio(audio_path)

        # 2. Detect segments
        segments = self.detect_segments(audio)

        # 3. Get pitch predictions
        frame_probs, onset_probs, voicing = self.get_pitch_predictions(cqt, extras)

        # Calculate max time covered by model
        # 125 frames at 7.8125 Hz = 16 seconds
        max_analysis_time = len(frame_probs) / 7.8125

        # 4. Aggregate pitch for each segment
        print(f"\nAggregating pitches for {len(segments)} segments...")

        # Filter segments to only those within analysis window
        valid_segments = [(s, e) for s, e in segments if s < max_analysis_time]
        if len(valid_segments) < len(segments):
            print(f"  ⚠️  Warning: {len(segments) - len(valid_segments)} segments beyond {max_analysis_time:.1f}s analysis window (ignored)")
            print(f"     Model only analyzes first ~16 seconds of audio")

        notes = []

        for start_time, end_time in valid_segments:
            midi_note, confidence = self.aggregate_segment_pitch(
                frame_probs, voicing, start_time, end_time
            )

            # Filter by confidence
            if confidence >= min_confidence:
                note_name = librosa.midi_to_note(midi_note)
                notes.append({
                    'start': float(start_time),
                    'end': float(end_time),
                    'duration': float(end_time - start_time),
                    'midi': int(midi_note),
                    'note': note_name,
                    'confidence': float(confidence)
                })

        print(f"  Extracted {len(notes)} notes (filtered by confidence >= {min_confidence})")
        print(f"{'='*70}\n")

        return notes


def print_notes(notes: List[dict]):
    """Print notes in a readable format."""
    if not notes:
        print("No notes detected!")
        return

    print(f"\n{'='*70}")
    print("DETECTED NOTES")
    print(f"{'='*70}")
    print(f"{'Start (s)':<10} {'End (s)':<10} {'Duration':<10} {'Note':<8} {'MIDI':<6} {'Conf':<6}")
    print(f"{'-'*70}")

    for note in notes:
        print(f"{note['start']:<10.2f} {note['end']:<10.2f} {note['duration']:<10.2f} "
              f"{note['note']:<8} {note['midi']:<6} {note['confidence']:<6.3f}")

    print(f"{'='*70}\n")


def visualize_notes(notes: List[dict], output_path: str = None):
    """Create a simple piano roll visualization."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    if not notes:
        print("No notes to visualize")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get MIDI range
    midi_notes = [n['midi'] for n in notes]
    min_midi = min(midi_notes) - 2
    max_midi = max(midi_notes) + 2

    # Draw notes as rectangles
    colors = plt.cm.viridis(np.linspace(0, 1, len(notes)))

    for i, note in enumerate(notes):
        rect = patches.Rectangle(
            (note['start'], note['midi'] - 0.4),
            note['duration'],
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=colors[i],
            alpha=0.7
        )
        ax.add_patch(rect)

        # Add note name
        if note['duration'] > 0.1:
            ax.text(
                note['start'] + note['duration']/2,
                note['midi'],
                note['note'],
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold'
            )

    # Format plot
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('MIDI Note', fontsize=12)
    ax.set_title('Detected Notes (Piano Roll)', fontsize=14, fontweight='bold')

    # Set axis limits
    max_time = max(n['end'] for n in notes) + 0.5
    ax.set_xlim(0, max_time)
    ax.set_ylim(min_midi, max_midi)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add note names on y-axis
    midi_ticks = range(min_midi, max_midi + 1)
    note_names = [librosa.midi_to_note(m) for m in midi_ticks]
    ax.set_yticks(midi_ticks)
    ax.set_yticklabels(note_names, fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Hum2Melody Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python hybrid_inference.py --audio my_hum.wav

  # Save to JSON
  python hybrid_inference.py --audio my_hum.wav --output notes.json

  # Visualize piano roll
  python hybrid_inference.py --audio my_hum.wav --visualize

  # Custom thresholds
  python hybrid_inference.py --audio my_hum.wav --onset-high 0.4 --onset-low 0.15
        """
    )

    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file (.wav, .mp3, etc.)')
    parser.add_argument('--checkpoint', type=str,
                       default='hum2melody_package/checkpoints/combined_hum2melody_full.pth',
                       help='Path to combined model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--onset-high', type=float, default=0.30,
                       help='Onset detection high threshold')
    parser.add_argument('--onset-low', type=float, default=0.10,
                       help='Onset detection low threshold')
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Minimum confidence to keep note')
    parser.add_argument('--visualize', action='store_true',
                       help='Show piano roll visualization')
    parser.add_argument('--viz-output', type=str, default=None,
                       help='Save visualization to file')

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return 1

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print(f"Please ensure the checkpoint exists at: {args.checkpoint}")
        return 1

    # Create predictor
    predictor = HybridHum2Melody(
        checkpoint_path=args.checkpoint,
        device=args.device,
        onset_high=args.onset_high,
        onset_low=args.onset_low
    )

    # Predict notes
    notes = predictor.predict(args.audio, min_confidence=args.min_confidence)

    # Print results
    print_notes(notes)

    # Save to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            'audio_file': str(Path(args.audio).name),
            'num_notes': len(notes),
            'notes': notes
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✓ Results saved to: {output_path}")

    # Visualize
    if args.visualize or args.viz_output:
        visualize_notes(notes, output_path=args.viz_output)

    print("\nDone! Ready to use notes in your application.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
