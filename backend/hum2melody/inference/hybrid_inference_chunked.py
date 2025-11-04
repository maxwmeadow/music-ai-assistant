#!/usr/bin/env python3
"""
Hybrid Hum2Melody Inference with Chunking Support

Extends hybrid_inference.py to process long audio files by chunking.

Usage:
    python hybrid_inference_chunked.py --audio long_recording.wav
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir / 'hum2melody_package'))
sys.path.insert(0, str(script_dir))

import numpy as np
import torch
import librosa
from typing import List, Tuple
from hybrid_inference import HybridHum2Melody, print_notes


class ChunkedHybridHum2Melody(HybridHum2Melody):
    """
    Extended version that processes long audio files in chunks.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 onset_high: float = 0.30, onset_low: float = 0.10,
                 offset_high: float = 0.30, offset_low: float = 0.10,
                 chunk_duration: float = 15.0,  # Process 15s chunks
                 overlap: float = 1.0):  # 1s overlap between chunks
        """
        Initialize chunked hybrid system.

        Args:
            chunk_duration: Duration of each chunk in seconds (default: 15s)
            overlap: Overlap between chunks in seconds (default: 1s)
        """
        super().__init__(checkpoint_path, device, onset_high, onset_low,
                        offset_high, offset_low)

        self.chunk_duration = chunk_duration
        self.overlap = overlap

        print(f"\nChunking enabled:")
        print(f"  Chunk duration: {chunk_duration}s")
        print(f"  Overlap: {overlap}s")

    def predict_chunked(self, audio_path: str, min_confidence: float = 0.1) -> List[dict]:
        """
        Predict notes from long audio by processing in chunks.

        Args:
            audio_path: Path to audio file
            min_confidence: Minimum confidence to keep note

        Returns:
            List of note dictionaries
        """
        print(f"\n{'='*70}")
        print(f"PREDICTING NOTES (CHUNKED): {audio_path}")
        print(f"{'='*70}")

        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration = len(audio) / sr
        print(f"Preprocessing: {audio_path}")
        print(f"  Loaded: {len(audio)} samples ({audio_duration:.2f}s)")

        # Normalize
        audio = audio / np.max(np.abs(audio))

        # If audio is short enough, use normal processing
        if audio_duration <= self.chunk_duration:
            print(f"  Audio fits in single chunk, using standard processing")
            return self.predict(audio_path, min_confidence)

        # Calculate chunks
        hop_size = self.chunk_duration - self.overlap
        num_chunks = int(np.ceil((audio_duration - self.overlap) / hop_size))

        print(f"\n  Audio is {audio_duration:.1f}s, processing in {num_chunks} chunks")
        print(f"  Chunk size: {self.chunk_duration}s, hop: {hop_size}s")

        # Process each chunk
        all_notes = []

        for i in range(num_chunks):
            start_time = i * hop_size
            end_time = min(start_time + self.chunk_duration, audio_duration)

            # Extract chunk
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_audio = audio[start_sample:end_sample]

            print(f"\n  Chunk {i+1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")

            # Process chunk
            chunk_notes = self._process_chunk(
                chunk_audio,
                chunk_offset=start_time,
                min_confidence=min_confidence
            )

            # Merge notes from this chunk
            for note in chunk_notes:
                # Skip notes in overlap region if we already have them
                if i > 0 and note['start'] < start_time + self.overlap / 2:
                    continue  # Skip, likely duplicate from previous chunk

                all_notes.append(note)

        print(f"\n{'='*70}")
        print(f"  Total notes extracted: {len(all_notes)}")
        print(f"{'='*70}\n")

        return all_notes

    def _process_chunk(self, chunk_audio: np.ndarray,
                      chunk_offset: float,
                      min_confidence: float) -> List[dict]:
        """
        Process a single chunk of audio.

        Args:
            chunk_audio: Audio chunk (numpy array)
            chunk_offset: Time offset of this chunk in original audio (seconds)
            min_confidence: Minimum confidence threshold

        Returns:
            List of notes with adjusted timestamps
        """
        sr = 16000

        # Extract CQT
        hop_length = 512
        cqt = librosa.cqt(
            y=chunk_audio,
            sr=sr,
            hop_length=hop_length,
            n_bins=88,
            bins_per_octave=12,
            fmin=27.5
        )

        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_normalized = (cqt_db + 80) / 80
        cqt_normalized = np.clip(cqt_normalized, 0, 1)

        # Pad to 500 frames
        target_frames = 500
        if cqt_normalized.shape[1] < target_frames:
            pad = target_frames - cqt_normalized.shape[1]
            cqt_normalized = np.pad(cqt_normalized, ((0, 0), (0, pad)))
        else:
            cqt_normalized = cqt_normalized[:, :target_frames]

        cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)
        extras = torch.zeros(1, 1, target_frames, 24)

        # Detect segments in this chunk
        segments = self.detect_segments(chunk_audio, sr)

        # Get pitch predictions
        frame_probs, _, voicing = self.get_pitch_predictions(cqt_tensor, extras)

        # Calculate max analysis time
        max_analysis_time = len(frame_probs) / 7.8125

        # Aggregate pitches
        notes = []
        valid_segments = [(s, e) for s, e in segments if s < max_analysis_time]

        for start_time, end_time in valid_segments:
            midi_note, confidence = self.aggregate_segment_pitch(
                frame_probs, voicing, start_time, end_time
            )

            if confidence >= min_confidence:
                note_name = librosa.midi_to_note(midi_note)
                notes.append({
                    'start': float(start_time + chunk_offset),  # Adjust for chunk offset
                    'end': float(end_time + chunk_offset),
                    'duration': float(end_time - start_time),
                    'midi': int(midi_note),
                    'note': note_name,
                    'confidence': float(confidence)
                })

        print(f"    Detected {len(segments)} segments, extracted {len(notes)} notes")

        return notes


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Hybrid Hum2Melody with chunking for long audio"
    )
    parser.add_argument('--audio', required=True, help='Audio file to process')
    parser.add_argument(
        '--checkpoint',
        default='hum2melody_package/checkpoints/combined_hum2melody_full.pth',
        help='Model checkpoint'
    )
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--onset-high', type=float, default=0.30)
    parser.add_argument('--onset-low', type=float, default=0.10)
    parser.add_argument('--min-confidence', type=float, default=0.10)
    parser.add_argument('--chunk-duration', type=float, default=15.0,
                       help='Chunk duration in seconds (default: 15s)')
    parser.add_argument('--overlap', type=float, default=1.0,
                       help='Overlap between chunks in seconds (default: 1s)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')

    args = parser.parse_args()

    # Initialize predictor
    predictor = ChunkedHybridHum2Melody(
        checkpoint_path=args.checkpoint,
        device=args.device,
        onset_high=args.onset_high,
        onset_low=args.onset_low,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap
    )

    # Predict
    notes = predictor.predict_chunked(
        args.audio,
        min_confidence=args.min_confidence
    )

    # Print results
    print_notes(notes)

    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'audio_file': args.audio,
                'num_notes': len(notes),
                'notes': notes
            }, f, indent=2)
        print(f"\n✅ Saved to: {args.output}")

    # Visualize
    if args.visualize and notes:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots(figsize=(16, 6))

            for note in notes:
                color = plt.cm.viridis(note['confidence'])
                rect = patches.Rectangle(
                    (note['start'], note['midi']-0.4),
                    note['duration'], 0.8,
                    linewidth=1, edgecolor='black',
                    facecolor=color, alpha=0.8
                )
                ax.add_patch(rect)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('MIDI Note')
            ax.set_title(f"Detected Notes: {Path(args.audio).name}")
            ax.grid(True, alpha=0.3)

            if notes:
                midi_min = min(n['midi'] for n in notes) - 2
                midi_max = max(n['midi'] for n in notes) + 2
                ax.set_ylim(midi_min, midi_max)
                ax.set_xlim(0, max(n['end'] for n in notes) + 1)

            plt.tight_layout()
            output_img = Path(args.audio).stem + '_chunked_notes.png'
            plt.savefig(output_img, dpi=150, bbox_inches='tight')
            print(f"📊 Saved visualization: {output_img}")
            plt.close()

        except ImportError:
            print("⚠️  matplotlib not available")

    return 0


if __name__ == '__main__':
    sys.exit(main())
