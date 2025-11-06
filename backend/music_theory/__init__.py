"""
Music Theory Post-Processing Module

Transforms hum2melody note predictions into musically coherent melodies.

Pipeline (in order):
1. Convert hum2melody note format to internal format
2. Detect tempo from note onset times
3. Detect key (from raw, potentially off-key notes)
4. Quantize pitches to detected key
5. Quantize rhythm to tempo grid
6. (Optional) Infer chord progression

This module is the music theory layer that sits between
hum2melody output and the final musical IR.

Note: Onset/offset detection is now handled by the hum2melody model itself.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .tempo_detector import TempoDetector
from .key_detector import KeyDetector
from .pitch_quantizer import PitchQuantizer
from .rhythmic_quantizer import RhythmicQuantizer
from .chord_analyzer import ChordAnalyzer


def convert_hum2melody_to_internal(notes: List[Dict]) -> List[Dict]:
    """
    Convert hum2melody note format to internal music_theory format.

    Hum2melody format:
        {'start', 'end', 'duration', 'midi', 'note', 'confidence'}

    Internal format:
        {'pitch', 'start', 'duration', 'confidence'}

    Args:
        notes: List of notes from hum2melody predictor

    Returns:
        List of notes in internal format
    """
    internal_notes = []
    for note in notes:
        internal_notes.append({
            'pitch': note['midi'],
            'start': note['start'],
            'duration': note['duration'],
            'confidence': note['confidence']
        })
    return internal_notes


def convert_internal_to_hum2melody(notes: List[Dict]) -> List[Dict]:
    """
    Convert internal music_theory format back to hum2melody format.

    Internal format:
        {'pitch', 'start', 'duration', 'confidence'}

    Hum2melody format:
        {'start', 'end', 'duration', 'midi', 'note', 'confidence'}

    Args:
        notes: List of notes in internal format

    Returns:
        List of notes in hum2melody format
    """
    import librosa

    hum2melody_notes = []
    for note in notes:
        hum2melody_notes.append({
            'start': note['start'],
            'end': note['start'] + note['duration'],
            'duration': note['duration'],
            'midi': int(note['pitch']),
            'note': librosa.midi_to_note(int(note['pitch'])),
            'confidence': note['confidence']
        })
    return hum2melody_notes


class MusicTheoryProcessor:
    """
    Main facade for music theory post-processing pipeline.

    Transforms hum2melody predictions into musically coherent melody with:
    - Detected musical key
    - Pitch quantization (strict to scale)
    - Rhythmic quantization (to tempo grid)
    - (Optional) Chord progression inference

    Note: Onset/offset detection is handled by hum2melody model.
    """

    def __init__(self, enable_chord_analysis: bool = False):
        """
        Initialize music theory processor.

        Args:
            enable_chord_analysis: Whether to enable chord progression inference
        """
        print("\n" + "="*60)
        print("MUSIC THEORY PROCESSOR INITIALIZATION")
        print("="*60)

        self.tempo_detector = TempoDetector()
        self.key_detector = KeyDetector()
        self.pitch_quantizer = PitchQuantizer()
        self.rhythmic_quantizer = RhythmicQuantizer()
        self.enable_chord_analysis = enable_chord_analysis

        if self.enable_chord_analysis:
            self.chord_analyzer = ChordAnalyzer()
            print("[ChordAnalyzer] Enabled")
        else:
            self.chord_analyzer = None
            print("[ChordAnalyzer] Disabled")

        print("="*60)
        print("[OK] Music Theory Processor Ready")
        print("="*60 + "\n")

    def process(
        self,
        notes: List[Dict],  # Notes from hum2melody in hum2melody format or internal format
        input_format: str = "hum2melody"  # "hum2melody" or "internal"
    ) -> Dict:
        """
        Complete music theory processing pipeline.

        Pipeline (in order):
        1. Convert note format if needed
        2. Detect tempo from note onset times
        3. Detect key (from raw, potentially off-key notes)
        4. Quantize pitches to detected key (strict to scale)
        5. Quantize rhythm to tempo grid
        6. (Optional) Infer chord progression

        Args:
            notes: List of notes from hum2melody predictor
                   Can be in hum2melody format or internal format
            input_format: "hum2melody" or "internal"

        Returns:
            {
                "notes": List[Dict],  # Fully quantized notes (hum2melody format)
                "metadata": {
                    "key": str,               # "C major"
                    "tempo": float,           # BPM
                    "time_signature": str,    # "4/4"
                    "grid_resolution": str,   # "1/16"
                    "key_confidence": float,
                    "tempo_confidence": float
                },
                "harmony": List[Dict] | None  # Chord progression (if enabled)
            }
        """
        print("\n" + "="*60)
        print("MUSIC THEORY PROCESSING PIPELINE")
        print("="*60)

        if not notes:
            print("⚠️  No notes provided - returning empty result")
            return self._empty_result()

        # ============================================================
        # STEP 1: Convert note format if needed
        # ============================================================
        if input_format == "hum2melody":
            print(f"\n[STEP 1] Converting {len(notes)} notes from hum2melody format...")
            internal_notes = convert_hum2melody_to_internal(notes)
        else:
            print(f"\n[STEP 1] Using {len(notes)} notes in internal format...")
            internal_notes = notes

        # ============================================================
        # STEP 2: Detect tempo
        # ============================================================
        print("\n[STEP 2] Detecting tempo from onset times...")

        onset_times = [note['start'] for note in internal_notes]
        tempo, tempo_confidence = self.tempo_detector.detect_tempo(onset_times)

        print(f"[OK] Tempo: {tempo:.1f} BPM (confidence: {tempo_confidence:.2f})")

        # ============================================================
        # STEP 3: Detect key (from raw, off-key notes)
        # ============================================================
        print("\n[STEP 3] Detecting musical key from notes...")

        key_root, mode, key_confidence = self.key_detector.detect_key(internal_notes)

        print(f"[OK] Key: {key_root} {mode} (confidence: {key_confidence:.2f})")

        # ============================================================
        # STEP 4: Quantize pitches to detected key
        # ============================================================
        print(f"\n[STEP 4] Quantizing pitches to {key_root} {mode}...")

        pitch_quantized = self.pitch_quantizer.quantize_melody(
            internal_notes,
            key_root,
            mode
        )

        print(f"[OK] Pitches quantized to {key_root} {mode} scale")

        # ============================================================
        # STEP 5: Quantize rhythm to tempo grid
        # ============================================================
        print("\n[STEP 5] Quantizing rhythm to tempo grid...")

        # Determine appropriate grid resolution
        durations = [n['duration'] for n in pitch_quantized]
        grid_resolution = self.tempo_detector.get_grid_resolution(tempo, durations)

        fully_quantized = self.rhythmic_quantizer.quantize_melody(
            pitch_quantized,
            tempo,
            grid_resolution
        )

        print(f"[OK] Rhythm quantized to {grid_resolution} grid at {tempo} BPM")

        # ============================================================
        # STEP 6: (Optional) Infer chord progression
        # ============================================================
        chords = None
        if self.enable_chord_analysis:
            print("\n[STEP 6] Inferring chord progression...")

            chords = self.chord_analyzer.infer_progression(
                fully_quantized,
                key_root,
                mode,
                tempo,
                bars_per_chord=1.0
            )

            print(f"[OK] Inferred {len(chords)} chords")
        else:
            print("\n[STEP 6] Chord analysis disabled - skipping")

        # ============================================================
        # Build result (convert back to hum2melody format)
        # ============================================================
        output_notes = convert_internal_to_hum2melody(fully_quantized)

        result = {
            "notes": output_notes,
            "metadata": {
                "key": f"{key_root} {mode}",
                "tempo": float(tempo),
                "time_signature": "4/4",  # Assume 4/4 for now
                "grid_resolution": grid_resolution,
                "key_confidence": float(key_confidence),
                "tempo_confidence": float(tempo_confidence)
            },
            "harmony": chords
        }

        print("\n" + "="*60)
        print("MUSIC THEORY PROCESSING COMPLETE")
        print("="*60)
        print(f"  Key: {key_root} {mode} (confidence: {key_confidence:.2f})")
        print(f"  Tempo: {tempo:.1f} BPM (confidence: {tempo_confidence:.2f})")
        print(f"  Notes: {len(output_notes)}")
        if chords:
            print(f"  Chords: {len(chords)}")
        print("="*60 + "\n")

        return result

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "notes": [],
            "metadata": {
                "key": "C major",
                "tempo": 120.0,
                "time_signature": "4/4",
                "grid_resolution": "1/16",
                "key_confidence": 0.0,
                "tempo_confidence": 0.0
            },
            "harmony": []
        }


# Convenience exports
__all__ = [
    'MusicTheoryProcessor',
    'TempoDetector',
    'KeyDetector',
    'PitchQuantizer',
    'RhythmicQuantizer',
    'ChordAnalyzer',
    'convert_hum2melody_to_internal',
    'convert_internal_to_hum2melody'
]


if __name__ == '__main__':
    print("\nTesting MusicTheoryProcessor...")

    processor = MusicTheoryProcessor()

    # Create synthetic model output
    # Simulating C major melody at 120 BPM
    time_steps = 200
    frame_probs = np.random.rand(time_steps, 88) * 0.2

    # Make C-E-G-C pattern (C major triad)
    # C4 = MIDI 60 = index 39 (60 - 21)
    # E4 = MIDI 64 = index 43
    # G4 = MIDI 67 = index 46

    frame_probs[0:50, 39] = 0.9    # C
    frame_probs[50:100, 43] = 0.85  # E
    frame_probs[100:150, 46] = 0.88 # G
    frame_probs[150:200, 39] = 0.92 # C

    # Create onset/offset probabilities (with over-prediction)
    onset_probs = np.zeros(time_steps)
    # Real onsets with clusters of false positives
    onset_probs[0:5] = [0.4, 0.8, 0.9, 0.7, 0.3]    # Onset 1
    onset_probs[48:53] = [0.3, 0.7, 0.85, 0.6, 0.2]  # Onset 2
    onset_probs[98:103] = [0.4, 0.75, 0.9, 0.65, 0.25] # Onset 3
    onset_probs[148:153] = [0.35, 0.8, 0.95, 0.7, 0.3] # Onset 4

    offset_probs = np.zeros(time_steps)
    offset_probs[48] = 0.8
    offset_probs[98] = 0.85
    offset_probs[148] = 0.82
    offset_probs[195] = 0.9

    # Process
    result = processor.process(
        frame_probs,
        onset_probs,
        offset_probs,
        frame_rate=10.0  # 10 fps for this test
    )

    # Check results
    print("\nResults:")
    print(f"  Detected key: {result['metadata']['key']}")
    print(f"  Tempo: {result['metadata']['tempo']} BPM")
    print(f"  Notes: {len(result['notes'])}")
    print(f"  Chords: {len(result['harmony'])}")

    print("\nNotes:")
    for i, note in enumerate(result['notes'][:5]):  # First 5 notes
        print(f"  {i+1}. MIDI {note['pitch']}, "
              f"start={note['start']:.2f}s, dur={note['duration']:.2f}s")

    print("\nChord Progression:")
    for chord in result['harmony']:
        print(f"  {chord['roman']}: {chord['root']} {chord['quality']} "
              f"(t={chord['start']:.1f}s)")

    assert result['metadata']['key'] in ["C major", "A minor"], \
        "Should detect C major or A minor (relative)"
    assert len(result['notes']) > 0, "Should have notes"

    print("\n[OK] MusicTheoryProcessor integration test passed!")
