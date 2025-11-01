"""
Music Theory Post-Processing Module

Transforms raw model predictions into musically coherent melodies.

Pipeline (in order):
1. Clean onsets/offsets (reduce over-prediction)
2. Extract raw notes
3. Detect tempo
4. DETECT KEY FIRST (from raw, off-key notes)
5. Quantize pitches to detected key
6. Quantize rhythm to tempo grid
7. Infer chord progression

This module is the complete music theory layer that sits between
raw ML predictions and the final musical IR.
"""

import numpy as np
from typing import List, Dict, Tuple

from .onset_cleaner import OnsetOffsetCleaner
from .tempo_detector import TempoDetector
from .key_detector import KeyDetector
from .pitch_quantizer import PitchQuantizer
from .rhythmic_quantizer import RhythmicQuantizer
from .chord_analyzer import ChordAnalyzer


class MusicTheoryProcessor:
    """
    Main facade for complete music theory post-processing pipeline.

    Transforms raw model output into musically coherent melody with:
    - Cleaned onsets/offsets (reduced over-prediction)
    - Detected musical key
    - Pitch quantization (strict to scale)
    - Rhythmic quantization (to tempo grid)
    - Inferred chord progression
    """

    def __init__(self):
        print("\n" + "="*60)
        print("MUSIC THEORY PROCESSOR INITIALIZATION")
        print("="*60)

        self.onset_cleaner = OnsetOffsetCleaner()
        self.tempo_detector = TempoDetector()
        self.key_detector = KeyDetector()
        self.pitch_quantizer = PitchQuantizer()
        self.rhythmic_quantizer = RhythmicQuantizer()
        self.chord_analyzer = ChordAnalyzer()

        print("="*60)
        print("✅ Music Theory Processor Ready")
        print("="*60 + "\n")

    def process(
        self,
        frame_probs: np.ndarray,    # (time_steps, 88)
        onset_probs: np.ndarray,    # (time_steps,)
        offset_probs: np.ndarray,   # (time_steps,)
        frame_rate: float = 7.8125  # fps
    ) -> Dict:
        """
        Complete music theory processing pipeline.

        Pipeline (in order):
        1. Clean onsets/offsets → reduce ~60-70%
        2. Extract raw notes from cleaned events
        3. Detect tempo from onset intervals
        4. ⭐ DETECT KEY FIRST (from raw, potentially off-key notes)
        5. Quantize pitches to detected key (strict)
        6. Quantize rhythm to tempo grid
        7. Infer chord progression from quantized melody

        Args:
            frame_probs: Frame-level pitch probabilities (time_steps, 88)
            onset_probs: Onset probabilities (time_steps,)
            offset_probs: Offset probabilities (time_steps,)
            frame_rate: Model output frame rate (fps)

        Returns:
            {
                "notes": List[Dict],  # Fully quantized notes
                "metadata": {
                    "key": str,               # "C major"
                    "tempo": float,           # BPM
                    "time_signature": str,    # "4/4"
                    "grid_resolution": str    # "1/16"
                },
                "harmony": List[Dict]  # Chord progression
            }
        """
        print("\n" + "="*60)
        print("MUSIC THEORY PROCESSING PIPELINE")
        print("="*60)

        # ============================================================
        # STEP 1 & 2: Clean onsets/offsets and extract notes
        # ============================================================
        print("\n[STEP 1-2] Cleaning onsets/offsets and extracting notes...")

        cleaned_onsets = self.onset_cleaner.cluster_onsets(onset_probs, frame_rate)
        onset_times = [t for t, _ in cleaned_onsets]
        onset_confidences = [c for _, c in cleaned_onsets]

        cleaned_offsets = self.onset_cleaner.cluster_offsets(
            offset_probs,
            onset_times,
            frame_rate
        )
        offset_times = [t for t, _ in cleaned_offsets]
        offset_confidences = [c for _, c in cleaned_offsets]

        raw_notes = self.onset_cleaner.extract_notes_from_cleaned_events(
            onset_times,
            onset_confidences,
            offset_times,
            offset_confidences,
            frame_probs,
            frame_rate
        )

        if not raw_notes:
            print("⚠️  No notes extracted - returning empty result")
            return self._empty_result()

        print(f"✅ Extracted {len(raw_notes)} raw notes")

        # ============================================================
        # STEP 3: Detect tempo
        # ============================================================
        print("\n[STEP 3] Detecting tempo...")

        tempo, tempo_confidence = self.tempo_detector.detect_tempo(onset_times)

        print(f"✅ Tempo: {tempo:.1f} BPM (confidence: {tempo_confidence:.2f})")

        # ============================================================
        # STEP 4: ⭐ DETECT KEY FIRST (from raw, off-key notes)
        # ============================================================
        print("\n[STEP 4] ⭐ Detecting musical key (from raw notes)...")

        key_root, mode, key_confidence = self.key_detector.detect_key(raw_notes)

        print(f"✅ Key: {key_root} {mode} (confidence: {key_confidence:.2f})")

        # ============================================================
        # STEP 5: Quantize pitches to detected key
        # ============================================================
        print("\n[STEP 5] Quantizing pitches to {key_root} {mode}...")

        pitch_quantized = self.pitch_quantizer.quantize_melody(
            raw_notes,
            key_root,
            mode
        )

        print(f"✅ Pitches quantized to {key_root} {mode} scale")

        # ============================================================
        # STEP 6: Quantize rhythm to tempo grid
        # ============================================================
        print("\n[STEP 6] Quantizing rhythm to tempo grid...")

        # Determine appropriate grid resolution
        durations = [n['duration'] for n in pitch_quantized]
        grid_resolution = self.tempo_detector.get_grid_resolution(tempo, durations)

        fully_quantized = self.rhythmic_quantizer.quantize_melody(
            pitch_quantized,
            tempo,
            grid_resolution
        )

        print(f"✅ Rhythm quantized to {grid_resolution} grid at {tempo} BPM")

        # ============================================================
        # STEP 7: Infer chord progression
        # ============================================================
        print("\n[STEP 7] Inferring chord progression...")

        chords = self.chord_analyzer.infer_progression(
            fully_quantized,
            key_root,
            mode,
            tempo,
            bars_per_chord=1.0
        )

        print(f"✅ Inferred {len(chords)} chord progression")

        # ============================================================
        # Build result
        # ============================================================
        result = {
            "notes": fully_quantized,
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
        print(f"  Key: {key_root} {mode}")
        print(f"  Tempo: {tempo:.1f} BPM")
        print(f"  Notes: {len(fully_quantized)}")
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
    'OnsetOffsetCleaner',
    'TempoDetector',
    'KeyDetector',
    'PitchQuantizer',
    'RhythmicQuantizer',
    'ChordAnalyzer'
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

    print("\n✅ MusicTheoryProcessor integration test passed!")
