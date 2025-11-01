"""
Pitch Quantizer - Strict Scale Quantization

Snaps MIDI pitches to the nearest scale degree in the detected key.
Uses STRICT quantization - all notes must fit the scale.
"""

import numpy as np
from typing import List, Dict


class PitchQuantizer:
    """
    Quantizes MIDI pitches to detected musical key (strict).

    All pitches are snapped to the nearest scale degree.
    """

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        print("[PitchQuantizer] Initialized (strict to scale)")

    def quantize_to_key(
        self,
        note: Dict,  # {pitch, start, duration, confidence}
        key_root: str,
        mode: str
    ) -> Dict:
        """
        Snap note pitch to nearest scale degree in key.

        Args:
            note: Note dict with pitch (MIDI number, can be float)
            key_root: Root note ("C", "D#", etc.)
            mode: "major" or "minor"

        Returns:
            Note with quantized pitch (int)
        """
        # Get scale notes for this key
        scale_pitch_classes = self._get_scale_pitch_classes(key_root, mode)

        # Round original pitch to nearest semitone first
        midi_rounded = round(note['pitch'])

        # Get pitch class and octave
        pitch_class = midi_rounded % 12
        octave = midi_rounded // 12

        # Find nearest scale degree
        # Check current octave, octave above, and octave below
        candidates = []

        for oct_offset in [-1, 0, 1]:
            test_octave = octave + oct_offset
            for scale_pc in scale_pitch_classes:
                candidate_midi = test_octave * 12 + scale_pc
                distance = abs(candidate_midi - midi_rounded)
                candidates.append((candidate_midi, distance))

        # Choose closest
        quantized_midi, distance = min(candidates, key=lambda x: x[1])

        # Create quantized note
        quantized_note = note.copy()
        quantized_note['pitch'] = int(quantized_midi)

        return quantized_note

    def quantize_melody(
        self,
        notes: List[Dict],
        key_root: str,
        mode: str
    ) -> List[Dict]:
        """
        Quantize all notes in melody to key.

        Args:
            notes: List of note dicts
            key_root: Root note
            mode: "major" or "minor"

        Returns:
            List of quantized notes
        """
        if not notes:
            return []

        quantized = [self.quantize_to_key(note, key_root, mode) for note in notes]

        # Count how many notes changed
        changes = sum(1 for orig, quant in zip(notes, quantized)
                     if round(orig['pitch']) != quant['pitch'])

        print(f"[PitchQuantizer] Quantized {len(notes)} notes to {key_root} {mode}")
        print(f"  Changed: {changes}/{len(notes)} notes ({100*changes/len(notes):.1f}%)")

        return quantized

    def _get_scale_pitch_classes(self, root: str, mode: str) -> List[int]:
        """
        Get pitch classes (0-11) for scale in key.

        Args:
            root: Root note name
            mode: "major" or "minor"

        Returns:
            List of pitch classes in scale
        """
        root_idx = self.NOTE_NAMES.index(root)

        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        else:  # minor
            intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor

        scale_pcs = [(root_idx + interval) % 12 for interval in intervals]

        return scale_pcs


if __name__ == '__main__':
    print("\nTesting PitchQuantizer...")

    quantizer = PitchQuantizer()

    # Test 1: Quantize to C major
    print("\nTest 1: Quantize off-key notes to C major")
    notes = [
        {'pitch': 60.3, 'start': 0.0, 'duration': 0.5, 'confidence': 0.8},  # C (slightly sharp)
        {'pitch': 61.0, 'start': 0.5, 'duration': 0.5, 'confidence': 0.7},  # C# (should snap to C or D)
        {'pitch': 64.2, 'start': 1.0, 'duration': 0.5, 'confidence': 0.9},  # E (slightly sharp)
        {'pitch': 66.8, 'start': 1.5, 'duration': 0.5, 'confidence': 0.8},  # F# (should snap to G)
    ]

    quantized = quantizer.quantize_melody(notes, "C", "major")

    print(f"  Original → Quantized:")
    for orig, quant in zip(notes, quantized):
        print(f"    {orig['pitch']:.1f} → {quant['pitch']}")

    # Check results
    assert quantized[0]['pitch'] == 60, "60.3 should quantize to 60 (C)"
    assert quantized[1]['pitch'] in [60, 62], "61 should quantize to 60 (C) or 62 (D)"
    assert quantized[2]['pitch'] == 64, "64.2 should quantize to 64 (E)"
    assert quantized[3]['pitch'] == 67, "66.8 should quantize to 67 (G)"
    print("  ✅ C major quantization correct")

    # Test 2: Quantize to A minor
    print("\nTest 2: Quantize to A minor")
    notes_minor = [
        {'pitch': 69.0, 'start': 0.0, 'duration': 0.5, 'confidence': 0.9},  # A
        {'pitch': 71.0, 'start': 0.5, 'duration': 0.5, 'confidence': 0.9},  # B
        {'pitch': 73.0, 'start': 1.0, 'duration': 0.5, 'confidence': 0.8},  # C# (should snap to C)
    ]

    quantized_minor = quantizer.quantize_melody(notes_minor, "A", "minor")

    print(f"  Original → Quantized:")
    for orig, quant in zip(notes_minor, quantized_minor):
        print(f"    {orig['pitch']} → {quant['pitch']}")

    assert quantized_minor[0]['pitch'] == 69, "69 should stay 69 (A)"
    assert quantized_minor[1]['pitch'] == 71, "71 should stay 71 (B)"
    assert quantized_minor[2]['pitch'] == 72, "73 (C#) should quantize to 72 (C) in A minor"
    print("  ✅ A minor quantization correct")

    # Test 3: Preserve timing and confidence
    print("\nTest 3: Preserve non-pitch attributes")
    note = {
        'pitch': 61.5,
        'start': 1.23,
        'duration': 0.456,
        'confidence': 0.789
    }

    quantized_note = quantizer.quantize_to_key(note, "C", "major")

    assert quantized_note['start'] == 1.23, "Start time should be preserved"
    assert quantized_note['duration'] == 0.456, "Duration should be preserved"
    assert quantized_note['confidence'] == 0.789, "Confidence should be preserved"
    print(f"  Pitch: {note['pitch']} → {quantized_note['pitch']}")
    print(f"  Start: {quantized_note['start']}")
    print(f"  Duration: {quantized_note['duration']}")
    print(f"  Confidence: {quantized_note['confidence']}")
    print("  ✅ Non-pitch attributes preserved")

    print("\n✅ All PitchQuantizer tests passed!")
