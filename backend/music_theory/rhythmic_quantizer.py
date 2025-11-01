"""
Rhythmic Quantizer - Snap to Musical Grid

Quantizes note start times and durations to musical grid based on tempo.
Supports standard divisions: 1/32, 1/16, 1/8, 1/4, 1/2, whole notes.
Also supports dotted notes (1.5x).
"""

import numpy as np
from typing import List, Dict


class RhythmicQuantizer:
    """
    Quantizes note timings to tempo-based musical grid.

    Snaps both start times and durations to valid musical divisions.
    """

    def __init__(self):
        # Standard note divisions (as fractions of a beat)
        self.grid_divisions = {
            "1/1": 1,      # Whole note
            "1/2": 2,      # Half note
            "1/4": 4,      # Quarter note
            "1/8": 8,      # Eighth note
            "1/16": 16,    # Sixteenth note
            "1/32": 32     # Thirty-second note
        }

        # Valid duration multipliers (in beats)
        # Includes standard and dotted notes
        self.valid_durations = [
            1/32,  # 32nd note
            3/64,  # Dotted 32nd
            1/16,  # 16th note
            3/32,  # Dotted 16th
            1/8,   # 8th note
            3/16,  # Dotted 8th
            1/4,   # Quarter note
            3/8,   # Dotted quarter
            1/2,   # Half note
            3/4,   # Dotted half
            1.0,   # Whole note
            1.5,   # Dotted whole
            2.0,   # Double whole
        ]

        print("[RhythmicQuantizer] Initialized")
        print(f"  Supported grids: {list(self.grid_divisions.keys())}")

    def quantize_melody(
        self,
        notes: List[Dict],
        tempo: float,
        grid_resolution: str = "1/16",
        allow_dotted: bool = True
    ) -> List[Dict]:
        """
        Quantize all notes in melody to musical grid.

        Args:
            notes: List of note dicts {pitch, start, duration, confidence}
            tempo: Tempo in BPM
            grid_resolution: Grid fineness ("1/16", etc.)
            allow_dotted: Whether to allow dotted note durations

        Returns:
            List of rhythmically quantized notes
        """
        if not notes:
            return []

        # Calculate time values
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        grid_step = beat_duration / self.grid_divisions[grid_resolution]

        print(f"[RhythmicQuantizer] Quantizing {len(notes)} notes")
        print(f"  Tempo: {tempo} BPM")
        print(f"  Beat duration: {beat_duration:.3f}s")
        print(f"  Grid: {grid_resolution} ({grid_step:.3f}s per step)")

        quantized = []

        for note in notes:
            # Quantize start time to grid
            quantized_start = self._snap_to_grid(note['start'], grid_step)

            # Quantize duration to valid musical duration
            duration_in_beats = note['duration'] / beat_duration
            quantized_duration_beats = self._find_nearest_duration(
                duration_in_beats,
                allow_dotted=allow_dotted
            )
            quantized_duration = quantized_duration_beats * beat_duration

            # Create quantized note
            quant_note = note.copy()
            quant_note['start'] = quantized_start
            quant_note['duration'] = quantized_duration

            quantized.append(quant_note)

        # Resolve overlaps
        quantized = self._resolve_overlaps(quantized)

        # Stats
        start_changes = sum(1 for orig, quant in zip(notes, quantized)
                           if abs(orig['start'] - quant['start']) > 0.001)
        duration_changes = sum(1 for orig, quant in zip(notes, quantized)
                              if abs(orig['duration'] - quant['duration']) > 0.001)

        print(f"  Start times changed: {start_changes}/{len(notes)}")
        print(f"  Durations changed: {duration_changes}/{len(notes)}")

        return quantized

    def _snap_to_grid(self, time: float, grid_step: float) -> float:
        """Round time to nearest grid point."""
        return round(time / grid_step) * grid_step

    def _find_nearest_duration(
        self,
        duration_beats: float,
        allow_dotted: bool = True,
        min_duration: float = 1/32
    ) -> float:
        """
        Find nearest valid musical duration.

        Args:
            duration_beats: Duration in beats
            allow_dotted: Whether to include dotted notes
            min_duration: Minimum duration in beats

        Returns:
            Nearest valid duration in beats
        """
        # Filter valid durations
        if allow_dotted:
            candidates = self.valid_durations
        else:
            # Only standard (non-dotted) divisions
            candidates = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0]

        # Ensure minimum duration
        candidates = [d for d in candidates if d >= min_duration]

        if not candidates:
            return min_duration

        # Find closest
        nearest = min(candidates, key=lambda x: abs(x - duration_beats))

        return nearest

    def _resolve_overlaps(self, notes: List[Dict]) -> List[Dict]:
        """
        Fix overlapping notes by adjusting durations.

        If note[i] overlaps with note[i+1], shorten note[i]'s duration
        to end exactly when note[i+1] starts.
        """
        if len(notes) <= 1:
            return notes

        # Sort by start time
        sorted_notes = sorted(notes, key=lambda n: n['start'])

        resolved = []

        for i in range(len(sorted_notes)):
            note = sorted_notes[i].copy()
            note_end = note['start'] + note['duration']

            # Check if overlaps with next note
            if i < len(sorted_notes) - 1:
                next_start = sorted_notes[i + 1]['start']

                if note_end > next_start:
                    # Overlap detected - shorten this note
                    new_duration = next_start - note['start']

                    if new_duration > 0.01:  # Minimum 10ms
                        note['duration'] = new_duration
                    else:
                        # Notes too close - skip this one
                        continue

            resolved.append(note)

        if len(resolved) < len(notes):
            print(f"  Removed {len(notes) - len(resolved)} overlapping notes")

        return resolved


if __name__ == '__main__':
    print("\nTesting RhythmicQuantizer...")

    quantizer = RhythmicQuantizer()

    # Test 1: Quantize to 1/16 grid at 120 BPM
    # At 120 BPM: beat = 0.5s, 1/16 = 0.03125s
    print("\nTest 1: Quantize to 1/16 grid at 120 BPM")

    notes = [
        {'pitch': 60, 'start': 0.07, 'duration': 0.23, 'confidence': 0.9},   # ~1/16, ~1/8
        {'pitch': 62, 'start': 0.52, 'duration': 0.48, 'confidence': 0.9},   # ~1/2, ~1/4
        {'pitch': 64, 'start': 1.03, 'duration': 0.96, 'confidence': 0.9},   # ~1, ~1/2
    ]

    quantized = quantizer.quantize_melody(notes, tempo=120, grid_resolution="1/16")

    print(f"  Original → Quantized (start, duration):")
    for orig, quant in zip(notes, quantized):
        print(f"    ({orig['start']:.3f}s, {orig['duration']:.3f}s) → "
              f"({quant['start']:.3f}s, {quant['duration']:.3f}s)")

    # At 120 BPM, 1/16 grid step = 0.03125s
    # Starts should be multiples of 0.03125
    grid_step = 0.03125
    for quant in quantized:
        remainder = (quant['start'] % grid_step)
        assert remainder < 0.0001 or remainder > grid_step - 0.0001, \
            f"Start {quant['start']} not on grid"

    print("  ✅ All starts on grid")

    # Test 2: Different tempo (80 BPM)
    print("\nTest 2: Quantize at 80 BPM")
    # At 80 BPM: beat = 0.75s, 1/8 = 0.09375s

    notes_80 = [
        {'pitch': 60, 'start': 0.1, 'duration': 0.7, 'confidence': 0.9},
        {'pitch': 62, 'start': 0.85, 'duration': 0.72, 'confidence': 0.9},
    ]

    quantized_80 = quantizer.quantize_melody(notes_80, tempo=80, grid_resolution="1/8")

    print(f"  Original → Quantized:")
    for orig, quant in zip(notes_80, quantized_80):
        print(f"    Start: {orig['start']:.3f}s → {quant['start']:.3f}s")
        print(f"    Duration: {orig['duration']:.3f}s → {quant['duration']:.3f}s")

    # Test 3: Overlap resolution
    print("\nTest 3: Resolve overlapping notes")

    overlapping = [
        {'pitch': 60, 'start': 0.0, 'duration': 0.6, 'confidence': 0.9},  # Overlaps next
        {'pitch': 62, 'start': 0.5, 'duration': 0.5, 'confidence': 0.9},
    ]

    resolved = quantizer._resolve_overlaps(overlapping)

    print(f"  Before:")
    for note in overlapping:
        print(f"    Start: {note['start']}, End: {note['start'] + note['duration']}")

    print(f"  After:")
    for note in resolved:
        print(f"    Start: {note['start']}, End: {note['start'] + note['duration']}")

    # First note should be shortened to end at 0.5
    assert abs((resolved[0]['start'] + resolved[0]['duration']) - 0.5) < 0.01, \
        "First note should end at 0.5"
    print("  ✅ Overlap resolved")

    # Test 4: Valid duration finding
    print("\nTest 4: Find nearest valid durations")

    test_durations = [0.1, 0.25, 0.4, 0.75, 1.2]  # In beats

    for dur in test_durations:
        nearest = quantizer._find_nearest_duration(dur, allow_dotted=True)
        print(f"  {dur} beats → {nearest} beats")

    print("\n✅ All RhythmicQuantizer tests passed!")
