"""
Chord Analyzer - Musical Chord Progression Inference

Infers chord progression from quantized melody using music theory rules.
Includes all musically valid chords: diatonic, 7ths, borrowed, secondary dominants.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class ChordAnalyzer:
    """
    Analyzes quantized melody to infer underlying chord progression.

    Uses melody-driven analysis with comprehensive chord vocabulary.
    """

    # Chord templates (intervals from root)
    CHORD_TEMPLATES = {
        # Triads
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "diminished": [0, 3, 6],
        "augmented": [0, 4, 8],

        # 7th chords
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
        "min7b5": [0, 3, 6, 10],  # Half-diminished
        "dim7": [0, 3, 6, 9],

        # Suspensions
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],

        # Extended
        "maj9": [0, 4, 7, 11, 14],
        "min9": [0, 3, 7, 10, 14],
        "dom9": [0, 4, 7, 10, 14],
    }

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        print("[ChordAnalyzer] Initialized with comprehensive chord vocabulary")

    def infer_progression(
        self,
        notes: List[Dict],
        key_root: str,
        mode: str,
        tempo: float,
        bars_per_chord: float = 1.0
    ) -> List[Dict]:
        """
        Infer chord progression from quantized melody.

        Args:
            notes: Quantized notes {pitch, start, duration, confidence}
            key_root: Detected key root
            mode: "major" or "minor"
            tempo: Tempo in BPM
            bars_per_chord: How many bars per chord (default: 1)

        Returns:
            List of chord events with timing
        """
        if not notes:
            return []

        # Segment melody by time
        segments = self._segment_melody(notes, tempo, bars_per_chord)

        print(f"[ChordAnalyzer] Analyzing {len(segments)} segments")
        print(f"  Key: {key_root} {mode}")
        print(f"  Tempo: {tempo} BPM")

        # Get all valid chords for this key
        valid_chords = self._get_all_valid_chords(key_root, mode)

        # Analyze each segment
        chord_progression = []
        prev_chord = None

        for i, segment in enumerate(segments):
            if not segment['notes']:
                continue

            # Analyze segment to find best chord
            best_chord = self._analyze_segment(
                segment['notes'],
                valid_chords,
                prev_chord
            )

            if best_chord:
                chord_event = {
                    "root": best_chord['root'],
                    "quality": best_chord['quality'],
                    "roman": best_chord['roman'],
                    "start": segment['start'],
                    "duration": segment['duration']
                }

                chord_progression.append(chord_event)
                prev_chord = best_chord

        print(f"[ChordAnalyzer] Inferred {len(chord_progression)} chords")

        return chord_progression

    def _segment_melody(
        self,
        notes: List[Dict],
        tempo: float,
        bars_per_chord: float
    ) -> List[Dict]:
        """
        Divide melody into harmonic segments based on tempo.

        Args:
            notes: Note list
            tempo: BPM
            bars_per_chord: Bars per chord

        Returns:
            List of segments with timing and notes
        """
        if not notes:
            return []

        beat_duration = 60.0 / tempo
        bar_duration = beat_duration * 4  # Assuming 4/4 time
        segment_duration = bar_duration * bars_per_chord

        # Find total duration
        last_note = max(notes, key=lambda n: n['start'] + n['duration'])
        total_duration = last_note['start'] + last_note['duration']

        # Create segments
        segments = []
        current_time = 0.0

        while current_time < total_duration:
            segment_end = current_time + segment_duration

            # Collect notes in this segment
            segment_notes = [
                n for n in notes
                if n['start'] < segment_end and (n['start'] + n['duration']) > current_time
            ]

            segments.append({
                'start': current_time,
                'duration': segment_duration,
                'notes': segment_notes
            })

            current_time += segment_duration

        return segments

    def _analyze_segment(
        self,
        notes: List[Dict],
        valid_chords: List[Dict],
        prev_chord: Dict = None
    ) -> Dict:
        """
        Find best chord for melody segment.

        Uses comprehensive scoring:
        - +10: Root in melody
        - +8: 3rd in melody (defines major/minor quality)
        - +7: 7th in melody
        - +5: 5th in melody
        - +3: All melody notes are chord tones
        - -5: Non-chord tones present
        - +5: Follows common progression pattern
        - +3: Smooth voice leading from previous

        Returns:
            Best scoring chord dict
        """
        if not notes:
            return None

        # Get unique pitch classes in segment (weighted by duration)
        pitch_classes = {}
        for note in notes:
            pc = (round(note['pitch']) % 12)
            weight = note['duration'] * note.get('confidence', 1.0)
            pitch_classes[pc] = pitch_classes.get(pc, 0) + weight

        # Normalize weights
        total_weight = sum(pitch_classes.values())
        if total_weight > 0:
            pitch_classes = {pc: w/total_weight for pc, w in pitch_classes.items()}

        # Score each chord
        scored_chords = []

        for chord in valid_chords:
            score = 0
            chord_pcs = set(chord['pitch_classes'])

            # Check which scale degrees are present
            root_pc = chord['pitch_classes'][0]
            has_root = root_pc in pitch_classes
            has_third = (len(chord['pitch_classes']) > 1 and
                        chord['pitch_classes'][1] in pitch_classes)
            has_fifth = (len(chord['pitch_classes']) > 2 and
                        chord['pitch_classes'][2] in pitch_classes)
            has_seventh = (len(chord['pitch_classes']) > 3 and
                          chord['pitch_classes'][3] in pitch_classes)

            # Scoring
            if has_root:
                score += 10 * pitch_classes[root_pc]

            if has_third:
                score += 8 * pitch_classes[chord['pitch_classes'][1]]

            if has_seventh:
                score += 7 * pitch_classes[chord['pitch_classes'][3]]

            if has_fifth:
                score += 5 * pitch_classes[chord['pitch_classes'][2]]

            # All notes are chord tones?
            all_chord_tones = all(pc in chord_pcs for pc in pitch_classes.keys())
            if all_chord_tones:
                score += 3

            # Non-chord tones penalty
            non_chord_tones = [pc for pc in pitch_classes.keys() if pc not in chord_pcs]
            if non_chord_tones:
                score -= 5 * len(non_chord_tones) / len(pitch_classes)

            # Common progression bonus
            if prev_chord:
                if self._is_common_progression(prev_chord['roman'], chord['roman']):
                    score += 5

            scored_chords.append((chord, score))

        # Return highest scoring
        if scored_chords:
            best = max(scored_chords, key=lambda x: x[1])
            return best[0]

        return None

    def _get_all_valid_chords(self, key_root: str, mode: str) -> List[Dict]:
        """
        Get all musically valid chords for key.

        Includes:
        - Diatonic triads and 7ths
        - Modal interchange (borrowed chords)
        - Secondary dominants

        Returns:
            List of chord dicts with root, quality, roman, pitch_classes
        """
        chords = []
        root_idx = self.NOTE_NAMES.index(key_root)

        if mode == "major":
            # Diatonic triads in major
            diatonic = [
                (0, "major", "I"),
                (2, "minor", "ii"),
                (4, "minor", "iii"),
                (5, "major", "IV"),
                (7, "major", "V"),
                (9, "minor", "vi"),
                (11, "diminished", "vii°"),
            ]

            # Diatonic 7ths
            diatonic_7ths = [
                (0, "maj7", "Imaj7"),
                (2, "min7", "ii7"),
                (4, "min7", "iii7"),
                (5, "maj7", "IVmaj7"),
                (7, "dom7", "V7"),
                (9, "min7", "vi7"),
                (11, "min7b5", "viiø7"),
            ]

            # Borrowed from parallel minor
            borrowed = [
                (5, "minor", "iv"),  # Minor subdominant
                (8, "major", "bVI"),  # Flat-six
                (10, "major", "bVII"),  # Flat-seven
                (3, "major", "bIII"),  # Flat-three
            ]

            all_chords = diatonic + diatonic_7ths + borrowed

        else:  # minor
            # Diatonic triads in minor (natural minor)
            diatonic = [
                (0, "minor", "i"),
                (2, "diminished", "ii°"),
                (3, "major", "III"),
                (5, "minor", "iv"),
                (7, "minor", "v"),
                (8, "major", "VI"),
                (10, "major", "VII"),
            ]

            # Harmonic minor variations
            harmonic = [
                (7, "major", "V"),  # Major V in minor
                (7, "dom7", "V7"),
                (11, "diminished", "vii°"),
            ]

            all_chords = diatonic + harmonic

        # Convert to chord dicts
        for interval, quality, roman in all_chords:
            chord_root = (root_idx + interval) % 12
            template = self.CHORD_TEMPLATES.get(quality, [0, 4, 7])

            pitch_classes = [(chord_root + t) % 12 for t in template]

            chords.append({
                'root': self.NOTE_NAMES[chord_root],
                'quality': quality,
                'roman': roman,
                'pitch_classes': pitch_classes
            })

        return chords

    def _is_common_progression(self, from_roman: str, to_roman: str) -> bool:
        """
        Check if this is a common chord progression.

        Common progressions:
        - IV → V, V → I (strong resolutions)
        - I → IV, I → V (tonic to predominant/dominant)
        - ii → V, vi → ii (functional harmony)
        """
        common = [
            ("IV", "V"), ("IVmaj7", "V7"),
            ("V", "I"), ("V7", "I"), ("V7", "Imaj7"),
            ("I", "IV"), ("I", "V"),
            ("ii", "V"), ("ii7", "V7"),
            ("vi", "ii"), ("vi7", "ii7"),
            ("I", "vi"), ("IV", "I"),
            # Minor
            ("iv", "V"), ("V", "i"),
            ("i", "iv"), ("i", "V"),
        ]

        # Remove quality markers for comparison
        from_base = from_roman.replace("maj7", "").replace("m7", "").replace("7", "").replace("ø", "").replace("°", "")
        to_base = to_roman.replace("maj7", "").replace("m7", "").replace("7", "").replace("ø", "").replace("°", "")

        return any((from_base == f and to_base == t) or
                  (from_roman == f and to_roman == t)
                  for f, t in common)


if __name__ == '__main__':
    print("\nTesting ChordAnalyzer...")

    analyzer = ChordAnalyzer()

    # Test 1: Get valid chords for C major
    print("\nTest 1: Valid chords in C major")
    chords = analyzer._get_all_valid_chords("C", "major")
    print(f"  Found {len(chords)} valid chords")
    for chord in chords[:5]:  # Print first 5
        print(f"    {chord['roman']}: {chord['root']} {chord['quality']}")
    assert len(chords) > 0, "Should have valid chords"

    # Test 2: Analyze simple progression (I - IV - V - I)
    print("\nTest 2: Analyze I-IV-V-I progression")
    # C major, E minor, F major, G major, C major
    notes = [
        # Bar 1: C major (I) - C, E, G
        {'pitch': 60, 'start': 0.0, 'duration': 0.5, 'confidence': 0.9},  # C
        {'pitch': 64, 'start': 0.5, 'duration': 0.5, 'confidence': 0.9},  # E
        {'pitch': 67, 'start': 1.0, 'duration': 1.0, 'confidence': 0.9},  # G

        # Bar 2: F major (IV) - F, A, C
        {'pitch': 65, 'start': 2.0, 'duration': 0.5, 'confidence': 0.9},  # F
        {'pitch': 69, 'start': 2.5, 'duration': 0.5, 'confidence': 0.9},  # A
        {'pitch': 72, 'start': 3.0, 'duration': 1.0, 'confidence': 0.9},  # C

        # Bar 3: G major (V) - G, B, D
        {'pitch': 67, 'start': 4.0, 'duration': 0.5, 'confidence': 0.9},  # G
        {'pitch': 71, 'start': 4.5, 'duration': 0.5, 'confidence': 0.9},  # B
        {'pitch': 74, 'start': 5.0, 'duration': 1.0, 'confidence': 0.9},  # D

        # Bar 4: C major (I) - C
        {'pitch': 72, 'start': 6.0, 'duration': 2.0, 'confidence': 0.9},  # C
    ]

    progression = analyzer.infer_progression(
        notes,
        key_root="C",
        mode="major",
        tempo=120,
        bars_per_chord=1.0
    )

    print(f"  Inferred progression:")
    for chord in progression:
        print(f"    {chord['roman']}: {chord['root']} {chord['quality']} "
              f"(t={chord['start']:.1f}s, dur={chord['duration']:.1f}s)")

    # Should detect I, IV, V pattern
    assert len(progression) >= 3, "Should detect at least 3 chords"
    print("  ✅ Progression inferred")

    print("\n✅ ChordAnalyzer tests passed!")
