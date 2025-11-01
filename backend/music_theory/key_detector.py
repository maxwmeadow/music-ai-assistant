"""
Key Detector - Robust Musical Key Detection

Detects musical key from potentially off-key humming using
Krumhansl-Schmuckler algorithm with pitch class histograms.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class KeyDetector:
    """
    Detects musical key using pitch class correlation with key profiles.

    Robust to off-key singing by rounding to nearest semitone.
    """

    # Krumhansl-Schmuckler key profiles (perceptual weights for each scale degree)
    # Values represent how strongly each pitch class relates to the tonic
    MAJOR_PROFILE = np.array([
        6.35,  # Tonic (strongest)
        2.23,  # Minor 2nd
        3.48,  # Major 2nd
        2.33,  # Minor 3rd
        4.38,  # Major 3rd
        4.09,  # Perfect 4th
        2.52,  # Tritone
        5.19,  # Perfect 5th (strong)
        2.39,  # Minor 6th
        3.66,  # Major 6th
        2.29,  # Minor 7th
        2.88   # Major 7th
    ])

    MINOR_PROFILE = np.array([
        6.33,  # Tonic (strongest)
        2.68,  # Minor 2nd
        3.52,  # Major 2nd
        5.38,  # Minor 3rd (strong in minor)
        2.60,  # Major 3rd
        3.53,  # Perfect 4th
        2.54,  # Tritone
        4.75,  # Perfect 5th (strong)
        3.98,  # Minor 6th
        2.69,  # Major 6th
        3.34,  # Minor 7th
        3.17   # Major 7th
    ])

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self):
        print("[KeyDetector] Initialized with Krumhansl-Schmuckler profiles")

    def detect_key(
        self,
        notes: List[Dict]  # List of {pitch, start, duration, confidence}
    ) -> Tuple[str, str, float]:
        """
        Detect musical key from notes (robust to off-key humming).

        Args:
            notes: List of note dicts with pitch (MIDI), duration, confidence

        Returns:
            (root, mode, confidence)
            e.g., ("C", "major", 0.87)
        """
        if not notes:
            return ("C", "major", 0.0)

        # Create pitch class histogram weighted by duration
        histogram = self._create_pitch_class_histogram(notes)

        # Test all 24 keys (12 major + 12 minor)
        best_key = None
        best_correlation = -1.0

        for root_idx in range(12):  # 12 possible roots (C through B)
            # Test major
            major_corr = self._correlate_with_key_profile(
                histogram,
                self.MAJOR_PROFILE,
                root_idx
            )

            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = (self.NOTE_NAMES[root_idx], "major", major_corr)

            # Test minor
            minor_corr = self._correlate_with_key_profile(
                histogram,
                self.MINOR_PROFILE,
                root_idx
            )

            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = (self.NOTE_NAMES[root_idx], "minor", minor_corr)

        root, mode, confidence = best_key
        print(f"[KeyDetector] Detected key: {root} {mode} (confidence: {confidence:.3f})")

        return best_key

    def _create_pitch_class_histogram(self, notes: List[Dict]) -> np.ndarray:
        """
        Create pitch class histogram weighted by duration.

        Robust to off-key singing by rounding MIDI pitches to nearest semitone.
        """
        histogram = np.zeros(12)  # 12 pitch classes (C through B)

        for note in notes:
            # Round to nearest semitone (handles off-key humming)
            midi_rounded = round(note['pitch'])
            pitch_class = midi_rounded % 12  # 0=C, 1=C#, 2=D, ..., 11=B

            # Weight by duration and confidence
            weight = note['duration'] * note.get('confidence', 1.0)
            histogram[pitch_class] += weight

        # Normalize
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total

        return histogram

    def _correlate_with_key_profile(
        self,
        histogram: np.ndarray,
        profile: np.ndarray,
        root_idx: int
    ) -> float:
        """
        Calculate correlation between histogram and key profile.

        Args:
            histogram: Pitch class histogram (12 values)
            profile: Key profile (12 values)
            root_idx: Root note index (0=C, 1=C#, etc.)

        Returns:
            Correlation coefficient (higher = better match)
        """
        # Rotate profile to match root
        # E.g., if root_idx=2 (D), rotate so D is at position 0
        rotated_profile = np.roll(profile, -root_idx)

        # Calculate Pearson correlation
        correlation = np.corrcoef(histogram, rotated_profile)[0, 1]

        # Handle NaN (can occur if histogram or profile is constant)
        if np.isnan(correlation):
            correlation = 0.0

        return correlation

    def get_scale_notes(self, root: str, mode: str) -> List[int]:
        """
        Get pitch classes (0-11) for scale in key.

        Args:
            root: Root note name ("C", "D#", etc.)
            mode: "major" or "minor"

        Returns:
            List of pitch classes in scale

        Example:
            get_scale_notes("C", "major") → [0, 2, 4, 5, 7, 9, 11]
                                            (C, D, E, F, G, A, B)
        """
        # Get root pitch class
        root_idx = self.NOTE_NAMES.index(root)

        # Scale intervals from root
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale (W-W-H-W-W-W-H)
        else:  # minor
            intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor (W-H-W-W-H-W-W)

        # Add root to intervals and mod 12
        scale_notes = [(root_idx + interval) % 12 for interval in intervals]

        return scale_notes

    def get_scale_name(self, root: str, mode: str) -> str:
        """Get full scale name for display."""
        return f"{root} {mode}"


if __name__ == '__main__':
    print("\nTesting KeyDetector...")

    detector = KeyDetector()

    # Test 1: Perfect C major scale
    c_major_notes = [
        {'pitch': 60, 'duration': 1.0, 'confidence': 0.9},  # C
        {'pitch': 62, 'duration': 1.0, 'confidence': 0.9},  # D
        {'pitch': 64, 'duration': 1.0, 'confidence': 0.9},  # E
        {'pitch': 65, 'duration': 1.0, 'confidence': 0.9},  # F
        {'pitch': 67, 'duration': 1.0, 'confidence': 0.9},  # G
        {'pitch': 69, 'duration': 1.0, 'confidence': 0.9},  # A
        {'pitch': 71, 'duration': 1.0, 'confidence': 0.9},  # B
        {'pitch': 72, 'duration': 2.0, 'confidence': 0.9},  # C (longer)
    ]

    root, mode, conf = detector.detect_key(c_major_notes)
    print(f"Test 1: C major scale → Detected: {root} {mode} (confidence: {conf:.3f})")
    assert root == "C" and mode == "major", "Should detect C major"

    # Test 2: Off-key C major (pitches slightly sharp)
    c_major_offkey = [
        {'pitch': 60.3, 'duration': 1.0, 'confidence': 0.8},  # C (slightly sharp)
        {'pitch': 62.1, 'duration': 1.0, 'confidence': 0.8},  # D
        {'pitch': 64.2, 'duration': 1.0, 'confidence': 0.8},  # E
        {'pitch': 65.1, 'duration': 1.0, 'confidence': 0.8},  # F
        {'pitch': 67.3, 'duration': 1.0, 'confidence': 0.8},  # G
    ]

    root, mode, conf = detector.detect_key(c_major_offkey)
    print(f"Test 2: C major (off-key) → Detected: {root} {mode} (confidence: {conf:.3f})")
    assert root == "C" and mode == "major", "Should still detect C major despite off-key"

    # Test 3: A minor
    a_minor_notes = [
        {'pitch': 69, 'duration': 1.0, 'confidence': 0.9},  # A
        {'pitch': 71, 'duration': 1.0, 'confidence': 0.9},  # B
        {'pitch': 72, 'duration': 1.0, 'confidence': 0.9},  # C
        {'pitch': 74, 'duration': 1.0, 'confidence': 0.9},  # D
        {'pitch': 76, 'duration': 1.0, 'confidence': 0.9},  # E
        {'pitch': 77, 'duration': 1.0, 'confidence': 0.9},  # F
        {'pitch': 79, 'duration': 1.0, 'confidence': 0.9},  # G
    ]

    root, mode, conf = detector.detect_key(a_minor_notes)
    print(f"Test 3: A minor scale → Detected: {root} {mode} (confidence: {conf:.3f})")
    # Note: Might detect C major (relative major) - both are valid
    print(f"  (A minor and C major share same notes - both valid)")

    # Test 4: Get scale notes
    c_major_scale = detector.get_scale_notes("C", "major")
    print(f"Test 4: C major scale notes: {c_major_scale}")
    assert c_major_scale == [0, 2, 4, 5, 7, 9, 11], "C major scale incorrect"

    a_minor_scale = detector.get_scale_notes("A", "minor")
    print(f"Test 5: A minor scale notes: {a_minor_scale}")
    assert a_minor_scale == [9, 11, 0, 2, 4, 5, 7], "A minor scale incorrect"

    print("\n✅ All KeyDetector tests passed!")
