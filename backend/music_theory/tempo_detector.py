"""
Tempo Detector - BPM Detection from Onset Intervals

Detects tempo by analyzing inter-onset intervals (IOIs) and finding
periodic patterns.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter


class TempoDetector:
    """
    Detects tempo from note onset times.

    Uses inter-onset interval analysis to find the underlying beat.
    """

    def __init__(
        self,
        min_bpm: int = 40,
        max_bpm: int = 200,
        default_bpm: float = 120.0
    ):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.default_bpm = default_bpm

        print(f"[TempoDetector] Initialized (BPM range: {min_bpm}-{max_bpm})")

    def detect_tempo(
        self,
        onset_times: List[float],
        min_onsets: int = 4
    ) -> Tuple[float, float]:
        """
        Detect tempo from onset times.

        Args:
            onset_times: List of onset times in seconds
            min_onsets: Minimum number of onsets required

        Returns:
            (bpm, confidence)
        """
        if len(onset_times) < min_onsets:
            print(f"[TempoDetector] Too few onsets ({len(onset_times)}), using default {self.default_bpm} BPM")
            return (self.default_bpm, 0.0)

        # Calculate inter-onset intervals (IOIs)
        iois = np.diff(onset_times)

        if len(iois) == 0:
            return (self.default_bpm, 0.0)

        print(f"[TempoDetector] Analyzing {len(iois)} inter-onset intervals")
        print(f"  IOI range: {iois.min():.3f}s - {iois.max():.3f}s")

        # Find most common IOI (this represents the beat duration)
        # Use histogram with bins to handle slight variations
        ioi_bins = np.histogram(iois, bins=20)
        most_common_ioi = ioi_bins[1][ioi_bins[0].argmax()]

        # Also try median (robust to outliers)
        median_ioi = np.median(iois)

        # And mode with clustering
        mode_ioi = self._find_mode_ioi(iois)

        print(f"  Most common IOI (hist): {most_common_ioi:.3f}s")
        print(f"  Median IOI: {median_ioi:.3f}s")
        print(f"  Mode IOI (clustered): {mode_ioi:.3f}s")

        # Use mode IOI as beat duration
        beat_duration = mode_ioi

        # Convert to BPM
        bpm = 60.0 / beat_duration

        # Check if we need to double/halve (common mistake)
        if bpm < self.min_bpm:
            bpm *= 2  # Double time
            print(f"  BPM too slow, doubling: {bpm:.1f}")
        elif bpm > self.max_bpm:
            bpm /= 2  # Half time
            print(f"  BPM too fast, halving: {bpm:.1f}")

        # Clip to valid range
        bpm = np.clip(bpm, self.min_bpm, self.max_bpm)

        # Calculate confidence based on IOI consistency
        ioi_std = np.std(iois)
        ioi_mean = np.mean(iois)
        coefficient_of_variation = ioi_std / (ioi_mean + 1e-6)

        # Lower CV = more consistent = higher confidence
        confidence = np.exp(-coefficient_of_variation * 2)  # 0 to 1
        confidence = np.clip(confidence, 0.0, 1.0)

        print(f"[TempoDetector] Detected tempo: {bpm:.1f} BPM (confidence: {confidence:.2f})")

        return (float(bpm), float(confidence))

    def _find_mode_ioi(self, iois: np.ndarray, tolerance: float = 0.05) -> float:
        """
        Find mode IOI with clustering (group similar values).

        Args:
            iois: Inter-onset intervals
            tolerance: Relative tolerance for grouping (5%)

        Returns:
            Mode IOI
        """
        if len(iois) == 0:
            return 0.5  # Default to 500ms

        # Sort IOIs
        sorted_iois = np.sort(iois)

        # Group similar IOIs
        clusters = []
        current_cluster = [sorted_iois[0]]

        for i in range(1, len(sorted_iois)):
            # Check if within tolerance of cluster mean
            cluster_mean = np.mean(current_cluster)
            if abs(sorted_iois[i] - cluster_mean) / (cluster_mean + 1e-6) < tolerance:
                current_cluster.append(sorted_iois[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_iois[i]]

        clusters.append(current_cluster)

        # Find largest cluster
        largest_cluster = max(clusters, key=len)

        # Return mean of largest cluster
        return np.mean(largest_cluster)

    def get_grid_resolution(
        self,
        tempo: float,
        note_durations: List[float]
    ) -> str:
        """
        Determine appropriate rhythmic grid resolution.

        Args:
            tempo: Tempo in BPM
            note_durations: List of note durations in seconds

        Returns:
            Grid resolution: "1/4", "1/8", "1/16", or "1/32"
        """
        if not note_durations:
            return "1/16"  # Default

        beat_duration = 60.0 / tempo

        # Find shortest note duration
        min_duration = min(note_durations)

        # Determine grid based on shortest note
        # 1/4 note = 1.0 beats
        # 1/8 note = 0.5 beats
        # 1/16 note = 0.25 beats
        # 1/32 note = 0.125 beats

        min_beats = min_duration / beat_duration

        if min_beats >= 0.75:  # >= 3/4 beat
            grid = "1/4"
        elif min_beats >= 0.375:  # >= 3/8 beat
            grid = "1/8"
        elif min_beats >= 0.1875:  # >= 3/16 beat
            grid = "1/16"
        else:
            grid = "1/32"

        print(f"[TempoDetector] Grid resolution: {grid} (min duration: {min_duration:.3f}s = {min_beats:.3f} beats)")

        return grid


if __name__ == '__main__':
    print("\nTesting TempoDetector...")

    detector = TempoDetector()

    # Test 1: Steady 120 BPM (quarter notes)
    # Beat duration = 60/120 = 0.5s
    onset_times_120 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    bpm, confidence = detector.detect_tempo(onset_times_120)
    print(f"Test 1: Steady 120 BPM quarter notes")
    print(f"  Detected: {bpm:.1f} BPM (confidence: {confidence:.2f})")
    assert 115 < bpm < 125, f"Should detect ~120 BPM, got {bpm}"
    print("  ✅ Correct tempo detected")

    # Test 2: Slower tempo (80 BPM)
    # Beat duration = 60/80 = 0.75s
    onset_times_80 = [0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5]

    bpm, confidence = detector.detect_tempo(onset_times_80)
    print(f"\nTest 2: Steady 80 BPM")
    print(f"  Detected: {bpm:.1f} BPM (confidence: {confidence:.2f})")
    assert 75 < bpm < 85, f"Should detect ~80 BPM, got {bpm}"
    print("  ✅ Correct tempo detected")

    # Test 3: Faster tempo (140 BPM)
    # Beat duration = 60/140 = 0.4286s
    beat_dur = 60/140
    onset_times_140 = [i * beat_dur for i in range(8)]

    bpm, confidence = detector.detect_tempo(onset_times_140)
    print(f"\nTest 3: Steady 140 BPM")
    print(f"  Detected: {bpm:.1f} BPM (confidence: {confidence:.2f})")
    assert 135 < bpm < 145, f"Should detect ~140 BPM, got {bpm}"
    print("  ✅ Correct tempo detected")

    # Test 4: Grid resolution
    # At 120 BPM (0.5s beat):
    # - 1/4 note = 0.5s
    # - 1/8 note = 0.25s
    # - 1/16 note = 0.125s

    durations_quarters = [0.5, 0.5, 0.5, 1.0]  # Quarter and half notes
    grid = detector.get_grid_resolution(120, durations_quarters)
    print(f"\nTest 4: Grid for quarter notes at 120 BPM")
    print(f"  Grid: {grid}")
    assert grid in ["1/4", "1/8"], f"Should be 1/4 or 1/8, got {grid}"

    durations_sixteenths = [0.125, 0.125, 0.25, 0.25]  # Sixteenth and eighth notes
    grid = detector.get_grid_resolution(120, durations_sixteenths)
    print(f"\nTest 5: Grid for sixteenth notes at 120 BPM")
    print(f"  Grid: {grid}")
    assert grid == "1/16", f"Should be 1/16, got {grid}"

    print("\n✅ All TempoDetector tests passed!")
