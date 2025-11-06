"""
Onset/Offset Cleaner - Reduces Over-Prediction

The model over-predicts onsets/offsets by ~250% because humming lacks
sharp attack/release. This module clusters nearby predictions and keeps
only the highest confidence prediction from each cluster.
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy import signal


class OnsetOffsetCleaner:
    """
    Reduces onset/offset over-prediction through clustering.

    Target: Reduce predictions by ~60-70% (from 250% over-prediction to reasonable).
    """

    def __init__(
        self,
        onset_threshold: float = 0.5,
        cluster_window: float = 0.15,  # 150ms window for clustering
        min_note_duration: float = 0.08  # Minimum 80ms note
    ):
        self.onset_threshold = onset_threshold
        self.cluster_window = cluster_window
        self.min_note_duration = min_note_duration

        print(f"[OnsetOffsetCleaner] Initialized:")
        print(f"  Onset threshold: {onset_threshold}")
        print(f"  Cluster window: {cluster_window}s")
        print(f"  Min note duration: {min_note_duration}s")

    def cluster_onsets(
        self,
        onset_probs: np.ndarray,  # (time_steps,)
        frame_rate: float = 7.8125  # fps
    ) -> List[Tuple[float, float]]:
        """
        Find onset peaks and cluster nearby predictions.

        Args:
            onset_probs: Onset probabilities at each time step
            frame_rate: Frames per second

        Returns:
            List of (time, confidence) for cleaned onsets
        """
        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            onset_probs,
            height=self.onset_threshold,
            distance=1  # At least 1 frame apart
        )

        if len(peaks) == 0:
            print("[OnsetCleaner] No onsets detected above threshold")
            return []

        # Convert frame indices to time
        peak_times = peaks / frame_rate
        peak_confidences = properties['peak_heights']

        print(f"[OnsetCleaner] Found {len(peaks)} onset peaks")

        # Cluster nearby peaks
        clusters = []
        current_cluster = [(peak_times[0], peak_confidences[0])]

        for i in range(1, len(peak_times)):
            time_diff = peak_times[i] - peak_times[i-1]

            if time_diff < self.cluster_window:
                # Same cluster
                current_cluster.append((peak_times[i], peak_confidences[i]))
            else:
                # New cluster - save previous and start new
                clusters.append(current_cluster)
                current_cluster = [(peak_times[i], peak_confidences[i])]

        # Don't forget last cluster
        clusters.append(current_cluster)

        # Keep highest confidence from each cluster
        cleaned_onsets = []
        for cluster in clusters:
            # Find max confidence in cluster
            best = max(cluster, key=lambda x: x[1])
            cleaned_onsets.append(best)

        print(f"[OnsetCleaner] After clustering: {len(cleaned_onsets)} onsets")
        print(f"[OnsetCleaner] Reduction: {len(peaks)} → {len(cleaned_onsets)} ({100 * (1 - len(cleaned_onsets)/len(peaks)):.1f}% reduction)")

        return cleaned_onsets

    def cluster_offsets(
        self,
        offset_probs: np.ndarray,  # (time_steps,)
        onset_times: List[float],   # Cleaned onset times
        frame_rate: float = 7.8125,
        max_note_duration: float = 4.0  # Max 4 seconds
    ) -> List[Tuple[float, float]]:
        """
        Match offsets to onsets.

        For each onset, find the nearest offset peak after it within a
        reasonable duration window.

        Args:
            offset_probs: Offset probabilities at each time step
            onset_times: Cleaned onset times (from cluster_onsets)
            frame_rate: Frames per second
            max_note_duration: Maximum note duration in seconds

        Returns:
            List of (time, confidence) for cleaned offsets (same length as onsets)
        """
        if not onset_times:
            return []

        # Find ALL offset peaks (lower threshold than onsets)
        offset_peaks, offset_props = signal.find_peaks(
            offset_probs,
            height=self.onset_threshold * 0.7,  # Lower threshold for offsets
            distance=1
        )

        offset_peak_times = offset_peaks / frame_rate
        offset_confidences = offset_props['peak_heights']

        print(f"[OffsetCleaner] Found {len(offset_peaks)} offset peaks")

        # Match each onset to nearest offset
        cleaned_offsets = []

        for onset_time in onset_times:
            # Search window: [onset + min_duration, onset + max_duration]
            search_start = onset_time + self.min_note_duration
            search_end = onset_time + max_note_duration

            # Find offsets in search window
            valid_offsets = [
                (time, conf)
                for time, conf in zip(offset_peak_times, offset_confidences)
                if search_start <= time <= search_end
            ]

            if valid_offsets:
                # Take offset with highest confidence
                best_offset = max(valid_offsets, key=lambda x: x[1])
                cleaned_offsets.append(best_offset)
            else:
                # No offset found - create one at onset + min_duration
                default_offset_time = onset_time + self.min_note_duration
                cleaned_offsets.append((default_offset_time, 0.3))

        print(f"[OffsetCleaner] Matched {len(cleaned_offsets)} offsets to onsets")

        return cleaned_offsets

    def extract_notes_from_cleaned_events(
        self,
        onset_times: List[float],
        onset_confidences: List[float],
        offset_times: List[float],
        offset_confidences: List[float],
        frame_probs: np.ndarray,  # (time_steps, 88)
        frame_rate: float = 7.8125
    ) -> List[Dict]:
        """
        Extract notes using cleaned onset/offset pairs.

        Args:
            onset_times: Cleaned onset times
            onset_confidences: Onset confidences
            offset_times: Matched offset times
            offset_confidences: Offset confidences
            frame_probs: Frame-level pitch probabilities (time_steps, 88)
            frame_rate: Frames per second

        Returns:
            List of notes: [{pitch, start, duration, confidence}, ...]
        """
        notes = []

        for i, (onset_time, offset_time) in enumerate(zip(onset_times, offset_times)):
            # Get frame indices for this note
            onset_frame = int(onset_time * frame_rate)
            offset_frame = int(offset_time * frame_rate)

            # Clip to valid range
            onset_frame = max(0, min(onset_frame, len(frame_probs) - 1))
            offset_frame = max(onset_frame + 1, min(offset_frame, len(frame_probs)))

            # Find dominant pitch in this time range
            pitch_activations = frame_probs[onset_frame:offset_frame, :]

            if len(pitch_activations) == 0:
                continue

            # Average across time to get pitch strengths
            avg_activations = pitch_activations.mean(axis=0)

            # Get strongest pitch
            best_pitch_idx = avg_activations.argmax()
            best_pitch_conf = avg_activations[best_pitch_idx]

            # Convert to MIDI (assuming min_midi=21)
            midi_pitch = 21 + best_pitch_idx

            # Calculate duration
            duration = offset_time - onset_time

            # Overall confidence (combine onset, offset, and pitch confidences)
            confidence = (
                0.4 * onset_confidences[i] +
                0.3 * offset_confidences[i] +
                0.3 * best_pitch_conf
            )

            notes.append({
                'pitch': midi_pitch,
                'start': onset_time,
                'duration': duration,
                'confidence': float(confidence)
            })

        print(f"[NoteExtractor] Extracted {len(notes)} notes from cleaned events")

        return notes


if __name__ == '__main__':
    print("\nTesting OnsetOffsetCleaner...")

    cleaner = OnsetOffsetCleaner()

    # Create synthetic onset probabilities with over-prediction
    # Simulating 3 real onsets but with clusters of false positives around each
    onset_probs = np.zeros(200)
    # Real onset at frame 20, but with spurious detections nearby
    onset_probs[18:23] = [0.3, 0.6, 0.9, 0.7, 0.4]  # Peak at 20
    # Real onset at frame 80
    onset_probs[78:83] = [0.4, 0.7, 0.85, 0.6, 0.3]  # Peak at 80
    # Real onset at frame 140
    onset_probs[138:143] = [0.35, 0.65, 0.95, 0.55, 0.25]  # Peak at 140

    print(f"Test 1: Simulated onset over-prediction")
    print(f"  Total onset peaks above threshold: {(onset_probs > 0.5).sum()}")

    cleaned_onsets = cleaner.cluster_onsets(onset_probs, frame_rate=10.0)
    print(f"  Cleaned onsets: {len(cleaned_onsets)}")
    assert len(cleaned_onsets) == 3, f"Expected 3 onsets, got {len(cleaned_onsets)}"
    print(f"  ✅ Correctly reduced to 3 onsets")

    # Create offset probabilities
    offset_probs = np.zeros(200)
    offset_probs[50] = 0.8
    offset_probs[110] = 0.75
    offset_probs[170] = 0.85

    onset_times_only = [t for t, _ in cleaned_onsets]
    cleaned_offsets = cleaner.cluster_offsets(
        offset_probs,
        onset_times_only,
        frame_rate=10.0
    )

    print(f"\nTest 2: Offset matching")
    print(f"  Matched {len(cleaned_offsets)} offsets to {len(cleaned_onsets)} onsets")
    assert len(cleaned_offsets) == len(cleaned_onsets), "Should have equal onsets and offsets"
    print(f"  ✅ Each onset matched to offset")

    # Test note extraction
    frame_probs = np.random.rand(200, 88) * 0.3
    # Make C4 (pitch 60, index 39) dominant in each segment
    frame_probs[18:50, 39] = 0.9
    frame_probs[78:110, 42] = 0.85  # E4
    frame_probs[138:170, 45] = 0.92  # G4

    onset_conf = [c for _, c in cleaned_onsets]
    offset_times = [t for t, _ in cleaned_offsets]
    offset_conf = [c for _, c in cleaned_offsets]

    notes = cleaner.extract_notes_from_cleaned_events(
        onset_times_only,
        onset_conf,
        offset_times,
        offset_conf,
        frame_probs,
        frame_rate=10.0
    )

    print(f"\nTest 3: Note extraction")
    print(f"  Extracted {len(notes)} notes")
    for i, note in enumerate(notes):
        print(f"    Note {i+1}: MIDI {note['pitch']}, start={note['start']:.2f}s, dur={note['duration']:.2f}s, conf={note['confidence']:.2f}")

    assert len(notes) == 3, f"Expected 3 notes, got {len(notes)}"
    print(f"  ✅ Extracted correct number of notes")

    print("\n✅ All OnsetOffsetCleaner tests passed!")
