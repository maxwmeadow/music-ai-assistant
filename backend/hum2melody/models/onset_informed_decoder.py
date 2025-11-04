"""
Onset-Informed Viterbi Decoder for Melody Post-Processing

This decoder uses:
1. Onset predictions to segment the audio into note candidates
2. Frame predictions to determine pitch within each segment
3. Viterbi algorithm with musical priors to smooth transitions

This is MUCH better than simple thresholding or median filtering.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Note:
    """Represents a detected note."""
    pitch: int  # MIDI note number
    start: float  # seconds
    duration: float  # seconds
    confidence: float  # 0-1


class OnsetInformedDecoder:
    """
    Decode frame and onset predictions into discrete notes.
    
    Algorithm:
    1. Detect onsets from onset predictions
    2. Segment between consecutive onsets
    3. For each segment, estimate the most likely pitch
    4. Apply Viterbi smoothing with musical transition priors
    5. Post-process to remove very short notes and merge duplicates
    """
    
    def __init__(
        self,
        min_midi: int = 21,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.4,
        min_note_duration: float = 0.08,  # 80ms minimum
        max_pitch_jump: int = 12,  # 1 octave
        transition_penalty: float = 0.5,
        onset_tolerance: float = 0.05,  # 50ms window around onsets
        frame_rate: float = 31.25  # frames per second
    ):
        self.min_midi = min_midi
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_duration = min_note_duration
        self.max_pitch_jump = max_pitch_jump
        self.transition_penalty = transition_penalty
        self.onset_tolerance = onset_tolerance
        self.frame_rate = frame_rate
    
    def decode(
        self,
        frame_probs: np.ndarray,  # (time, 88) - pitch probabilities
        onset_probs: np.ndarray   # (time,) - onset probabilities
    ) -> List[Note]:
        """
        Main decoding function.
        
        Args:
            frame_probs: Frame-level pitch probabilities (after sigmoid)
            onset_probs: Onset probabilities (after sigmoid)
        
        Returns:
            List of Note objects
        """
        # Step 1: Detect onsets
        onset_frames = self._detect_onsets(onset_probs)
        
        if len(onset_frames) == 0:
            # No onsets detected - treat as one continuous segment
            onset_frames = np.array([0])
        
        # Step 2: Segment between onsets
        segments = self._create_segments(onset_frames, len(frame_probs))
        
        # Step 3: Estimate pitch for each segment
        segment_pitches = []
        segment_confidences = []
        
        for start_frame, end_frame in segments:
            if end_frame - start_frame < 1:
                continue
            
            segment_probs = frame_probs[start_frame:end_frame]
            pitch, confidence = self._estimate_pitch_in_segment(segment_probs)
            
            if confidence > self.frame_threshold:
                segment_pitches.append(pitch)
                segment_confidences.append(confidence)
            else:
                segment_pitches.append(-1)  # Silence
                segment_confidences.append(0.0)
        
        # Step 4: Apply Viterbi smoothing
        smoothed_pitches = self._viterbi_smooth(
            segment_pitches,
            segment_confidences
        )
        
        # Step 5: Convert to Note objects
        notes = []
        for i, (start_frame, end_frame) in enumerate(segments):
            if i >= len(smoothed_pitches):
                break
            
            pitch = smoothed_pitches[i]
            
            if pitch == -1:  # Silence
                continue
            
            start_time = start_frame / self.frame_rate
            duration = (end_frame - start_frame) / self.frame_rate
            confidence = segment_confidences[i] if i < len(segment_confidences) else 0.5
            
            notes.append(Note(
                pitch=int(pitch + self.min_midi),
                start=float(start_time),
                duration=float(duration),
                confidence=float(confidence)
            ))
        
        # Step 6: Post-process
        notes = self._merge_same_pitch_notes(notes)
        notes = self._filter_short_notes(notes)
        
        return notes
    
    def _detect_onsets(self, onset_probs: np.ndarray) -> np.ndarray:
        """
        Detect onset frames using peak detection.
        
        Returns array of frame indices where onsets occur.
        """
        # Threshold
        candidate_frames = np.where(onset_probs > self.onset_threshold)[0]
        
        if len(candidate_frames) == 0:
            return np.array([])
        
        # Peak detection - keep only local maxima
        onsets = []
        tolerance_frames = int(self.onset_tolerance * self.frame_rate)
        
        i = 0
        while i < len(candidate_frames):
            # Find local maximum in window
            window_start = i
            window_end = i + 1
            
            while (window_end < len(candidate_frames) and 
                   candidate_frames[window_end] - candidate_frames[window_start] <= tolerance_frames):
                window_end += 1
            
            # Get frame with highest probability in this window
            window_frames = candidate_frames[window_start:window_end]
            window_probs = onset_probs[window_frames]
            best_idx = window_frames[np.argmax(window_probs)]
            
            onsets.append(best_idx)
            i = window_end
        
        return np.array(onsets)
    
    def _create_segments(
        self,
        onset_frames: np.ndarray,
        total_frames: int
    ) -> List[Tuple[int, int]]:
        """
        Create segments between consecutive onsets.
        
        Returns list of (start_frame, end_frame) tuples.
        """
        segments = []
        
        for i in range(len(onset_frames)):
            start = onset_frames[i]
            
            if i < len(onset_frames) - 1:
                end = onset_frames[i + 1]
            else:
                end = total_frames
            
            segments.append((start, end))
        
        return segments
    
    def _estimate_pitch_in_segment(
        self,
        segment_probs: np.ndarray  # (segment_len, 88)
    ) -> Tuple[int, float]:
        """
        Estimate the most likely pitch in a segment.
        
        Strategy: Use weighted median pitch across the segment,
        weighted by frame confidence.
        
        Returns:
            (pitch_index, confidence)
        """
        if len(segment_probs) == 0:
            return -1, 0.0
        
        # For each frame, get the most confident pitch
        max_probs = segment_probs.max(axis=1)
        max_pitches = segment_probs.argmax(axis=1)
        
        # Filter out low-confidence frames
        confident_mask = max_probs > self.frame_threshold
        
        if not confident_mask.any():
            return -1, 0.0
        
        confident_pitches = max_pitches[confident_mask]
        confident_probs = max_probs[confident_mask]
        
        # Weighted histogram
        pitch_scores = np.zeros(88)
        for pitch, prob in zip(confident_pitches, confident_probs):
            pitch_scores[pitch] += prob
        
        # Choose pitch with highest score
        best_pitch = pitch_scores.argmax()
        confidence = pitch_scores[best_pitch] / len(segment_probs)
        
        return int(best_pitch), float(confidence)
    
    def _viterbi_smooth(
        self,
        pitches: List[int],
        confidences: List[float]
    ) -> List[int]:
        """
        Apply Viterbi algorithm to smooth pitch transitions.
        
        Uses musical priors:
        - Same pitch: no penalty
        - Small steps (1-2 semitones): small penalty
        - Large steps (>5 semitones): larger penalty
        - Very large jumps (>12 semitones): heavy penalty
        """
        if len(pitches) <= 1:
            return pitches
        
        num_states = 89  # 88 pitches + 1 silence (-1)
        num_frames = len(pitches)
        
        # Initialize DP tables
        dp = np.full((num_frames, num_states), -np.inf)
        backpointer = np.zeros((num_frames, num_states), dtype=int)
        
        # Initial probabilities (first frame)
        for state in range(num_states):
            pitch = state - 1  # -1 for silence, 0-87 for pitches
            if pitch == pitches[0]:
                dp[0, state] = np.log(confidences[0] + 1e-8)
            else:
                dp[0, state] = np.log(1e-8)
        
        # Forward pass
        for t in range(1, num_frames):
            for curr_state in range(num_states):
                curr_pitch = curr_state - 1
                
                # Observation probability
                if curr_pitch == pitches[t]:
                    obs_prob = np.log(confidences[t] + 1e-8)
                else:
                    obs_prob = np.log(1e-8)
                
                # Transition probabilities from all previous states
                for prev_state in range(num_states):
                    prev_pitch = prev_state - 1
                    
                    # Transition cost
                    trans_cost = self._transition_cost(prev_pitch, curr_pitch)
                    
                    # Total score
                    score = dp[t-1, prev_state] + obs_prob - trans_cost
                    
                    if score > dp[t, curr_state]:
                        dp[t, curr_state] = score
                        backpointer[t, curr_state] = prev_state
        
        # Backward pass (backtracking)
        best_path = []
        best_final_state = dp[-1].argmax()
        
        state = best_final_state
        for t in range(num_frames - 1, -1, -1):
            best_path.append(state - 1)  # Convert state to pitch
            if t > 0:
                state = backpointer[t, state]
        
        best_path.reverse()
        
        return best_path
    
    def _transition_cost(self, prev_pitch: int, curr_pitch: int) -> float:
        """
        Calculate transition cost between two pitches.
        
        Musical priors:
        - Same pitch: free (cost 0)
        - Small step: cheap
        - Large jump: expensive
        - Silence transitions: moderate cost
        """
        # Silence transitions
        if prev_pitch == -1 or curr_pitch == -1:
            return 0.5  # Moderate cost for silence entry/exit
        
        # Same pitch (sustain)
        if prev_pitch == curr_pitch:
            return 0.0
        
        # Calculate interval
        interval = abs(curr_pitch - prev_pitch)
        
        # Penalize based on interval size
        if interval <= 2:
            return 0.1 * self.transition_penalty  # Whole tone or less
        elif interval <= 5:
            return 0.3 * self.transition_penalty  # Up to a fourth
        elif interval <= 7:
            return 0.5 * self.transition_penalty  # Fifth
        elif interval <= 12:
            return 1.0 * self.transition_penalty  # Up to an octave
        else:
            return 2.0 * self.transition_penalty  # More than octave (unusual)
    
    def _merge_same_pitch_notes(self, notes: List[Note]) -> List[Note]:
        """Merge consecutive notes with the same pitch."""
        if len(notes) <= 1:
            return notes
        
        merged = []
        current = notes[0]
        
        for next_note in notes[1:]:
            # Same pitch and close in time?
            gap = next_note.start - (current.start + current.duration)
            
            if (next_note.pitch == current.pitch and 
                gap < self.min_note_duration):
                # Merge
                current.duration = (next_note.start + next_note.duration) - current.start
                current.confidence = max(current.confidence, next_note.confidence)
            else:
                # Save current and start new
                merged.append(current)
                current = next_note
        
        merged.append(current)
        return merged
    
    def _filter_short_notes(self, notes: List[Note]) -> List[Note]:
        """Remove notes shorter than minimum duration."""
        return [
            note for note in notes
            if note.duration >= self.min_note_duration
        ]


def test_decoder():
    """Test the onset-informed decoder."""
    print("Testing OnsetInformedDecoder...")
    
    # Create synthetic predictions
    num_frames = 200
    frame_probs = np.zeros((num_frames, 88))
    onset_probs = np.zeros(num_frames)
    
    # Create some notes
    # Note 1: C4 (pitch 39) from frame 0-50
    frame_probs[0:50, 39] = 0.9
    onset_probs[0] = 0.95
    
    # Note 2: E4 (pitch 43) from frame 50-100
    frame_probs[50:100, 43] = 0.85
    onset_probs[50] = 0.90
    
    # Note 3: G4 (pitch 46) from frame 100-150
    frame_probs[100:150, 46] = 0.80
    onset_probs[100] = 0.85
    
    # Create decoder
    decoder = OnsetInformedDecoder(min_midi=21)
    
    # Decode
    notes = decoder.decode(frame_probs, onset_probs)
    
    print(f"\n✅ Decoded {len(notes)} notes:")
    for i, note in enumerate(notes):
        print(f"  {i+1}. MIDI {note.pitch} @ {note.start:.2f}s for {note.duration:.2f}s (conf: {note.confidence:.2f})")
    
    # Verify
    assert len(notes) == 3, f"Expected 3 notes, got {len(notes)}"
    assert notes[0].pitch == 60, f"Note 1 should be C4 (60), got {notes[0].pitch}"
    assert notes[1].pitch == 64, f"Note 2 should be E4 (64), got {notes[1].pitch}"
    assert notes[2].pitch == 67, f"Note 3 should be G4 (67), got {notes[2].pitch}"
    
    print("\n✅ All tests passed!")
    print("\nKey features:")
    print("  ✅ Uses onset predictions to segment audio")
    print("  ✅ Estimates pitch per segment (robust to noise)")
    print("  ✅ Viterbi smoothing with musical priors")
    print("  ✅ Merges duplicate notes and filters short notes")


if __name__ == '__main__':
    test_decoder()
