"""
Amplitude-Based Onset/Offset Detector

This detector uses primarily RMS energy with spectral flux as supplementary
information. Much better for sustained notes like humming.

Key improvements over pure spectral flux:
- Primary: RMS energy envelope for detecting sound vs silence
- Secondary: Spectral flux for refining exact onset timing
- Handles sustained notes correctly
- More intuitive threshold behavior
"""

import numpy as np
import librosa
import scipy.ndimage
from typing import List, Tuple, Optional


def compute_rms_envelope(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS energy envelope of audio signal.

    RMS (Root Mean Square) is a good indicator of the "loudness" or
    energy of the signal at each point in time.

    Args:
        y: Audio time series
        sr: Sample rate
        frame_length: Window size for RMS computation
        hop_length: Number of samples between frames

    Returns:
        rms: RMS energy envelope (normalized 0-1)
        times: Time in seconds for each frame
    """
    # Compute RMS energy
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # Smooth with median filter to remove spikes
    rms = scipy.ndimage.median_filter(rms, size=3)

    # Light Gaussian smoothing
    rms = scipy.ndimage.gaussian_filter1d(rms, sigma=1.0)

    # Normalize to [0, 1]
    if rms.max() > 0:
        rms = rms / rms.max()

    # Convert frames to time
    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    return rms, times


def compute_spectral_flux(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral flux for refining onset detection.

    Spectral flux measures rapid changes in spectrum, good for
    detecting exact attack transients.

    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between frames

    Returns:
        flux: Spectral flux envelope (normalized 0-1)
        times: Time in seconds for each frame
    """
    # Compute magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Compute spectral flux (positive differences)
    diff = np.diff(S, axis=1)
    flux = np.sum(np.maximum(0, diff), axis=0)

    # Pad to match length
    flux = np.concatenate(([0], flux))

    # Smooth
    flux = scipy.ndimage.median_filter(flux, size=3)
    flux = scipy.ndimage.gaussian_filter1d(flux, sigma=1.0)

    # Normalize
    if flux.max() > 0:
        flux = flux / flux.max()

    # Convert frames to time
    times = librosa.frames_to_time(
        np.arange(len(flux)),
        sr=sr,
        hop_length=hop_length
    )

    return flux, times


def amplitude_based_segment_detection(
    rms: np.ndarray,
    flux: np.ndarray,
    times: np.ndarray,
    onset_high: float = 0.30,
    onset_low: float = 0.10,
    offset_high: float = 0.30,
    offset_low: float = 0.10,
    min_note_len: float = 0.05,
    max_note_len: float = 10.0,
    merge_gap: float = 0.02,
    spectral_weight: float = 0.3
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """
    Detect note segments using primarily RMS energy.

    State machine:
    - SILENCE: RMS below onset_low
    - SOUNDING: RMS above offset_low

    Transitions:
    - SILENCE -> SOUNDING: When RMS rises above onset_high
    - SOUNDING -> SILENCE: When RMS falls below offset_high

    Args:
        rms: RMS energy envelope (0-1)
        flux: Spectral flux envelope (0-1)
        times: Time in seconds for each frame
        onset_high: Threshold to trigger note start
        onset_low: Threshold to sustain note start (hysteresis)
        offset_high: Threshold to trigger note end
        offset_low: Threshold to sustain note (hysteresis)
        min_note_len: Minimum note duration
        max_note_len: Maximum note duration
        merge_gap: Merge notes separated by less than this
        spectral_weight: Weight for spectral flux (0=pure amplitude, 1=equal weight)

    Returns:
        segments: List of (onset_time, offset_time) tuples
        onset_times: List of onset times for visualization
        offset_times: List of offset times for visualization
    """
    # Combine RMS and spectral flux
    # Primarily amplitude-based, but use spectral flux to refine onsets
    combined = rms * (1 - spectral_weight) + flux * spectral_weight

    N = len(rms)
    state = 'silence'
    segments = []
    onset_times = []
    offset_times = []
    cur_onset = None

    for i in range(N):
        t = times[i]
        energy = rms[i]  # Use pure RMS for state decisions
        combined_val = combined[i]  # Use combined for onset detection

        if state == 'silence':
            # Detect onset: energy rises above threshold
            # Use combined signal for onset to catch sharp attacks
            if combined_val >= onset_high:
                state = 'sounding'
                cur_onset = t
                onset_times.append(t)

        elif state == 'sounding':
            # Detect offset: energy falls below threshold
            # Use pure RMS for offset to handle sustained notes
            if energy <= offset_high:
                # Confirm offset by checking if we stay low
                # Look ahead a few frames to avoid false positives
                lookahead = min(i + 5, N)
                if np.mean(rms[i:lookahead]) <= offset_high:
                    cur_offset = t
                    offset_times.append(cur_offset)

                    if cur_onset is not None:
                        duration = cur_offset - cur_onset

                        # Apply duration constraints
                        if duration >= min_note_len:
                            if duration > max_note_len:
                                cur_offset = cur_onset + max_note_len

                            segments.append((cur_onset, cur_offset))

                    cur_onset = None
                    state = 'silence'

    # Handle note still sounding at end
    if state == 'sounding' and cur_onset is not None:
        cur_offset = times[-1]
        offset_times.append(cur_offset)
        duration = cur_offset - cur_onset
        if duration >= min_note_len:
            segments.append((cur_onset, cur_offset))

    # Merge segments separated by tiny gaps
    if len(segments) > 0:
        merged = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue

            prev = merged[-1]
            gap = seg[0] - prev[1]

            if gap <= merge_gap:
                # Merge segments
                merged[-1] = (prev[0], seg[1])
            else:
                merged.append(seg)

        segments = merged

    return segments, onset_times, offset_times


def detect_onsets_offsets_amplitude(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    onset_high: float = 0.30,
    onset_low: float = 0.10,
    offset_high: float = 0.30,
    offset_low: float = 0.10,
    min_note_len: float = 0.05,
    max_note_len: float = 12.0,
    merge_gap: float = 0.03,
    spectral_weight: float = 0.2
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    """
    Main entry point for amplitude-based onset/offset detection.

    This detector uses RMS energy as the primary signal for detecting
    note boundaries, with spectral flux as a supplementary feature.

    Much better than pure spectral flux for sustained notes like humming.

    Args:
        y: Audio time series
        sr: Sample rate
        frame_length: Window size for RMS
        hop_length: Hop length in samples
        onset_high: Energy threshold to trigger onset (0-1)
        onset_low: Energy threshold to sustain onset (0-1, for hysteresis)
        offset_high: Energy threshold to trigger offset (0-1)
        offset_low: Energy threshold to sustain note (0-1, for hysteresis)
        min_note_len: Minimum note duration in seconds
        max_note_len: Maximum note duration in seconds
        merge_gap: Merge segments separated by less than this
        spectral_weight: Weight for spectral flux (0=pure RMS, 0.5=equal weight)

    Returns:
        segments: List of (onset_time, offset_time) tuples
        onset_times: List of individual onset times for visualization
        offset_times: List of individual offset times for visualization

    Example:
        >>> y, sr = librosa.load('humming.wav', sr=16000)
        >>> segments, onsets, offsets = detect_onsets_offsets_amplitude(y, sr)
        >>> for onset, offset in segments:
        ...     print(f"Note: {onset:.3f}s - {offset:.3f}s")
    """
    print(f"[AmplitudeDetector] Using amplitude-based detection")
    print(f"  RMS weight: {1-spectral_weight:.1%}, Spectral weight: {spectral_weight:.1%}")

    # Compute RMS energy envelope
    rms, times = compute_rms_envelope(y, sr, frame_length, hop_length)

    # Compute spectral flux for onset refinement
    flux, _ = compute_spectral_flux(y, sr, frame_length, hop_length)

    # Ensure same length (might differ slightly due to padding)
    min_len = min(len(rms), len(flux))
    rms = rms[:min_len]
    flux = flux[:min_len]
    times = times[:min_len]

    # Detect segments
    segments, onset_times, offset_times = amplitude_based_segment_detection(
        rms, flux, times,
        onset_high=onset_high,
        onset_low=onset_low,
        offset_high=offset_high,
        offset_low=offset_low,
        min_note_len=min_note_len,
        max_note_len=max_note_len,
        merge_gap=merge_gap,
        spectral_weight=spectral_weight
    )

    print(f"  Detected {len(segments)} segments")

    return segments, onset_times, offset_times


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python amplitude_onset_offset_detector.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Loading audio: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio loaded: {len(y)/sr:.2f}s @ {sr}Hz")

    print("\nDetecting onsets/offsets with amplitude-based method...")
    segments, onsets, offsets = detect_onsets_offsets_amplitude(
        y, sr,
        onset_high=0.15,
        offset_high=0.10,
        min_note_len=0.1
    )

    print(f"\nDetected {len(segments)} segments:\n")
    for i, (onset, offset) in enumerate(segments, 1):
        duration = offset - onset
        print(f"Segment {i:3d}: {onset:6.3f}s - {offset:6.3f}s  (duration: {duration:6.3f}s)")
