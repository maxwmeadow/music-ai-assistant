"""
Multi-Band Onset/Offset Detector using Spectral Flux

This module implements a sophisticated onset and offset detection algorithm
using multi-band spectral flux analysis with hysteresis thresholding.

Key features:
- Multi-band spectral analysis (4 frequency bands)
- Separate onset and offset envelope computation
- Hysteresis state machine for stable segment detection
- Tunable sensitivity parameters

Usage:
    from data.onset_offset_detector import detect_onsets_offsets

    y, sr = librosa.load('audio.wav', sr=16000)
    segments = detect_onsets_offsets(y, sr)
    # Returns: [(onset1, offset1), (onset2, offset2), ...]
"""

import numpy as np
import librosa
import scipy.ndimage
from typing import List, Tuple, Optional


def multi_band_spectral_flux(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    bands: Optional[List[Tuple[float, float]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-band spectral flux novelty function.

    Spectral flux measures the rate of change in the magnitude spectrum,
    split across multiple frequency bands to capture different aspects
    of the audio (e.g., low frequencies for bass, high for harmonics).

    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between frames
        bands: List of (fmin, fmax) tuples in Hz. If None, uses default 4 bands.

    Returns:
        combined: Combined spectral flux (frames,)
        times: Time in seconds for each frame
    """
    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Default frequency bands (Hz): low, mid, high, very-high
    if bands is None:
        bands = [(0, 200), (200, 2000), (2000, 5000), (5000, sr // 2)]

    band_fluxes = []
    for fmin, fmax in bands:
        # Find frequency bin indices for this band
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]

        if idx.size == 0:
            # Empty band, skip
            band_fluxes.append(np.zeros(S.shape[1]))
            continue

        # Extract spectral magnitude for this band
        S_band = S[idx, :]

        # Spectral flux: positive difference between successive frames
        diff = np.diff(S_band, axis=1)

        # Only keep positive changes (energy increases)
        pos = np.clip(diff, a_min=0, a_max=None)

        # Sum across frequency bins to get single flux value per frame
        flux = pos.sum(axis=0)

        # Pad to same length as frames (diff reduces length by 1)
        flux = np.concatenate(([0.0], flux))

        band_fluxes.append(flux)

    # Normalize each band to [0, 1]
    band_fluxes = np.array(band_fluxes)
    max_vals = np.max(band_fluxes, axis=1, keepdims=True)
    band_fluxes = band_fluxes / (max_vals + 1e-8)  # Avoid divide-by-zero

    # Combine bands (simple sum)
    combined = np.sum(band_fluxes, axis=0)

    # Convert frames to time
    times = librosa.frames_to_time(
        np.arange(combined.shape[0]),
        sr=sr,
        hop_length=hop_length
    )

    return combined, times


def compute_onset_offset_envelopes(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute separate onset and offset envelopes from spectral flux.

    Uses signed spectral flux to separate:
    - Onset envelope: Positive spectral flux (energy increases)
    - Offset envelope: Negative spectral flux (energy decreases)

    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Number of samples between frames

    Returns:
        onset_env: Onset detection envelope (frames,)
        offset_env: Offset detection envelope (frames,)
        times: Time in seconds for each frame
    """
    # Frequency bands for multi-band analysis
    bands = [(0, 200), (200, 2000), (2000, 5000), (5000, sr // 2)]

    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute SIGNED flux per band (preserves positive/negative changes)
    signed_band_flux = []
    for fmin, fmax in bands:
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]

        if idx.size == 0:
            signed_band_flux.append(np.zeros(S.shape[1]))
            continue

        S_band = S[idx, :]

        # Signed difference (preserves both increases and decreases)
        diff = np.diff(S_band, axis=1)

        # Sum across frequency bins
        signed = diff.sum(axis=0)

        # Pad to same length
        signed = np.concatenate(([0.0], signed))

        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(signed))
        if max_abs > 0:
            signed = signed / max_abs

        signed_band_flux.append(signed)

    # Combine bands
    signed_band_flux = np.array(signed_band_flux)
    signed = np.sum(signed_band_flux, axis=0)

    # Separate into onset and offset envelopes
    onset_env = np.clip(signed, a_min=0, a_max=None)    # Positive = onset
    offset_env = np.clip(-signed, a_min=0, a_max=None)  # Negative = offset

    # Smooth envelopes to reduce jitter
    # Median filter removes spikes
    onset_env = scipy.ndimage.median_filter(onset_env, size=3)
    offset_env = scipy.ndimage.median_filter(offset_env, size=3)

    # Gaussian filter for temporal smoothing
    onset_env = scipy.ndimage.gaussian_filter1d(onset_env, sigma=1.0)
    offset_env = scipy.ndimage.gaussian_filter1d(offset_env, sigma=1.0)

    # Convert frames to time
    times = librosa.frames_to_time(
        np.arange(onset_env.shape[0]),
        sr=sr,
        hop_length=hop_length
    )

    return onset_env, offset_env, times


def hysteresis_segment_detection(
    onset_env: np.ndarray,
    offset_env: np.ndarray,
    times: np.ndarray,
    onset_high: float = 0.35,
    onset_low: float = 0.15,
    offset_high: float = 0.35,
    offset_low: float = 0.15,
    min_note_len: float = 0.05,
    max_note_len: float = 10.0,
    merge_gap: float = 0.02
) -> List[Tuple[float, float]]:
    """
    Use dual-threshold hysteresis to detect note segments.

    State machine:
    - SILENCE state: Wait for onset envelope > onset_high
    - SOUNDING state: Wait for offset envelope > offset_high

    Hysteresis prevents jitter by requiring strong signal to trigger
    state change, but allowing weaker signal to sustain state.

    Args:
        onset_env: Onset detection envelope (normalized 0-1)
        offset_env: Offset detection envelope (normalized 0-1)
        times: Time in seconds for each frame
        onset_high: High threshold for onset trigger
        onset_low: Low threshold for onset sustain
        offset_high: High threshold for offset trigger
        offset_low: Low threshold for offset sustain
        min_note_len: Minimum note duration in seconds
        max_note_len: Maximum note duration in seconds
        merge_gap: Merge segments separated by < this gap

    Returns:
        segments: List of (onset_time, offset_time) tuples
    """
    # Normalize envelopes to [0, 1]
    def normalize(a):
        mx = a.max()
        if mx == 0:
            return a
        return a / mx

    o = normalize(onset_env)
    z = normalize(offset_env)
    N = len(o)

    # State machine
    state = 'silence'
    segments = []
    cur_on = None

    for i in range(N):
        t = times[i]

        if state == 'silence':
            # Trigger onset if:
            # 1. Onset envelope exceeds high threshold, OR
            # 2. Onset envelope exceeds low threshold AND offset is low
            if o[i] >= onset_high or (o[i] >= onset_low and z[i] < offset_low):
                state = 'sounding'
                cur_on = t

        else:  # state == 'sounding'
            # Trigger offset if:
            # 1. Offset envelope exceeds high threshold, OR
            # 2. Offset envelope exceeds low threshold AND onset is low
            if z[i] >= offset_high or (z[i] >= offset_low and o[i] < onset_low):
                cur_off = t

                # Validate segment duration
                if cur_on is not None:
                    duration = cur_off - cur_on

                    # Enforce minimum duration
                    if duration >= min_note_len:
                        # Enforce maximum duration
                        if duration > max_note_len:
                            cur_off = cur_on + max_note_len

                        segments.append((cur_on, cur_off))

                # Reset state
                cur_on = None
                state = 'silence'

    # If still sounding at end of audio, close segment
    if state == 'sounding' and cur_on is not None:
        cur_off = times[-1]
        duration = cur_off - cur_on
        if duration >= min_note_len:
            segments.append((cur_on, cur_off))

    # Merge segments separated by tiny gaps
    if len(segments) == 0:
        return segments

    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        gap = seg[0] - prev[1]

        if gap <= merge_gap:
            # Merge with previous segment
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)

    return merged


def detect_onsets_offsets(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    onset_high: float = 0.35,
    onset_low: float = 0.12,
    offset_high: float = 0.35,
    offset_low: float = 0.12,
    min_note_len: float = 0.05,
    max_note_len: float = 12.0,
    merge_gap: float = 0.03
) -> List[Tuple[float, float]]:
    """
    Main entry point for onset/offset detection.

    Detects note segments in audio using multi-band spectral flux
    analysis with hysteresis thresholding.

    Args:
        y: Audio time series
        sr: Sample rate
        n_fft: FFT window size (default: 2048)
        hop_length: Hop length in samples (default: 512, matches CQT)
        onset_high: High threshold for onset trigger (0-1, default: 0.35)
        onset_low: Low threshold for onset sustain (0-1, default: 0.12)
        offset_high: High threshold for offset trigger (0-1, default: 0.35)
        offset_low: Low threshold for offset sustain (0-1, default: 0.12)
        min_note_len: Minimum note duration in seconds (default: 0.05)
        max_note_len: Maximum note duration in seconds (default: 12.0)
        merge_gap: Merge segments separated by < this gap (default: 0.03)

    Returns:
        segments: List of (onset_time, offset_time) tuples in seconds

    Example:
        >>> y, sr = librosa.load('humming.wav', sr=16000)
        >>> segments = detect_onsets_offsets(y, sr)
        >>> for onset, offset in segments:
        ...     print(f"Note: {onset:.3f}s - {offset:.3f}s (dur: {offset-onset:.3f}s)")
    """
    # Compute onset and offset envelopes
    onset_env, offset_env, times = compute_onset_offset_envelopes(
        y, sr, n_fft=n_fft, hop_length=hop_length
    )

    # Detect segments using hysteresis
    segments = hysteresis_segment_detection(
        onset_env, offset_env, times,
        onset_high=onset_high,
        onset_low=onset_low,
        offset_high=offset_high,
        offset_low=offset_low,
        min_note_len=min_note_len,
        max_note_len=max_note_len,
        merge_gap=merge_gap
    )

    return segments


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onset_offset_detector.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"Loading audio: {audio_path}")

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio loaded: {len(y)/sr:.2f}s @ {sr}Hz")

    print("\nDetecting onsets/offsets...")
    segments = detect_onsets_offsets(
        y, sr,
        n_fft=2048,
        hop_length=512,
        onset_high=0.35,
        onset_low=0.10,
        offset_high=0.30,
        offset_low=0.08,
        min_note_len=0.05,
        merge_gap=0.02
    )

    print(f"\nDetected {len(segments)} segments:\n")
    for i, (onset, offset) in enumerate(segments, 1):
        duration = offset - onset
        print(f"Segment {i:3d}: onset {onset:6.3f}s  offset {offset:6.3f}s  duration {duration:6.3f}s")

    total_duration = sum(off - on for on, off in segments)
    print(f"\nTotal sounding time: {total_duration:.2f}s ({total_duration/len(y)*sr*100:.1f}% of audio)")
