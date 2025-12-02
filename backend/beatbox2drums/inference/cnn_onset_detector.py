#!/usr/bin/env python3
"""
CNN-based onset detector for beatbox audio.

Uses trained CNN to detect drum onsets with post-processing.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Optional
import os

# Optimize TensorFlow for CPU inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
import tensorflow as tf

# Enable CPU threading for TensorFlow
num_threads = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


class CNNOnsetDetector:
    """
    CNN-based onset detector with post-processing.

    Trained specifically on beatbox audio for high accuracy.
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 2048,
        hop_length: int = 441,
        window_frames: int = 12,
        fmax: float = 8000.0,

        # Post-processing parameters
        onset_threshold: float = 0.5,
        peak_delta: float = 0.05,  # 50ms NMS window (non-maximum suppression)
        use_hpss: bool = True,
        verbose: bool = False
    ):
        """
        Initialize CNN onset detector.

        Args:
            model_path: Path to trained CNN model (.h5 file)
            sample_rate: Audio sample rate (must match training)
            n_mels: Number of mel bands (must match training)
            n_fft: FFT window size
            hop_length: Hop length for spectrogram
            window_frames: Window size in frames (must match training)
            fmax: Maximum frequency for mel spectrogram
            onset_threshold: Probability threshold for onset detection
            peak_delta: Minimum time between detected peaks (seconds)
            use_hpss: Use harmonic-percussive separation
            verbose: Print debug information
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_frames = window_frames
        self.fmax = fmax

        self.onset_threshold = onset_threshold
        self.peak_delta = peak_delta
        self.use_hpss = use_hpss
        self.verbose = verbose

        # Load CNN model
        self.model = tf.keras.models.load_model(model_path)

        if self.verbose:
            print(f"[CNNOnsetDetector] Loaded model from {model_path}")
            self.model.summary()

    def extract_mel_spectrogram(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio signal (mono)

        Returns:
            mel_db: Mel spectrogram in dB [n_mels, n_frames]
            times: Time stamps for each frame [n_frames]
        """
        # Harmonic-percussive separation
        if self.use_hpss:
            _, audio_percussive = librosa.effects.hpss(audio, margin=(1.0, 5.0))
        else:
            audio_percussive = audio

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_percussive,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmax=self.fmax
        )

        # Convert to dB
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Get time stamps
        times = librosa.frames_to_time(
            np.arange(mel_db.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        return mel_db, times

    def create_sliding_windows(
        self,
        mel_spec: np.ndarray
    ) -> np.ndarray:
        """
        Create sliding windows over mel spectrogram.

        Args:
            mel_spec: Mel spectrogram [n_mels, n_frames]

        Returns:
            windows: Array of shape [n_windows, n_mels, window_frames, 1]
        """
        n_mels, n_frames = mel_spec.shape
        n_windows = n_frames - self.window_frames + 1

        windows = np.zeros((n_windows, n_mels, self.window_frames, 1))

        for i in range(n_windows):
            window = mel_spec[:, i:i + self.window_frames]
            windows[i, :, :, 0] = window

        return windows

    def predict_onset_probabilities(
        self,
        windows: np.ndarray
    ) -> np.ndarray:
        """
        Predict onset probabilities for each window.

        Args:
            windows: Array of spectrogram windows [n_windows, n_mels, window_frames, 1]

        Returns:
            probabilities: Onset probabilities [n_windows]
        """
        # Predict in batches
        predictions = self.model.predict(windows, batch_size=256, verbose=0)

        # Extract onset class probability (class 1)
        onset_probs = predictions[:, 1]

        return onset_probs

    def post_process_onsets(
        self,
        onset_probs: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Post-process onset probabilities to get final onset times.

        Uses Non-Maximum Suppression (NMS) with a time window to merge
        duplicate detections and keep only the highest probability peak
        within each window.

        Args:
            onset_probs: Onset probabilities for each frame [n_frames]
            times: Time stamps for each frame [n_frames]

        Returns:
            onset_times: Final detected onset times [n_onsets]
        """
        # Apply threshold
        above_threshold = onset_probs >= self.onset_threshold

        if not np.any(above_threshold):
            return np.array([])

        # Find all peaks above threshold
        peaks = []
        for i in range(len(onset_probs)):
            if above_threshold[i]:
                # Check if this is a local maximum
                is_peak = True

                # Check left neighbor
                if i > 0 and onset_probs[i] <= onset_probs[i - 1]:
                    is_peak = False

                # Check right neighbor
                if i < len(onset_probs) - 1 and onset_probs[i] <= onset_probs[i + 1]:
                    is_peak = False

                if is_peak:
                    current_time = times[i + self.window_frames // 2]  # Center of window
                    peaks.append({
                        'time': current_time,
                        'prob': onset_probs[i],
                        'index': i
                    })

        if len(peaks) == 0:
            return np.array([])

        # Sort peaks by probability (highest first)
        peaks = sorted(peaks, key=lambda x: x['prob'], reverse=True)

        # Non-maximum suppression: keep highest probability peaks,
        # suppress nearby peaks within peak_delta window
        selected_peaks = []
        suppressed = set()

        for peak in peaks:
            if peak['index'] in suppressed:
                continue

            # Add this peak
            selected_peaks.append(peak['time'])

            # Suppress all nearby peaks within NMS window
            for other_peak in peaks:
                if other_peak['index'] in suppressed:
                    continue

                time_diff = abs(other_peak['time'] - peak['time'])
                if time_diff < self.peak_delta and other_peak['index'] != peak['index']:
                    suppressed.add(other_peak['index'])

        # Sort by time for output
        selected_peaks = sorted(selected_peaks)

        return np.array(selected_peaks)

    def detect_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect onsets from audio array.

        Args:
            audio: Audio signal (mono, sampled at self.sample_rate)

        Returns:
            onset_times: Array of onset times in seconds
        """
        if self.verbose:
            print(f"[CNNOnsetDetector] Detecting onsets...")
            print(f"  Audio duration: {len(audio) / self.sample_rate:.2f}s")

        # Extract mel spectrogram
        mel_spec, times = self.extract_mel_spectrogram(audio)

        if self.verbose:
            print(f"  Mel spectrogram shape: {mel_spec.shape}")

        # Create sliding windows
        windows = self.create_sliding_windows(mel_spec)

        if self.verbose:
            print(f"  Created {len(windows)} windows")

        # Predict onset probabilities
        onset_probs = self.predict_onset_probabilities(windows)

        if self.verbose:
            print(f"  Max onset probability: {np.max(onset_probs):.3f}")
            print(f"  Mean onset probability: {np.mean(onset_probs):.3f}")

        # Post-process to get final onset times
        onset_times = self.post_process_onsets(onset_probs, times)

        if self.verbose:
            print(f"  Detected {len(onset_times)} onsets")

        return onset_times

    def detect_from_file(self, audio_path: str) -> np.ndarray:
        """
        Detect onsets from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            onset_times: Array of onset times in seconds
        """
        if self.verbose:
            print(f"[CNNOnsetDetector] Loading {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Detect onsets
        return self.detect_from_audio(audio)

    def compute_detection_stats(
        self,
        detected_onsets: np.ndarray,
        ground_truth_onsets: np.ndarray,
        tolerance: float = 0.05
    ) -> dict:
        """
        Compute onset detection statistics vs ground truth.

        Args:
            detected_onsets: Detected onset times
            ground_truth_onsets: Ground truth onset times
            tolerance: Time tolerance for matching (seconds)

        Returns:
            Dictionary with precision, recall, F1 score, and mean error
        """
        if len(ground_truth_onsets) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': len(detected_onsets),
                'false_negatives': 0,
                'mean_error': 0.0
            }

        if len(detected_onsets) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(ground_truth_onsets),
                'mean_error': 0.0
            }

        # Find matches
        matched_gt = set()
        matched_det = set()
        errors = []

        for i, gt_time in enumerate(ground_truth_onsets):
            # Find closest detected onset
            diffs = np.abs(detected_onsets - gt_time)
            min_idx = np.argmin(diffs)
            min_diff = diffs[min_idx]

            if min_diff <= tolerance and min_idx not in matched_det:
                matched_gt.add(i)
                matched_det.add(min_idx)
                errors.append(min_diff)

        true_positives = len(matched_gt)
        false_positives = len(detected_onsets) - len(matched_det)
        false_negatives = len(ground_truth_onsets) - len(matched_gt)

        precision = true_positives / len(detected_onsets) if len(detected_onsets) > 0 else 0.0
        recall = true_positives / len(ground_truth_onsets) if len(ground_truth_onsets) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_error = np.mean(errors) if errors else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'mean_error': mean_error
        }


def create_default_cnn_detector(model_path: str) -> CNNOnsetDetector:
    """
    Create CNN onset detector with default parameters.

    Args:
        model_path: Path to trained CNN model

    Returns:
        Configured CNNOnsetDetector
    """
    return CNNOnsetDetector(
        model_path=model_path,
        sample_rate=16000,
        n_mels=80,
        window_frames=12,
        onset_threshold=0.5,
        peak_delta=0.020,  # 20ms minimum between onsets
        use_hpss=True,
        verbose=False
    )


if __name__ == '__main__':
    # Test CNN onset detector
    import sys

    if len(sys.argv) < 3:
        print("Usage: python cnn_onset_detector.py <model_path> <audio_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    audio_file = sys.argv[2]

    print(f"Testing CNN Onset Detector")
    print(f"Model: {model_path}")
    print(f"Audio: {audio_file}")
    print()

    detector = create_default_cnn_detector(model_path)
    detector.verbose = True

    onsets = detector.detect_from_file(audio_file)

    print(f"\nDetected {len(onsets)} onsets:")
    for i, onset_time in enumerate(onsets):
        print(f"  {i+1}. {onset_time:.3f}s")
