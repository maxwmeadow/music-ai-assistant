#!/usr/bin/env python3
"""
Complete Beatbox-to-Drums Inference Pipeline

Combines CNN onset detection and CNN-based drum classification.
Raw audio -> Onset CNN -> Classifier CNN -> Drum hits with types
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
import sys

# Import components
from .cnn_onset_detector import CNNOnsetDetector
from ..models.drum_classifier import DrumClassifierCNN
from ..models.drum_classifier_multi_input import DrumClassifierCNN_MultiInput
from .feature_extraction import extract_spectral_features


class DrumHit:
    """Represents a single drum hit with type, time, and confidence."""

    def __init__(self, drum_type: str, time: float, confidence: float, probabilities: dict = None):
        self.drum_type = drum_type
        self.time = time
        self.confidence = confidence
        self.probabilities = probabilities or {}  # All class probabilities: {'kick': 0.1, 'snare': 0.8, 'hihat': 0.1}

    def __repr__(self):
        return f"DrumHit({self.drum_type}, t={self.time:.3f}s, conf={self.confidence:.3f})"

    def to_dict(self):
        result = {
            'drum_type': self.drum_type,
            'time': self.time,
            'confidence': self.confidence
        }
        if self.probabilities:
            result['probabilities'] = self.probabilities
        return result


class Beatbox2DrumsPipeline:
    """
    End-to-end pipeline: Audio -> Onset Detection -> Drum Classification

    This pipeline uses:
    1. CNN-based onset detector to find potential drum hits
    2. CNN-based classifier to determine the drum type for each onset
    3. Optional confidence filtering to remove uncertain predictions
    """

    # Class names matching the trained classifier
    CLASS_NAMES = ['kick', 'snare', 'hihat']
    CLASS_TO_IDX = {'kick': 0, 'snare': 1, 'hihat': 2}
    IDX_TO_CLASS = {0: 'kick', 1: 'snare', 2: 'hihat'}

    # MIDI mapping for compatibility
    DRUM_MIDI = {
        'kick': 36,   # Bass Drum 1 (C2)
        'snare': 38,  # Acoustic Snare (D2)
        'hihat': 42   # Closed Hi-Hat (F#2)
    }

    def __init__(
        self,
        onset_checkpoint_path: str,
        classifier_checkpoint_path: str,
        onset_threshold: float = 0.5,
        onset_peak_delta: float = 0.05,  # 50ms NMS window
        classifier_confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        use_multi_input: bool = True,  # Use new multi-input model by default
        feature_norm_path: Optional[str] = None
    ):
        """
        Initialize the pipeline.

        Args:
            onset_checkpoint_path: Path to trained CNN onset detector checkpoint
            classifier_checkpoint_path: Path to trained CNN classifier checkpoint
            onset_threshold: Minimum probability for onset detection
            onset_peak_delta: NMS window size in seconds (default 50ms)
            classifier_confidence_threshold: Minimum confidence for classification
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        print("="*70)
        print("Initializing Beatbox2Drums Pipeline")
        print("="*70)

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # Audio parameters (must match training)
        self.sample_rate = 16000
        self.n_mels_classifier = 128  # Classifier uses 128 mel bands
        self.n_mels_onset = 80        # Onset detector uses 80 mel bands
        self.hop_length = 441         # ~27.55ms frames at 16kHz
        self.window_frames = 12       # Classifier window size

        # Thresholds
        self.onset_threshold = onset_threshold
        self.onset_peak_delta = onset_peak_delta
        self.classifier_confidence_threshold = classifier_confidence_threshold
        self.use_multi_input = use_multi_input

        # Feature normalization stats (for multi-input model)
        self.feature_mean = None
        self.feature_std = None

        # Load onset detector
        print("\nLoading CNN Onset Detector...")
        print(f"  Checkpoint: {onset_checkpoint_path}")
        self.onset_detector = CNNOnsetDetector(
            model_path=onset_checkpoint_path,
            onset_threshold=onset_threshold,
            peak_delta=onset_peak_delta,
            verbose=False
        )
        print("  ✓ Onset detector loaded")

        # Load classifier
        print("\nLoading CNN Drum Classifier...")
        print(f"  Checkpoint: {classifier_checkpoint_path}")
        print(f"  Multi-input mode: {use_multi_input}")
        self.classifier = self._load_classifier(classifier_checkpoint_path, use_multi_input)
        print("  ✓ Classifier loaded")

        # Load feature normalization stats if using multi-input model
        if use_multi_input:
            if feature_norm_path is None:
                # Default path: same directory as classifier checkpoint
                checkpoint_dir = Path(classifier_checkpoint_path).parent
                feature_norm_path = checkpoint_dir / 'feature_normalization.npz'

            print(f"\nLoading feature normalization stats...")
            print(f"  Path: {feature_norm_path}")

            if Path(feature_norm_path).exists():
                norm_data = np.load(feature_norm_path)
                self.feature_mean = norm_data['mean']
                self.feature_std = norm_data['std'] + 1e-8  # Add epsilon for stability
                print("  ✓ Feature normalization loaded")
            else:
                print(f"  WARNING: Feature normalization file not found at {feature_norm_path}")
                print("  Using zero mean and unit variance (this may affect performance)")
                self.feature_mean = np.zeros(8)
                self.feature_std = np.ones(8)

        print("\n" + "="*70)
        print("Pipeline ready!")
        print("="*70)
        print()

    def _load_classifier(self, checkpoint_path: str, use_multi_input: bool = True):
        """Load the drum classifier from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if use_multi_input:
            # Create multi-input model (3 classes: kick, snare, hihat)
            model = DrumClassifierCNN_MultiInput(
                num_classes=3,
                num_features=8,
                dropout=0.3
            )
        else:
            # Create standard CNN model
            model = DrumClassifierCNN(num_classes=3, dropout=0.3)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_acc' in checkpoint:
                print(f"  Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def predict(
        self,
        audio_path: str,
        return_details: bool = False
    ) -> List[DrumHit]:
        """
        Predict drum hits from audio file.

        Args:
            audio_path: Path to audio file
            return_details: If True, return additional info (onset probs, etc.)

        Returns:
            List of DrumHit objects
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        return self.predict_from_audio(y, return_details=return_details)

    def predict_from_audio(
        self,
        audio: np.ndarray,
        return_details: bool = False
    ) -> List[DrumHit]:
        """
        Predict drum hits from audio array.

        Args:
            audio: Audio waveform (mono, 16kHz)
            return_details: If True, return dict with additional info

        Returns:
            List of DrumHit objects (or dict if return_details=True)
        """
        # Step 1: Detect onsets using CNN
        onset_times = self.onset_detector.detect_from_audio(audio)

        if len(onset_times) == 0:
            if return_details:
                return {'drum_hits': [], 'onset_times': [], 'rejected_count': 0}
            return []

        # Step 2: Extract mel spectrogram for classification
        mel_spec = self._extract_classifier_mel_spectrogram(audio)
        times = librosa.frames_to_time(
            np.arange(mel_spec.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # Step 3: Classify each onset
        drum_hits = []
        rejected_count = 0

        for onset_time in onset_times:
            # Extract mel spectrogram window at onset
            mel_window = self._extract_window_at_time(
                mel_spec, times, onset_time, self.window_frames
            )

            if mel_window is None:
                rejected_count += 1
                continue

            # Extract spectral features if using multi-input model
            if self.use_multi_input:
                # Extract audio window for spectral features (100ms after onset)
                start_sample = int(onset_time * self.sample_rate)
                end_sample = start_sample + int(0.1 * self.sample_rate)  # 100ms window

                if start_sample < 0 or end_sample > len(audio):
                    rejected_count += 1
                    continue

                audio_window = audio[start_sample:end_sample]
                spectral_features = extract_spectral_features(audio_window, sr=self.sample_rate)

                # Classify the window with features
                drum_type, confidence, probabilities = self._classify_window(mel_window, spectral_features)
            else:
                # Classify with mel spectrogram only
                drum_type, confidence, probabilities = self._classify_window(mel_window)

            # Apply confidence threshold
            if confidence >= self.classifier_confidence_threshold:
                drum_hits.append(DrumHit(drum_type, onset_time, confidence, probabilities))
            else:
                rejected_count += 1

        # Sort by time
        drum_hits.sort(key=lambda h: h.time)

        if return_details:
            return {
                'drum_hits': drum_hits,
                'onset_times': onset_times,
                'rejected_count': rejected_count,
                'total_onsets': len(onset_times),
                'confidence_threshold': self.classifier_confidence_threshold
            }

        return drum_hits

    def _extract_classifier_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram for the classifier (128 bands).

        Returns:
            mel_spec_db: (128, n_frames) in dB scale
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels_classifier,
            n_fft=2048,
            hop_length=self.hop_length,
            fmax=8000.0
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def _extract_window_at_time(
        self,
        mel_spec: np.ndarray,
        times: np.ndarray,
        target_time: float,
        window_frames: int
    ) -> Optional[np.ndarray]:
        """
        Extract a window starting at target_time (100ms = 12 frames at hop_length=441).

        Args:
            mel_spec: (n_mels, n_frames)
            times: Time for each frame
            target_time: Target time in seconds
            window_frames: Window size in frames (12 frames = 200ms)

        Returns:
            window: (n_mels, window_frames) or None if insufficient context
        """
        # Find closest frame to onset time
        frame_idx = np.argmin(np.abs(times - target_time))

        # Extract window starting at onset (100ms after = 12 frames)
        # Note: At hop_length=441, 12 frames = ~200ms
        start_frame = frame_idx
        end_frame = start_frame + window_frames

        # Check boundaries
        if start_frame < 0 or end_frame > mel_spec.shape[1]:
            return None

        window = mel_spec[:, start_frame:end_frame]
        return window

    def _classify_window(self, mel_window: np.ndarray, spectral_features: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Classify a mel spectrogram window with optional spectral features.

        Args:
            mel_window: (n_mels, window_frames) mel spectrogram
            spectral_features: (8,) spectral features (for multi-input model)

        Returns:
            (drum_type, confidence) tuple
        """
        # Convert mel window to tensor: (1, 1, n_mels, window_frames)
        mel_tensor = torch.FloatTensor(mel_window).unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            if self.use_multi_input and spectral_features is not None:
                # Normalize spectral features
                features_normalized = (spectral_features - self.feature_mean) / self.feature_std
                features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0)
                features_tensor = features_tensor.to(self.device)

                # Multi-input inference
                logits = self.classifier(mel_tensor, features_tensor)  # (1, 3)
            else:
                # Single-input inference
                logits = self.classifier(mel_tensor)  # (1, 3)

            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Get prediction
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        drum_type = self.IDX_TO_CLASS[pred_idx]

        # Build probabilities dict for all classes
        probabilities = {
            'kick': float(probs[0]),
            'snare': float(probs[1]),
            'hihat': float(probs[2])
        }

        return drum_type, confidence, probabilities

    def get_statistics(self, drum_hits: List[DrumHit]) -> Dict:
        """Get statistics about detected drum hits."""
        stats = {
            'total': len(drum_hits),
            'by_type': {drum_type: 0 for drum_type in self.CLASS_NAMES},
            'avg_confidence': 0.0,
            'confidence_by_type': {drum_type: [] for drum_type in self.CLASS_NAMES}
        }

        if len(drum_hits) == 0:
            return stats

        for hit in drum_hits:
            stats['by_type'][hit.drum_type] += 1
            stats['confidence_by_type'][hit.drum_type].append(hit.confidence)

        stats['avg_confidence'] = np.mean([h.confidence for h in drum_hits])

        # Calculate average confidence per type
        for drum_type in self.CLASS_NAMES:
            if len(stats['confidence_by_type'][drum_type]) > 0:
                stats['confidence_by_type'][drum_type] = np.mean(
                    stats['confidence_by_type'][drum_type]
                )
            else:
                stats['confidence_by_type'][drum_type] = 0.0

        return stats


def test_pipeline():
    """Test the pipeline with a sample file."""
    import sys

    # Check paths
    onset_checkpoint = Path("beatbox2drums/cnn_onset_checkpoints/best_model.pth")
    classifier_checkpoint = Path("beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth")

    if not onset_checkpoint.exists():
        print(f"Onset checkpoint not found: {onset_checkpoint}")
        sys.exit(1)

    if not classifier_checkpoint.exists():
        print(f"Classifier checkpoint not found: {classifier_checkpoint}")
        sys.exit(1)

    # Create pipeline
    pipeline = Beatbox2DrumsPipeline(
        onset_checkpoint_path=str(onset_checkpoint),
        classifier_checkpoint_path=str(classifier_checkpoint),
        onset_threshold=0.5,
        onset_peak_delta=0.05,
        classifier_confidence_threshold=0.3
    )

    # Find a test file
    test_files = list(Path("beatbox2drums/dataset/combined/audio").glob("*.wav"))[:1]

    if not test_files:
        print("No test files found")
        sys.exit(1)

    print(f"\nTesting with: {test_files[0].name}")
    print("-" * 70)

    # Run prediction
    results = pipeline.predict(str(test_files[0]), return_details=True)

    # Show results
    print(f"\nResults:")
    print(f"  Total onsets detected: {results['total_onsets']}")
    print(f"  Drum hits after filtering: {len(results['drum_hits'])}")
    print(f"  Rejected (low confidence): {results['rejected_count']}")
    print()

    # Show statistics
    stats = pipeline.get_statistics(results['drum_hits'])
    print("Statistics:")
    for drum_type in pipeline.CLASS_NAMES:
        count = stats['by_type'][drum_type]
        avg_conf = stats['confidence_by_type'][drum_type]
        print(f"  {drum_type}: {count} hits (avg conf: {avg_conf:.3f})")
    print(f"  Overall avg confidence: {stats['avg_confidence']:.3f}")
    print()

    # Show first few hits
    print("First 10 drum hits:")
    for hit in results['drum_hits'][:10]:
        print(f"  {hit}")


if __name__ == '__main__':
    test_pipeline()
