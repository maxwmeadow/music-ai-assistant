"""
Spectral feature extraction for drum classification.
"""

import numpy as np
import librosa


def extract_spectral_features(y, sr=16000):
    """
    Extract 8 key spectral features from audio signal.

    Features extracted:
    1-2. Spectral bandwidth (mean, std)
    3-4. Spectral rolloff (mean, std)
    5-6. Spectral centroid (mean, std)
    7-8. Spectral flatness (mean, std)

    Args:
        y: Audio time series (numpy array)
        sr: Sample rate (default: 16000)

    Returns:
        numpy array of shape (8,) containing the features
    """
    features = []

    # Compute STFT
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=441))

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr, hop_length=441)[0]
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    # Spectral rolloff (85th percentile)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, hop_length=441, roll_percent=0.85)[0]
    features.extend([np.mean(rolloff), np.std(rolloff)])

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=441)[0]
    features.extend([np.mean(centroid), np.std(centroid)])

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(S=S, hop_length=441)[0]
    features.extend([np.mean(flatness), np.std(flatness)])

    return np.array(features, dtype=np.float32)
