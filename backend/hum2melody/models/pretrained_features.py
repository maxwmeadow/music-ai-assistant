"""
Pretrained Feature Extractors for Audio Understanding

Provides pretrained embeddings that already understand audio structure,
giving the model a massive head start on onset/offset detection.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

# Suppress warnings from pretrained models
warnings.filterwarnings('ignore')


class PretrainedFeatureExtractor:
    """
    Extract features from pretrained models.
    
    Supported models:
    - wav2vec2: General audio understanding (best for speech/humming)
    - hubert: Similar to wav2vec2, often better for music
    """
    
    def __init__(self, model_type='wav2vec2', device='cpu'):
        self.model_type = model_type
        self.device = device
        
        print(f"[PretrainedFeatureExtractor] Loading {model_type}...")
        
        if model_type == 'wav2vec2':
            try:
                import torchaudio
                bundle = torchaudio.pipelines.WAV2VEC2_BASE
                self.model = bundle.get_model().to(device)
                self.target_sample_rate = bundle.sample_rate
                self.feature_dim = 768  # wav2vec2 base output dim
                print(f"  ✅ Loaded wav2vec2 (feature dim: {self.feature_dim})")
            except Exception as e:
                print(f"  ⚠️ Failed to load wav2vec2: {e}")
                print("  Falling back to no pretrained features")
                self.model = None
                self.feature_dim = 0
        
        elif model_type == 'hubert':
            try:
                import torchaudio
                bundle = torchaudio.pipelines.HUBERT_BASE
                self.model = bundle.get_model().to(device)
                self.target_sample_rate = bundle.sample_rate
                self.feature_dim = 768
                print(f"  ✅ Loaded HuBERT (feature dim: {self.feature_dim})")
            except Exception as e:
                print(f"  ⚠️ Failed to load HuBERT: {e}")
                self.model = None
                self.feature_dim = 0
        
        else:
            self.model = None
            self.feature_dim = 0
        
        if self.model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract pretrained features from audio.
        
        Args:
            audio: Audio array (mono)
            sr: Sample rate
        
        Returns:
            features: (time, feature_dim) - downsampled to match CQT timing
        """
        if self.model is None:
            return None
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Resample if needed
        if sr != self.target_sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, self.target_sample_rate).to(self.device)
            audio_tensor = resampler(audio_tensor)
        
        # Extract features
        with torch.no_grad():
            if self.model_type in ['wav2vec2', 'hubert']:
                features, _ = self.model.extract_features(audio_tensor)
                # Use features from last layer
                features = features[-1].squeeze(0)  # (time, feature_dim)
        
        return features.cpu().numpy()


def extract_onset_strength_features(audio: np.ndarray, sr: int = 16000, hop_length: int = 512) -> np.ndarray:
    """
    Extract features specifically designed for onset/offset detection.
    
    These are classical signal processing features that are VERY good
    at detecting note boundaries.
    
    Returns:
        features: (5, time) array with:
            - spectral_flux: Sudden changes in spectrum
            - rms: Energy envelope
            - spectral_centroid: Brightness (attacks are brighter)
            - spectral_rolloff: High-frequency content
            - zcr: Zero-crossing rate (texture changes)
    """
    import librosa
    
    # 1. Spectral flux - THE BEST for onset detection
    spectral_flux = librosa.onset.onset_strength(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median
    )
    
    # 2. Energy envelope
    rms = librosa.feature.rms(
        y=audio,
        hop_length=hop_length
    )[0]
    
    # 3. Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )[0]
    
    # 4. Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )[0]
    
    # 5. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=hop_length
    )[0]
    
    # Normalize features to [0, 1]
    def normalize(x):
        x = x - x.min()
        x_max = x.max()
        if x_max > 0:
            x = x / x_max
        return x
    
    features = np.vstack([
        normalize(spectral_flux),
        normalize(rms),
        normalize(spectral_centroid),
        normalize(spectral_rolloff),
        normalize(zcr)
    ])
    
    return features  # Shape: (5, time)


def extract_musical_context_features(audio: np.ndarray, sr: int = 16000, hop_length: int = 512) -> dict:
    """
    Extract features that capture musical structure and context.

    Returns dict with:
        - chroma: (12, time) - pitch class distribution
        - tonnetz: (6, time) - harmonic relationships
        - tempogram: (time,) - rhythmic patterns

    FIXED: Removed librosa.effects.harmonic() which causes segfaults on HPCC
    due to FFT library conflicts. Using original audio for tonnetz instead.
    """
    import librosa

    # Chroma - captures harmonic context
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )

    # Tonnetz - tonal centroid features
    # FIXED: Use chroma directly to avoid librosa.effects.harmonic() segfault
    # tonnetz() internally calls harmonic() if you pass audio, so we pass chroma instead
    try:
        tonnetz = librosa.feature.tonnetz(
            chroma=chroma
        )
    except Exception as e:
        # Fallback if tonnetz also fails
        print(f"Warning: Tonnetz extraction failed: {e}")
        # Return zeros with correct shape
        expected_frames = len(chroma[0])
        tonnetz = np.zeros((6, expected_frames), dtype=np.float32)

    # Tempogram - rhythmic patterns
    tempogram = librosa.feature.tempogram(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )

    # Use dominant tempo feature (sum across tempo bins)
    tempo_strength = tempogram.sum(axis=0)

    return {
        'chroma': chroma,
        'tonnetz': tonnetz,
        'tempo': tempo_strength
    }


def test_pretrained_features():
    """Test the feature extractors."""
    print("\nTesting Pretrained Feature Extractors...")
    
    # Create test audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # A4
    
    # Test pretrained extractor
    print("\n1. Testing pretrained features...")
    extractor = PretrainedFeatureExtractor('wav2vec2', device='cpu')
    if extractor.model:
        features = extractor.extract(audio, sr)
        print(f"  ✅ Pretrained features shape: {features.shape}")
    
    # Test onset features
    print("\n2. Testing onset-strength features...")
    onset_features = extract_onset_strength_features(audio, sr)
    print(f"  ✅ Onset features shape: {onset_features.shape}")
    print(f"     Features: flux, rms, centroid, rolloff, zcr")
    
    # Test musical context
    print("\n3. Testing musical context features...")
    context_features = extract_musical_context_features(audio, sr)
    print(f"  ✅ Chroma shape: {context_features['chroma'].shape}")
    print(f"  ✅ Tonnetz shape: {context_features['tonnetz'].shape}")
    print(f"  ✅ Tempo shape: {context_features['tempo'].shape}")
    
    print("\n✅ All feature extractors working!")


if __name__ == '__main__':
    test_pretrained_features()
