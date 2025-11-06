import librosa
import numpy as np
import io
from typing import Tuple

class AudioProcessor:
    """
    Handles audio preprocessing for different model types.
    Converts raw audio bytes into features suitable for ML models.
    """
    
    def __init__(self, target_sr: int = 16000):
        """
        Args:
            target_sr: Target sample rate for audio processing (default 16kHz)
        """
        self.target_sr = target_sr
    
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes and return audio array and sample rate.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load audio from bytes using librosa
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
        return audio, int(sr)  # Cast to int since sample rates are always integers
    
    def preprocess_for_hum2melody(self, audio_bytes: bytes) -> dict:
        """
        Preprocess audio for melody extraction from humming.
        
        Focus on pitch detection and melody contour extraction.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Dictionary containing processed features
        """
        # Load and normalize audio
        audio, sr = self.load_audio(audio_bytes)
        
        # Apply noise reduction (simple approach)
        audio = self._reduce_noise(audio)
        
        # Extract mel spectrogram (good for pitch-based features)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract pitch/fundamental frequency
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=sr,
            fmin=80,  # Typical human voice range
            fmax=400
        )
        
        # Get onset times (note starts)
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            units='frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        return {
            "mel_spectrogram": mel_spec_db,
            "pitches": pitches,
            "magnitudes": magnitudes,
            "onset_times": onset_times.tolist(),
            "duration": librosa.get_duration(y=audio, sr=sr),
            "sample_rate": sr,
            "audio_shape": audio.shape
        }
    
    def preprocess_for_beatbox(self, audio_bytes: bytes) -> dict:
        """
        Preprocess audio for drum pattern extraction from beatboxing.
        
        Focus on onset detection and percussive elements.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Dictionary containing processed features
        """
        # Load audio
        audio, sr = self.load_audio(audio_bytes)
        
        # Separate harmonic and percussive components
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
        
        # Extract onset strength (percussive energy)
        onset_env = librosa.onset.onset_strength(
            y=audio_percussive,
            sr=sr
        )
        
        # Detect onsets (drum hits)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units='frames',
            backtrack=True  # Improve onset accuracy
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Get spectral features for kick/snare classification
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_percussive,
            sr=sr
        )
        
        # MFCC features (useful for drum sound classification)
        mfcc = librosa.feature.mfcc(
            y=audio_percussive,
            sr=sr,
            n_mfcc=13
        )
        
        return {
            "onset_times": onset_times.tolist(),
            "onset_strength": onset_env.tolist(),
            "tempo": float(tempo),
            "spectral_centroid": spectral_centroid.tolist(),
            "mfcc": mfcc.tolist(),
            "duration": librosa.get_duration(y=audio, sr=sr),
            "sample_rate": sr,
            "audio_shape": audio.shape
        }
    
    def _reduce_noise(self, audio: np.ndarray, noise_threshold: float = 0.02) -> np.ndarray:
        """
        Simple noise reduction using amplitude thresholding.
        
        Args:
            audio: Input audio array
            noise_threshold: Amplitude threshold below which to suppress
            
        Returns:
            Noise-reduced audio array
        """
        # Simple noise gate
        audio_normalized = audio / np.max(np.abs(audio))
        audio_normalized[np.abs(audio_normalized) < noise_threshold] = 0
        return audio_normalized
    
    def save_audio(self, audio: np.ndarray, sr: int, output_path: str):
        """
        Save processed audio to file.
        
        Args:
            audio: Audio array
            sr: Sample rate
            output_path: Path to save audio file
        """
        import soundfile as sf
        sf.write(output_path, audio, sr)