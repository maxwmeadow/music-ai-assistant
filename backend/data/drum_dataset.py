"""
Drum Dataset for BeatBox2Drums Training

Loads audio files and corresponding drum hit labels, converts to mel spectrograms
and frame-level multi-hot targets for training.

Label Format (JSON from Ayaan's auto-labeling):
{
    'audio_path': 'drums/drum_001.wav',
    'duration': 30.0,
    'drum_hits': {
        'kick': [{'time': 0.0, 'velocity': 0.8}, {'time': 0.5, 'velocity': 0.7}],
        'snare': [{'time': 0.25, 'velocity': 0.9}],
        'hihat': [{'time': 0.125, 'velocity': 0.6}]
    },
    'total_hits': 28
}
"""

import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings


class DrumDataset(Dataset):
    """
    PyTorch Dataset for drum detection and classification.
    
    Args:
        labels_path (str): Path to labels JSON file or manifest file listing label paths
        audio_root (str): Root directory for audio files (if paths in labels are relative)
        sample_rate (int): Target sample rate for audio (default: 16000)
        n_mels (int): Number of mel frequency bands (default: 128)
        target_frames (int): Target number of time frames (default: 500)
        hop_length (int): Hop length for STFT (default: 512)
        validate (bool): Validate labels on load (default: True)
    """
    
    # Drum type to index mapping
    DRUM_TYPES = {
        'kick': 0,
        'snare': 1,
        'hihat': 2,
        'silence': 3
    }
    
    def __init__(
        self,
        labels_path: str,
        audio_root: Optional[str] = None,
        sample_rate: int = 16000,
        n_mels: int = 128,
        target_frames: int = 500,
        hop_length: int = 512,
        validate: bool = True
    ):
        self.labels_path = Path(labels_path)
        self.audio_root = Path(audio_root) if audio_root else None
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.hop_length = hop_length
        self.num_drum_types = 4  # kick, snare, hihat, silence
        
        # Frame rate calculation
        # Sample rate: 16000 Hz, Hop length: 512 samples
        # Frame rate: 16000 / 512 = 31.25 frames/second
        self.frame_rate = sample_rate / hop_length
        
        # Load all labels
        self.samples = self._load_labels()
        
        if validate:
            self._validate_labels()
        
        print(f"[DrumDataset] Loaded {len(self.samples)} samples")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Mel bands: {n_mels}")
        print(f"  Target frames: {target_frames} (~{target_frames/self.frame_rate:.1f}s)")
        print(f"  Frame rate: {self.frame_rate:.2f} fps")
        print(f"  Drum types: {list(self.DRUM_TYPES.keys())}")
    
    def _load_labels(self) -> List[Dict]:
        """Load all label files."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        # Check if it's a manifest file (contains list of paths) or single JSON
        with open(self.labels_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # It's either a manifest of paths or a list of label dicts
            if len(data) > 0 and isinstance(data[0], str):
                # List of paths to label files
                samples = []
                for label_path in data:
                    with open(label_path, 'r') as f:
                        samples.append(json.load(f))
                return samples
            else:
                # List of label dicts
                return data
        else:
            # Single label dict
            return [data]
    
    def _validate_labels(self):
        """Validate label integrity."""
        valid_samples = []
        invalid_count = 0
        
        for idx, sample in enumerate(self.samples):
            try:
                # Check required fields
                assert 'audio_path' in sample, "Missing 'audio_path'"
                assert 'drum_hits' in sample, "Missing 'drum_hits'"
                
                drum_hits = sample['drum_hits']
                
                # Check that drum_hits has the expected structure
                for drum_type in ['kick', 'snare', 'hihat']:
                    assert drum_type in drum_hits, f"Missing drum type: {drum_type}"
                    assert isinstance(drum_hits[drum_type], list), \
                        f"drum_hits['{drum_type}'] must be a list"
                
                # Check minimum hit count (avoid empty samples)
                total_hits = sum(len(drum_hits[dt]) for dt in ['kick', 'snare', 'hihat'])
                assert total_hits >= 5, f"Too few hits: {total_hits} (need ≥5)"
                
                # Sanity check: not too many hits (likely labeling error)
                assert total_hits <= 1000, f"Too many hits: {total_hits} (max 1000)"
                
                # Check audio file exists
                audio_path = self._get_audio_path(sample['audio_path'])
                assert audio_path.exists(), f"Audio file not found: {audio_path}"
                
                valid_samples.append(sample)
                
            except AssertionError as e:
                invalid_count += 1
                warnings.warn(f"Invalid sample {idx}: {e}")
        
        self.samples = valid_samples
        
        if invalid_count > 0:
            print(f"[DrumDataset] Filtered out {invalid_count} invalid samples")
    
    def _get_audio_path(self, audio_path: str) -> Path:
        """Get absolute audio file path."""
        audio_path_obj = Path(audio_path)
        
        if audio_path_obj.is_absolute():
            return audio_path_obj
        elif self.audio_root:
            return self.audio_root / audio_path_obj
        else:
            return audio_path_obj
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            mel_tensor: Mel spectrogram, shape (1, target_frames, n_mels)
            target_tensor: Frame-level drum targets, shape (target_frames, 4)
        """
        sample = self.samples[idx]
        
        # Load and process audio
        audio_path = self._get_audio_path(sample['audio_path'])
        mel_spec = self._load_mel_spectrogram(audio_path)
        
        # Convert labels to frame-level targets
        target = self._create_frame_targets(
            labels=sample['drum_hits'],
            total_frames=mel_spec.shape[1]
        )
        
        # Ensure target matches mel_spec length
        if target.shape[0] != mel_spec.shape[1]:
            if target.shape[0] < mel_spec.shape[1]:
                padding = np.zeros((mel_spec.shape[1] - target.shape[0], self.num_drum_types))
                target = np.vstack([target, padding])
            else:
                target = target[:mel_spec.shape[1], :]
        
        # Convert to tensors
        mel_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0)  # (1, time, n_mels)
        target_tensor = torch.FloatTensor(target)  # (time, 4)
        
        return mel_tensor, target_tensor
    
    def _load_mel_spectrogram(self, audio_path: Path) -> np.ndarray:
        """
        Load audio and extract mel spectrogram.
        
        Returns:
            mel_spec: Shape (n_mels, target_frames)
        """
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True
        )
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=2048,
            fmin=20,   # Lower range for kick drums
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        # Pad or truncate to target_frames
        if mel_spec_normalized.shape[1] < self.target_frames:
            pad_width = self.target_frames - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(
                mel_spec_normalized,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )
        elif mel_spec_normalized.shape[1] > self.target_frames:
            mel_spec_normalized = mel_spec_normalized[:, :self.target_frames]
        
        return mel_spec_normalized
    
    def _create_frame_targets(
        self,
        labels: Dict[str, List[Dict]],
        total_frames: int
    ) -> np.ndarray:
        """
        Convert drum hit labels to frame-level multi-hot encoding.
        
        Args:
            labels: Drum hits dict with format:
                {
                    'kick': [{'time': 0.0, 'velocity': 0.8}, ...],
                    'snare': [{'time': 0.25, 'velocity': 0.9}, ...],
                    'hihat': [{'time': 0.125, 'velocity': 0.6}, ...]
                }
            total_frames: Total number of frames in the mel spectrogram
            
        Returns:
            targets: Shape (total_frames, 4) - multi-hot encoding
                Column 0: kick
                Column 1: snare
                Column 2: hihat
                Column 3: silence (set to 1 for frames with no hits)
        """
        # Initialize all frames as silence
        targets = np.zeros((total_frames, self.num_drum_types), dtype=np.float32)
        targets[:, 3] = 1.0  # All frames start as silence
        
        # Frame rate math:
        # Sample rate: 16000 Hz, Hop length: 512 samples
        # Frame rate: 16000 / 512 = 31.25 frames/second
        # For hit at time T seconds: frame_index = int(T * 31.25)
        
        # Process each drum type
        for drum_type in ['kick', 'snare', 'hihat']:
            drum_idx = self.DRUM_TYPES[drum_type]
            hits = labels.get(drum_type, [])
            
            for hit in hits:
                hit_time = hit['time']
                # velocity = hit['velocity']  # Could use this for weighted loss later
                
                # Convert time to frame index
                frame_idx = int(hit_time * self.frame_rate)
                
                # Ensure frame index is within bounds
                if 0 <= frame_idx < total_frames:
                    targets[frame_idx, drum_idx] = 1.0
                    targets[frame_idx, 3] = 0.0  # Not silence anymore
        
        return targets
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading audio."""
        sample = self.samples[idx]
        
        drum_hits = sample['drum_hits']
        total_hits = sum(len(drum_hits[dt]) for dt in ['kick', 'snare', 'hihat'])
        
        return {
            'audio_path': str(self._get_audio_path(sample['audio_path'])),
            'total_hits': total_hits,
            'kick_hits': len(drum_hits['kick']),
            'snare_hits': len(drum_hits['snare']),
            'hihat_hits': len(drum_hits['hihat']),
            'duration': sample.get('duration', None)
        }


def test_dataset():
    """Test dataset with dummy data."""
    import tempfile
    import soundfile as sf
    
    print("Testing DrumDataset...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy audio file (white noise)
        audio_path = tmpdir / 'test_drums.wav'
        sr = 16000
        duration = 2.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        sf.write(audio_path, audio, sr)
        
        # Create dummy label in Ayaan's format
        label = {
            'audio_path': str(audio_path),
            'duration': duration,
            'drum_hits': {
                'kick': [
                    {'time': 0.0, 'velocity': 0.8},
                    {'time': 0.5, 'velocity': 0.7},
                    {'time': 1.0, 'velocity': 0.9}
                ],
                'snare': [
                    {'time': 0.25, 'velocity': 0.9},
                    {'time': 0.75, 'velocity': 0.85}
                ],
                'hihat': [
                    {'time': 0.125, 'velocity': 0.6},
                    {'time': 0.375, 'velocity': 0.65},
                    {'time': 0.625, 'velocity': 0.7},
                    {'time': 0.875, 'velocity': 0.6}
                ]
            },
            'total_hits': 9
        }
        
        label_path = tmpdir / 'test_label.json'
        with open(label_path, 'w') as f:
            json.dump([label], f)
        
        # Create dataset
        dataset = DrumDataset(
            labels_path=str(label_path),
            validate=True
        )
        
        print(f"\n✓ Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        mel_tensor, target_tensor = dataset[0]
        
        print(f"\nSample shapes:")
        print(f"  Mel spectrogram: {mel_tensor.shape}")
        print(f"  Target: {target_tensor.shape}")
        print(f"  Mel range: [{mel_tensor.min():.3f}, {mel_tensor.max():.3f}]")
        
        # Analyze targets
        kick_frames = (target_tensor[:, 0] == 1.0).sum()
        snare_frames = (target_tensor[:, 1] == 1.0).sum()
        hihat_frames = (target_tensor[:, 2] == 1.0).sum()
        silence_frames = (target_tensor[:, 3] == 1.0).sum()
        
        print(f"\nTarget statistics:")
        print(f"  Kick frames: {kick_frames}")
        print(f"  Snare frames: {snare_frames}")
        print(f"  Hihat frames: {hihat_frames}")
        print(f"  Silence frames: {silence_frames}")
        
        # Test info retrieval
        info = dataset.get_sample_info(0)
        print(f"\nSample info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Verify target encoding
        assert mel_tensor.shape == (1, 500, 128), "Mel shape mismatch"
        assert target_tensor.shape == (500, 4), "Target shape mismatch"
        assert kick_frames == 3, f"Expected 3 kick frames, got {kick_frames}"
        assert snare_frames == 2, f"Expected 2 snare frames, got {snare_frames}"
        assert hihat_frames == 4, f"Expected 4 hihat frames, got {hihat_frames}"
        
        print("\n✓ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()