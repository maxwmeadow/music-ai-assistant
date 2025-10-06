"""
Melody Dataset for Hum2Melody Training

Loads audio files and corresponding note labels, converts to mel spectrograms
and frame-level note targets for training.

Label Format (JSON):
{
    'audio_path': 'path/to/audio.wav',
    'notes': [60, 62, 64, 65, 67],  # MIDI note numbers
    'start_times': [0.0, 0.5, 1.0, 1.5, 2.0],  # Start time in seconds
    'durations': [0.4, 0.4, 0.4, 0.8, 1.2],  # Duration in seconds
    'confidence': 0.85  # Optional: labeling confidence
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

class MelodyDataset(Dataset):
    """
    PyTorch Dataset for melody transcription.
    
    Args:
        labels_path (str): Path to labels JSON file or directory containing label JSONs
        audio_root (str): Root directory for audio files (if paths in labels are relative)
        sample_rate (int): Target sample rate for audio (default: 16000)
        n_mels (int): Number of mel frequency bands (default: 128)
        target_frames (int): Target number of time frames (default: 500)
        hop_length (int): Hop length for STFT (default: 512)
        min_midi (int): Minimum MIDI note number (default: 21, A0)
        max_midi (int): Maximum MIDI note number (default: 108, C8)
        validate (bool): Validate labels on load (default: True)
    """
    
    def __init__(
        self,
        labels_path: str,
        audio_root: Optional[str] = None,
        sample_rate: int = 16000,
        n_mels: int = 128,
        target_frames: int = 500,
        hop_length: int = 512,
        min_midi: int = 21,
        max_midi: int = 108,
        validate: bool = True
    ):
        self.labels_path = Path(labels_path)
        self.audio_root = Path(audio_root) if audio_root else None
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.hop_length = hop_length
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.num_notes = max_midi - min_midi + 1  # 88 notes
        
        # Frame rate calculation
        self.frame_rate = sample_rate / hop_length  # 16000 / 512 = 31.25 fps
        
        # Load all labels
        self.samples = self._load_labels()
        
        if validate:
            self._validate_labels()
        
        print(f"[MelodyDataset] Loaded {len(self.samples)} samples")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Mel bands: {n_mels}")
        print(f"  Target frames: {target_frames} (~{target_frames/self.frame_rate:.1f}s)")
        print(f"  MIDI range: {min_midi}-{max_midi} ({self.num_notes} notes)")
    
    def _load_labels(self) -> List[Dict]:
        """Load all label files."""
        samples = []
        
        if self.labels_path.is_file():
            # Single JSON file with list of samples
            with open(self.labels_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    # Assume it's a manifest with 'audio_files' and 'label_files'
                    audio_files = data.get('audio_files', [])
                    label_files = data.get('label_files', [])
                    
                    for audio_path, label_path in zip(audio_files, label_files):
                        with open(label_path, 'r') as lf:
                            label_data = json.load(lf)
                            label_data['audio_path'] = audio_path
                            samples.append(label_data)
        
        elif self.labels_path.is_dir():
            # Directory of JSON label files
            for label_file in sorted(self.labels_path.glob('*.json')):
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    samples.append(label_data)
        
        else:
            raise ValueError(f"Labels path not found: {self.labels_path}")
        
        return samples
    
    def _validate_labels(self):
        """Validate label data and filter out invalid samples."""
        valid_samples = []
        invalid_count = 0
        
        for idx, sample in enumerate(self.samples):
            try:
                # Check required fields
                assert 'audio_path' in sample, "Missing 'audio_path'"
                assert 'notes' in sample, "Missing 'notes'"
                assert 'start_times' in sample, "Missing 'start_times'"
                assert 'durations' in sample, "Missing 'durations'"
                
                # Check data integrity
                notes = sample['notes']
                start_times = sample['start_times']
                durations = sample['durations']
                
                assert len(notes) == len(start_times) == len(durations), \
                    "Length mismatch between notes, start_times, and durations"
                
                # Check MIDI range
                notes_array = np.array(notes)
                assert np.all((notes_array >= self.min_midi) & (notes_array <= self.max_midi)), \
                    f"Notes outside MIDI range {self.min_midi}-{self.max_midi}"
                
                # Check timing
                assert all(t >= 0 for t in start_times), "Negative start times"
                assert all(d > 0 for d in durations), "Non-positive durations"
                
                # Check audio file exists
                audio_path = self._get_audio_path(sample['audio_path'])
                assert audio_path.exists(), f"Audio file not found: {audio_path}"
                
                valid_samples.append(sample)
                
            except AssertionError as e:
                invalid_count += 1
                warnings.warn(f"Invalid sample {idx}: {e}")
        
        self.samples = valid_samples
        
        if invalid_count > 0:
            print(f"[MelodyDataset] Filtered out {invalid_count} invalid samples")
    
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
            target_tensor: Frame-level note targets, shape (target_frames, num_notes)
        """
        sample = self.samples[idx]
        
        # Load and process audio
        audio_path = self._get_audio_path(sample['audio_path'])
        mel_spec = self._load_mel_spectrogram(audio_path)
        
        # Convert labels to frame-level targets
        target = self._labels_to_frames(
            notes=sample['notes'],
            start_times=sample['start_times'],
            durations=sample['durations'],
            total_frames=mel_spec.shape[1]  # Use actual mel_spec frames
        )
        
        # Ensure target matches mel_spec length
        if target.shape[0] != mel_spec.shape[1]:
            # Pad or truncate target to match
            if target.shape[0] < mel_spec.shape[1]:
                padding = np.zeros((mel_spec.shape[1] - target.shape[0], self.num_notes))
                target = np.vstack([target, padding])
            else:
                target = target[:mel_spec.shape[1], :]
        
        # Convert to tensors
        mel_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0)  # Add channel dim. why transpose? i dont know but this bug took a while and that fixed it
        target_tensor = torch.FloatTensor(target)
        
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
            fmin=80,   # Human voice range
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        # Typical dB range is -80 to 0
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)
        
        # Pad or truncate to target_frames
        if mel_spec_normalized.shape[1] < self.target_frames:
            # Pad with zeros on the right
            pad_width = self.target_frames - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(
                mel_spec_normalized, 
                ((0, 0), (0, pad_width)), 
                mode='constant'
            )
        elif mel_spec_normalized.shape[1] > self.target_frames:
            # Truncate
            mel_spec_normalized = mel_spec_normalized[:, :self.target_frames]
        
        return mel_spec_normalized
    
    def _labels_to_frames(
        self,
        notes: List[int],
        start_times: List[float],
        durations: List[float],
        total_frames: int
    ) -> np.ndarray:
        """
        Convert note labels to frame-level targets.
        
        Args:
            notes: List of MIDI note numbers
            start_times: List of note start times (seconds)
            durations: List of note durations (seconds)
            total_frames: Total number of frames
        
        Returns:
            target: Binary matrix of shape (total_frames, num_notes)
                    1 if note is active in that frame, 0 otherwise
        """
        # Initialize target matrix
        target = np.zeros((total_frames, self.num_notes), dtype=np.float32)
        
        for note, start_time, duration in zip(notes, start_times, durations):
            # Convert to frame indices
            start_frame = int(start_time * self.frame_rate)
            end_frame = int((start_time + duration) * self.frame_rate)
            
            # Clip to valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))
            
            # Convert MIDI to note index (0-87)
            note_idx = note - self.min_midi
            
            if 0 <= note_idx < self.num_notes:
                # Set frames where note is active
                target[start_frame:end_frame, note_idx] = 1.0
        
        return target
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading audio."""
        sample = self.samples[idx]
        
        return {
            'audio_path': str(self._get_audio_path(sample['audio_path'])),
            'num_notes': len(sample['notes']),
            'duration': max(
                s + d for s, d in zip(sample['start_times'], sample['durations'])
            ),
            'notes': sample['notes'],
            'confidence': sample.get('confidence', None)
        }


def test_dataset():
    """Test dataset with dummy data."""
    import tempfile
    import os
    
    print("Testing MelodyDataset...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy audio file (sine wave)
        audio_path = tmpdir / 'test_audio.wav'
        sr = 16000
        duration = 2.0
        freq = 440  # A4
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t)
        
        import soundfile as sf
        sf.write(audio_path, audio, sr)
        
        # Create dummy label
        label = {
            'audio_path': str(audio_path),
            'notes': [60, 64, 67],
            'start_times': [0.0, 0.5, 1.0],
            'durations': [0.4, 0.4, 0.8],
            'confidence': 0.9
        }
        
        label_path = tmpdir / 'test_label.json'
        with open(label_path, 'w') as f:
            json.dump([label], f)
        
        # Create dataset
        dataset = MelodyDataset(
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
        print(f"  Active frames: {target_tensor.sum(dim=1).nonzero().shape[0]}")
        print(f"  Total active notes: {target_tensor.sum():.0f}")
        
        # Test info retrieval
        info = dataset.get_sample_info(0)
        print(f"\nSample info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()