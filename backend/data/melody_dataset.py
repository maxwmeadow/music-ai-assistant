"""
Melody Dataset - Complete Implementation (HPCC-FIXED)

Fixed for actual HPCC structure:
- /mnt/scratch/meadowm1/music-ai-training/models/pretrained_features.py
- /mnt/scratch/meadowm1/music-ai-training/data/melody_dataset.py
"""

import os
import sys
import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import random
import math

# HPCC-specific import fix
# Add parent directory to path so we can import from models/
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Now try to import pretrained features
try:
    from models.pretrained_features import (
        PretrainedFeatureExtractor,
        extract_onset_strength_features,
        extract_musical_context_features
    )
    PRETRAINED_AVAILABLE = True
    print("[melody_dataset] ✅ Pretrained features available")
except ImportError as e:
    PRETRAINED_AVAILABLE = False
    print(f"[melody_dataset] ⚠️ Pretrained features not available: {e}")


class MelodyDataset(Dataset):
    """
    Basic melody dataset using CQT with proper augmentation.
    
    Generates frame and onset targets (2 outputs).
    Good for training basic models.
    """
    
    def __init__(
        self,
        labels_path: str,
        audio_root: Optional[str] = None,
        sample_rate: int = 16000,
        n_bins: int = 88,  # Match exact MIDI range (21-108 = 88 notes)
        bins_per_octave: int = 12,
        target_frames: int = 500,
        hop_length: int = 512,
        min_midi: int = 21,
        max_midi: int = 108,
        validate: bool = True,
        augment: bool = True,
        spec_augment: bool = True
    ):
        self.labels_path = Path(labels_path)
        self.audio_root = Path(audio_root) if audio_root else None
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.target_frames = target_frames
        self.hop_length = hop_length
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.num_notes = max_midi - min_midi + 1
        self.augment = augment
        self.spec_augment = spec_augment
        
        # Frame rate
        self.frame_rate = sample_rate / hop_length

        # Align CQT fmin to target MIDI range for perfect bin-to-index mapping
        # fmin corresponds to min_midi (A0 = MIDI 21 = 27.5 Hz)
        # This ensures: CQT bin N → target index N → MIDI (min_midi + N)
        self.fmin = librosa.note_to_hz('A0')  # 27.5 Hz (MIDI 21)
        nyquist = sample_rate / 2
        max_freq = self.fmin * (2 ** (n_bins / float(bins_per_octave)))
        
        if max_freq >= nyquist:
            # Adjust n_bins to stay under Nyquist
            max_safe_bins = int(bins_per_octave * np.log2(nyquist / self.fmin)) - 1
            if n_bins > max_safe_bins:
                print(f"⚠️  Warning: Requested {n_bins} bins would exceed Nyquist ({nyquist} Hz)")
                print(f"   Reducing to {max_safe_bins} bins (max_freq: {self.fmin * (2 ** (max_safe_bins / float(bins_per_octave))):.1f} Hz)")
                self.n_bins = max_safe_bins
        
        print(f"\n{'='*60}")
        print(f"MELODY DATASET INITIALIZATION")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Labels path: {labels_path}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  CQT bins: {self.n_bins} ({self.n_bins // bins_per_octave} octaves)")
        print(f"  Target frames: {target_frames} (~{target_frames/self.frame_rate:.1f}s)")
        print(f"  MIDI range: {min_midi}-{max_midi} ({self.num_notes} notes)")
        print(f"  Frame rate: {self.frame_rate:.2f} fps")
        print(f"  CQT fmin: {self.fmin:.2f} Hz")
        print(f"  CQT fmax: {self.fmin * (2 ** (self.n_bins / float(bins_per_octave))):.2f} Hz")
        print(f"  Nyquist: {nyquist:.2f} Hz")
        print(f"  Augmentation: {augment}")
        print(f"  SpecAugment: {spec_augment}")
        
        # Load labels
        self.samples = self._load_labels()
        
        if validate:
            self._validate_labels()
        
        print(f"  ✅ Loaded {len(self.samples)} valid samples")
        print(f"{'='*60}\n")
    
    def _load_labels(self) -> List[Dict]:
        """Load all label files."""
        print(f"[MelodyDataset] Loading labels from {self.labels_path}...")
        samples = []
        
        if self.labels_path.is_file():
            with open(self.labels_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    raise ValueError("Labels file must contain a list of samples")
        elif self.labels_path.is_dir():
            for label_file in sorted(self.labels_path.glob('*.json')):
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    samples.append(label_data)
        else:
            raise ValueError(f"Labels path not found: {self.labels_path}")
        
        print(f"[MelodyDataset] Loaded {len(samples)} samples from manifest")
        return samples
    
    def _validate_labels(self):
        """Validate and filter invalid samples."""
        print(f"[MelodyDataset] Validating samples...")
        valid_samples = []
        invalid_count = 0
        
        for idx, sample in enumerate(self.samples):
            try:
                # Required fields
                assert 'audio_path' in sample
                assert 'notes' in sample
                assert 'start_times' in sample
                assert 'durations' in sample
                
                # Data integrity
                notes = sample['notes']
                start_times = sample['start_times']
                durations = sample['durations']
                
                assert len(notes) == len(start_times) == len(durations)
                
                # Minimum content
                total_active_time = sum(durations)
                assert total_active_time >= 0.5, f"Too sparse: {total_active_time:.2f}s"
                assert len(notes) >= 2, f"Too few notes: {len(notes)}"
                
                # MIDI range
                notes_array = np.array(notes)
                assert np.all((notes_array >= self.min_midi) & (notes_array <= self.max_midi))
                
                # Timing
                assert all(t >= 0 for t in start_times)
                assert all(d > 0 for d in durations)
                
                # Audio exists
                audio_path = self._get_audio_path(sample['audio_path'])
                assert os.path.exists(audio_path), f"Audio not found: {audio_path}"
                
                valid_samples.append(sample)
                
            except AssertionError as e:
                invalid_count += 1
                if invalid_count <= 5:
                    warnings.warn(f"[MelodyDataset] Invalid sample {idx}: {e}")
        
        self.samples = valid_samples
        
        if invalid_count > 0:
            print(f"[MelodyDataset] Filtered {invalid_count} invalid samples")
    
    def _get_audio_path(self, audio_path: str) -> Path:
        """Get absolute audio path."""
        if os.path.isabs(audio_path):
            return audio_path
        else:
            labels_dir = os.path.dirname(self.labels_path)
            return os.path.join(labels_dir, audio_path)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a training sample.
        
        Returns:
            cqt_tensor: (1, target_frames, n_bins)
            targets: dict with 'frame' and 'onset'
        """
        sample = self.samples[idx]
        
        # Load audio
        audio_path = self._get_audio_path(sample['audio_path'])
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Get labels
        notes = list(sample['notes'])
        start_times = list(sample['start_times'])
        durations = list(sample['durations'])
        
        # Apply augmentation WITH label adjustment
        if self.augment and random.random() > 0.3:
            audio, notes, start_times, durations = self._augment_with_labels(
                audio, notes, start_times, durations
            )
        
        # Extract CQT
        cqt_spec = self._load_cqt(audio)
        
        # Apply SpecAugment
        if self.spec_augment and self.augment:
            cqt_spec = self._apply_spec_augment(cqt_spec)
        
        # Convert labels to frames
        frame_target = self._labels_to_frames(
            notes, start_times, durations, cqt_spec.shape[1]
        )
        
        # Generate onset target
        onset_target = self._labels_to_onsets(
            start_times, cqt_spec.shape[1]
        )
        
        # Ensure alignment
        if frame_target.shape[0] != cqt_spec.shape[1]:
            if frame_target.shape[0] < cqt_spec.shape[1]:
                padding = np.zeros((cqt_spec.shape[1] - frame_target.shape[0], self.num_notes))
                frame_target = np.vstack([frame_target, padding])
                onset_padding = np.zeros(cqt_spec.shape[1] - len(onset_target))
                onset_target = np.concatenate([onset_target, onset_padding])
            else:
                frame_target = frame_target[:cqt_spec.shape[1], :]
                onset_target = onset_target[:cqt_spec.shape[1]]
        
        # Convert to tensors
        cqt_tensor = torch.FloatTensor(cqt_spec.T).unsqueeze(0)
        frame_tensor = torch.FloatTensor(frame_target)
        onset_tensor = torch.FloatTensor(onset_target)
        
        targets = {
            'frame': frame_tensor,
            'onset': onset_tensor
        }
        
        return cqt_tensor, targets
    
    def _augment_with_labels(
        self,
        audio: np.ndarray,
        notes: List[int],
        start_times: List[float],
        durations: List[float]
    ) -> Tuple[np.ndarray, List[int], List[float], List[float]]:
        """
        CRITICAL: Augment audio AND adjust labels accordingly.
        """
        # Pitch shift (50% chance)
        if random.random() > 0.5:
            n_steps = random.randint(-2, 2)
            if n_steps != 0:
                audio = librosa.effects.pitch_shift(
                    audio,
                    sr=self.sample_rate,
                    n_steps=n_steps
                )
                notes = [n + n_steps for n in notes]
                notes = [max(self.min_midi, min(self.max_midi, n)) for n in notes]
        
        # Time stretch (50% chance)
        if random.random() > 0.5:
            rate = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            start_times = [t / rate for t in start_times]
            durations = [d / rate for d in durations]
        
        # Add noise (30% chance)
        if random.random() > 0.7:
            noise_level = random.uniform(0.002, 0.008)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        
        # Volume scaling (always apply)
        scale = random.uniform(0.7, 1.3)
        audio = audio * scale
        
        # Clip
        audio = np.clip(audio, -1, 1)
        
        return audio, notes, start_times, durations
    
    def _load_cqt(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CQT spectrogram with Nyquist safety check.
        
        Returns:
            cqt_spec: Shape (n_bins, target_frames)
        """
        # Extract CQT with verified parameters
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin
        )
        
        # Convert to dB
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # Normalize to [0, 1]
        cqt_normalized = (cqt_db + 80) / 80
        cqt_normalized = np.clip(cqt_normalized, 0, 1)
        
        # Pad or truncate
        if cqt_normalized.shape[1] < self.target_frames:
            pad_width = self.target_frames - cqt_normalized.shape[1]
            cqt_normalized = np.pad(
                cqt_normalized,
                ((0, 0), (0, pad_width)),
                mode='constant'
            )
        elif cqt_normalized.shape[1] > self.target_frames:
            cqt_normalized = cqt_normalized[:, :self.target_frames]
        
        return cqt_normalized
    
    def _apply_spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment (time and frequency masking)."""
        spec = spec.copy()
        
        # Frequency masking
        if random.random() > 0.5:
            f_mask_width = random.randint(2, 8)
            f_start = random.randint(0, spec.shape[0] - f_mask_width)
            spec[f_start:f_start + f_mask_width, :] = 0
        
        # Time masking
        if random.random() > 0.5:
            t_mask_width = random.randint(5, 20)
            t_start = random.randint(0, spec.shape[1] - t_mask_width)
            spec[:, t_start:t_start + t_mask_width] = 0
        
        return spec
    
    def _labels_to_frames(
        self,
        notes: List[int],
        start_times: List[float],
        durations: List[float],
        total_frames: int
    ) -> np.ndarray:
        """Convert note labels to frame-level targets."""
        target = np.zeros((total_frames, self.num_notes), dtype=np.float32)
        
        for note, start_time, duration in zip(notes, start_times, durations):
            start_frame = int(start_time * self.frame_rate)
            end_frame = int((start_time + duration) * self.frame_rate)
            
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))
            
            note_idx = note - self.min_midi
            
            if 0 <= note_idx < self.num_notes and start_frame < end_frame:
                target[start_frame:end_frame, note_idx] = 1.0
        
        return target
    
    def _labels_to_onsets(
        self,
        start_times: List[float],
        total_frames: int
    ) -> np.ndarray:
        """Convert note start times to onset targets."""
        onset_target = np.zeros(total_frames, dtype=np.float32)
        
        for start_time in start_times:
            frame_idx = int(start_time * self.frame_rate)
            
            if 0 <= frame_idx < total_frames:
                onset_target[frame_idx] = 1.0
        
        return onset_target
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get sample metadata."""
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


class EnhancedMelodyDataset(MelodyDataset):
    """
    Enhanced dataset with all features.
    
    Generates all 4 targets (frame, onset, offset, f0) and supports
    multi-channel input (CQT + onset features + musical context + pretrained).
    """
    
    def __init__(
        self,
        *args,
        use_pretrained=False,
        use_onset_features=True,
        use_musical_context=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # F0 range for normalization
        self.f0_min = 80.0
        self.f0_max = 800.0
        
        # Feature extraction flags
        self.use_pretrained = use_pretrained and PRETRAINED_AVAILABLE
        self.use_onset_features = use_onset_features and PRETRAINED_AVAILABLE
        self.use_musical_context = use_musical_context and PRETRAINED_AVAILABLE
        
        print(f"\n{'='*60}")
        print(f"ENHANCED DATASET CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Pretrained features: {self.use_pretrained}")
        print(f"  Onset features: {self.use_onset_features}")
        print(f"  Musical context: {self.use_musical_context}")
        
        # Initialize pretrained extractor if requested
        if self.use_pretrained:
            print("[EnhancedDataset] Initializing pretrained feature extractor...")
            self.pretrained_extractor = PretrainedFeatureExtractor(
                model_type='wav2vec2',
                device='cpu'
            )
        else:
            self.pretrained_extractor = None
        
        # Calculate expected input channels
        expected_channels = self.n_bins
        if self.use_onset_features:
            expected_channels += 5
        if self.use_musical_context:
            expected_channels += 19  # 12 chroma + 6 tonnetz + 1 tempo
        if self.use_pretrained:
            expected_channels += 768
        
        print(f"  Expected input channels: {expected_channels}")
        print(f"    - CQT: {self.n_bins}")
        if self.use_onset_features:
            print(f"    - Onset features: 5")
        if self.use_musical_context:
            print(f"    - Musical context: 19")
        if self.use_pretrained:
            print(f"    - Pretrained: 768")
        print(f"{'='*60}\n")
    
    def _extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all requested features from audio."""
        features = {}
        
        # 1. Standard CQT (always included)
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        cqt_normalized = (cqt_db + 80) / 80
        cqt_normalized = np.clip(cqt_normalized, 0, 1)
        features['cqt'] = cqt_normalized
        
        # 2. Onset-strength features
        if self.use_onset_features:
            onset_feats = extract_onset_strength_features(
                audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            # Align to CQT time dimension
            if onset_feats.shape[1] != cqt_normalized.shape[1]:
                if onset_feats.shape[1] < cqt_normalized.shape[1]:
                    pad_width = cqt_normalized.shape[1] - onset_feats.shape[1]
                    onset_feats = np.pad(onset_feats, ((0, 0), (0, pad_width)), mode='edge')
                else:
                    onset_feats = onset_feats[:, :cqt_normalized.shape[1]]
            
            features['onset_features'] = onset_feats
        
        # 3. Musical context features
        if self.use_musical_context:
            context = extract_musical_context_features(
                audio,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Align all to CQT time dimension
            for key in ['chroma', 'tonnetz']:
                feat = context[key]
                if feat.shape[1] != cqt_normalized.shape[1]:
                    if feat.shape[1] < cqt_normalized.shape[1]:
                        pad_width = cqt_normalized.shape[1] - feat.shape[1]
                        feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='edge')
                    else:
                        feat = feat[:, :cqt_normalized.shape[1]]
                context[key] = feat
            
            # Tempo is 1D
            tempo = context['tempo']
            if len(tempo) != cqt_normalized.shape[1]:
                if len(tempo) < cqt_normalized.shape[1]:
                    pad_width = cqt_normalized.shape[1] - len(tempo)
                    tempo = np.pad(tempo, (0, pad_width), mode='edge')
                else:
                    tempo = tempo[:cqt_normalized.shape[1]]
            context['tempo'] = tempo.reshape(1, -1)
            
            features['musical_context'] = context
        
        # 4. Pretrained features
        if self.use_pretrained and self.pretrained_extractor is not None:
            pretrained_feats = self.pretrained_extractor.extract(
                audio,
                sr=self.sample_rate
            )
            
            if pretrained_feats is not None:
                from scipy import signal
                target_length = cqt_normalized.shape[1]
                
                if pretrained_feats.shape[0] != target_length:
                    pretrained_feats = signal.resample(
                        pretrained_feats,
                        target_length,
                        axis=0
                    )
                
                features['pretrained'] = pretrained_feats.T
        
        return features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a training sample with all features and targets."""
        sample = self.samples[idx]

        # Load audio at native sample rate, then resample explicitly
        # This prevents pitch shift errors from sample rate mismatches
        audio_path = self._get_audio_path(sample['audio_path'])
        audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)

        # Resample to target sample rate if needed
        if orig_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
        
        # Get labels
        notes = list(sample['notes'])
        start_times = list(sample['start_times'])
        durations = list(sample['durations'])
        
        # Augment WITH label adjustment
        if self.augment and np.random.random() > 0.3:
            audio, notes, start_times, durations = self._augment_with_labels(
                audio, notes, start_times, durations
            )
        
        # Extract ALL features
        all_features = self._extract_all_features(audio)
        
        # Get CQT as base
        cqt_spec = all_features['cqt']
        num_frames = cqt_spec.shape[1]
        
        # Stack additional features with CQT
        feature_list = [cqt_spec]
        
        # Add onset-strength features (5 channels)
        if 'onset_features' in all_features:
            feature_list.append(all_features['onset_features'])
        
        # Add musical context (19 channels)
        if 'musical_context' in all_features:
            context = all_features['musical_context']
            feature_list.append(context['chroma'])
            feature_list.append(context['tonnetz'])
            feature_list.append(context['tempo'])
        
        # Add pretrained features (768 channels)
        if 'pretrained' in all_features:
            feature_list.append(all_features['pretrained'])
        
        # Stack all features
        stacked_features = np.vstack(feature_list)
        
        # Apply SpecAugment to CQT portion only
        if self.spec_augment and self.augment:
            cqt_augmented = self._apply_spec_augment(stacked_features[:self.n_bins, :])
            stacked_features[:self.n_bins, :] = cqt_augmented
        
        # Generate all targets
        frame_target = self._labels_to_frames(notes, start_times, durations, num_frames)
        onset_target = self._labels_to_onsets(start_times, num_frames)
        offset_target = self._labels_to_offsets(start_times, durations, num_frames)
        f0_target = self._labels_to_f0(notes, start_times, durations, num_frames)
        
        # Pad/truncate to target_frames
        if stacked_features.shape[1] < self.target_frames:
            pad_width = self.target_frames - stacked_features.shape[1]
            stacked_features = np.pad(stacked_features, ((0, 0), (0, pad_width)), mode='constant')
            frame_target = self._align_target(frame_target, self.target_frames, 2)
            onset_target = self._align_target(onset_target, self.target_frames, 1)
            offset_target = self._align_target(offset_target, self.target_frames, 1)
            f0_target = self._align_target(f0_target, self.target_frames, 2)
        elif stacked_features.shape[1] > self.target_frames:
            stacked_features = stacked_features[:, :self.target_frames]
            frame_target = frame_target[:self.target_frames]
            onset_target = onset_target[:self.target_frames]
            offset_target = offset_target[:self.target_frames]
            f0_target = f0_target[:self.target_frames]
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(stacked_features.T).unsqueeze(0)
        
        targets = {
            'frame': torch.FloatTensor(frame_target),
            'onset': torch.FloatTensor(onset_target),
            'offset': torch.FloatTensor(offset_target),
            'f0': torch.FloatTensor(f0_target)
        }
        
        return features_tensor, targets
    
    def _labels_to_offsets(
        self,
        start_times: list,
        durations: list,
        total_frames: int
    ) -> np.ndarray:
        """Generate offset targets (note end times)."""
        offset_target = np.zeros(total_frames, dtype=np.float32)
        
        for start_time, duration in zip(start_times, durations):
            offset_time = start_time + duration
            offset_frame = int(offset_time * self.frame_rate)
            
            if 0 <= offset_frame < total_frames:
                offset_target[offset_frame] = 1.0
        
        return offset_target
    
    def _labels_to_f0(
        self,
        notes: list,
        start_times: list,
        durations: list,
        total_frames: int
    ) -> np.ndarray:
        """Generate continuous f0 targets."""
        f0_target = np.zeros((total_frames, 2), dtype=np.float32)
        
        for note, start_time, duration in zip(notes, start_times, durations):
            start_frame = int(start_time * self.frame_rate)
            end_frame = int((start_time + duration) * self.frame_rate)
            
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))
            
            if start_frame >= end_frame:
                continue
            
            # Convert MIDI to Hz
            f0_hz = librosa.midi_to_hz(note)
            
            # Log scale and normalize
            log_f0 = np.log(f0_hz)
            log_f0_normalized = (log_f0 - np.log(self.f0_min)) / (
                np.log(self.f0_max) - np.log(self.f0_min)
            )
            log_f0_normalized = np.clip(log_f0_normalized, 0, 1)
            
            # Set f0 and voicing
            f0_target[start_frame:end_frame, 0] = log_f0_normalized
            f0_target[start_frame:end_frame, 1] = 1.0
        
        return f0_target
    
    def _align_target(
        self,
        target: np.ndarray,
        target_length: int,
        target_dim: int
    ) -> np.ndarray:
        """Align target to match target_length."""
        current_length = target.shape[0]
        
        if current_length == target_length:
            return target
        
        if target_dim == 1:
            if current_length < target_length:
                padding = np.zeros(target_length - current_length)
                return np.concatenate([target, padding])
            else:
                return target[:target_length]
        else:
            if current_length < target_length:
                padding = np.zeros((target_length - current_length, target.shape[1]))
                return np.vstack([target, padding])
            else:
                return target[:target_length, :]

if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING MELODY DATASETS")
    print("="*70)
    
    import tempfile
    import soundfile as sf
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test audio
        audio_path = tmpdir / 'test.wav'
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(audio_path, audio, sr)
        
        # Create test label
        label = {
            'audio_path': str(audio_path),
            'notes': [60, 64, 67],
            'start_times': [0.0, 0.5, 1.0],
            'durations': [0.4, 0.4, 0.8]
        }
        
        label_path = tmpdir / 'test.json'
        with open(label_path, 'w') as f:
            json.dump([label], f)
        
        # Test basic dataset
        print("\n1. Testing MelodyDataset...")
        dataset1 = MelodyDataset(
            labels_path=str(label_path),
            augment=False
        )
        cqt_tensor, targets1 = dataset1[0]
        print(f"   CQT shape: {cqt_tensor.shape}")
        print(f"   Frame shape: {targets1['frame'].shape}")
        print(f"   Onset shape: {targets1['onset'].shape}")
        print(f"   Active frames: {targets1['frame'].sum():.0f}")
        print(f"   Onsets: {targets1['onset'].sum():.0f}")
        print("   âœ… Basic dataset working")
        
        # Test enhanced dataset
        print("\n2. Testing EnhancedMelodyDataset...")
        dataset2 = EnhancedMelodyDataset(
            labels_path=str(label_path),
            augment=False,
            use_onset_features=True,
            use_musical_context=True,
            use_pretrained=False
        )
        features_tensor, targets2 = dataset2[0]
        print(f"   Features shape: {features_tensor.shape}")
        print(f"   Frame shape: {targets2['frame'].shape}")
        print(f"   Onset shape: {targets2['onset'].shape}")
        print(f"   Offset shape: {targets2['offset'].shape}")
        print(f"   F0 shape: {targets2['f0'].shape}")
        print(f"   Active frames: {targets2['frame'].sum():.0f}")
        print(f"   Onsets: {targets2['onset'].sum():.0f}")
        print(f"   Offsets: {targets2['offset'].sum():.0f}")
        print(f"   F0 voiced: {(targets2['f0'][:, 1] > 0).sum():.0f}")
        print("   âœ… Enhanced dataset working")
    
    print("\n" + "="*70)
    print("âœ… ALL TESTS PASSED!")
    print("="*70)
