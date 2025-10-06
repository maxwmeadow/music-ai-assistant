"""
Melody Predictor - Inference Pipeline for Hum2Melody

Handles:
- Model loading from checkpoint
- Audio preprocessing (reuses dataset logic)
- Inference with trained model
- Post-processing predictions to discrete notes
- Conversion to IR format for backend

Usage:
    predictor = MelodyPredictor('checkpoints/best_model.pth')
    track = predictor.predict_from_file('audio.wav')
    # OR
    track = predictor.predict_from_bytes(audio_bytes)
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings

from models.hum2melody_model import Hum2MelodyCRNN
from schemas import Track, Note


class MelodyPredictor:
    """
    Inference engine for melody prediction from audio.
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        device (str): Device to run inference on ('cuda' or 'cpu')
        threshold (float): Activation threshold for note detection (default: 0.5)
        min_note_duration (float): Minimum note duration in seconds (default: 0.1)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        min_note_duration: float = 0.1
    ):
        self.checkpoint_path = Path(checkpoint_path)
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[MelodyPredictor] Using device: {self.device}")
        
        # Hyperparameters (must match training)
        self.sample_rate = 16000
        self.n_mels = 128
        self.target_frames = 500
        self.hop_length = 512
        self.min_midi = 21
        self.max_midi = 108
        self.num_notes = 88
        
        # Post-processing parameters
        self.threshold = threshold
        self.min_note_duration = min_note_duration
        
        # Frame rate calculation
        self.frame_rate = self.sample_rate / self.hop_length  # 31.25 fps
        
        # Load model
        self.model = self._load_model()
        
        print(f"[MelodyPredictor] Model loaded successfully")
        print(f"  Threshold: {threshold}")
        print(f"  Min note duration: {min_note_duration}s")
    
    def _load_model(self) -> Hum2MelodyCRNN:
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Create model
        model = Hum2MelodyCRNN(
            n_mels=self.n_mels,
            hidden_size=256,
            num_notes=self.num_notes
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[MelodyPredictor] Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode and move to device
        model.eval()
        model.to(self.device)
        
        return model
    
    def predict_from_file(self, audio_path: str) -> Track:
        """
        Predict melody from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Track object with predicted notes in IR format
        """
        audio_path_obj = Path(audio_path)
        
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path_obj, sr=self.sample_rate, mono=True)
        
        return self.predict_from_audio(audio)
    
    def predict_from_bytes(self, audio_bytes: bytes) -> Track:
        """
        Predict melody from audio bytes.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            Track object with predicted notes in IR format
        """
        import io
        
        # Load audio from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
        
        return self.predict_from_audio(audio)
    
    def predict_from_audio(self, audio: np.ndarray) -> Track:
        """
        Predict melody from audio array.
        
        Args:
            audio: Audio samples (numpy array)
            
        Returns:
            Track object with predicted notes in IR format
        """
        # Preprocess audio to mel spectrogram
        mel_spec = self._preprocess_audio(audio)
        
        # Run inference
        probabilities = self._run_inference(mel_spec)
        
        # Post-process to discrete notes
        notes = self._postprocess_predictions(probabilities)
        
        # Create Track in IR format
        track = Track(
            id="melody",
            instrument="lead_synth",
            notes=notes,
            samples=None
        )
        
        return track
    
    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio to mel spectrogram.
        Same logic as MelodyDataset._load_mel_spectrogram
        
        Args:
            audio: Audio samples
            
        Returns:
            Mel spectrogram tensor, shape (1, 1, target_frames, n_mels)
        """
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=2048,
            fmin=80,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
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
        
        # Convert to tensor: (n_mels, time) -> (1, time, n_mels) -> (1, 1, time, n_mels)
        mel_tensor = torch.FloatTensor(mel_spec_normalized.T).unsqueeze(0).unsqueeze(0)
        
        return mel_tensor
    
    def _run_inference(self, mel_spec: torch.Tensor) -> np.ndarray:
        """
        Run model inference.
        
        Args:
            mel_spec: Mel spectrogram tensor (1, 1, target_frames, n_mels)
            
        Returns:
            Probabilities array, shape (output_frames, num_notes)
        """
        mel_spec = mel_spec.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            logits = self.model(mel_spec)  # (1, 62, 88)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            
            # Remove batch dimension and convert to numpy
            probabilities = probabilities.squeeze(0).cpu().numpy()  # (62, 88)
        
        return probabilities
    
    def _postprocess_predictions(self, probabilities: np.ndarray) -> List[Note]:
        """
        Post-process predictions to discrete notes.
        
        This is the critical step: converting frame-level probabilities to
        discrete note events with start times and durations.
        
        Args:
            probabilities: Probability matrix, shape (num_frames, num_notes)
            
        Returns:
            List of Note objects
        """
        num_frames, num_notes = probabilities.shape
        
        # Apply threshold to get binary activations
        activations = probabilities > self.threshold  # (num_frames, num_notes)
        
        notes = []
        
        # For each MIDI note (0-87 mapping to MIDI 21-108)
        for note_idx in range(num_notes):
            # Get activation for this note across time
            note_activations = activations[:, note_idx]
            
            # Find segments where note is active
            segments = self._find_segments(note_activations)
            
            # Convert each segment to a Note
            for start_frame, end_frame in segments:
                # Convert frames to time
                start_time = start_frame / self.frame_rate
                end_time = end_frame / self.frame_rate
                duration = end_time - start_time
                
                # Filter out very short notes (likely noise)
                if duration < self.min_note_duration:
                    continue
                
                # Convert note index to MIDI number
                midi_note = note_idx + self.min_midi
                
                # Calculate average velocity from probabilities
                segment_probs = probabilities[start_frame:end_frame, note_idx]
                velocity = float(np.mean(segment_probs))
                
                notes.append(Note(
                    pitch=int(midi_note),
                    duration=float(duration),
                    velocity=float(velocity)
                ))
        
        # Sort notes by start time (implicit - they're already ordered)
        # In the current implementation, we don't store start_time in Note
        # The notes are played sequentially by duration
        
        return notes
    
    def _find_segments(self, activations: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous segments where a note is active.
        
        Args:
            activations: Binary array of shape (num_frames,)
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        segments = []
        
        in_segment = False
        start_frame = 0
        
        for frame_idx in range(len(activations)):
            if activations[frame_idx] and not in_segment:
                # Start of new segment
                in_segment = True
                start_frame = frame_idx
            elif not activations[frame_idx] and in_segment:
                # End of segment
                in_segment = False
                segments.append((start_frame, frame_idx))
        
        # Handle case where segment continues to end
        if in_segment:
            segments.append((start_frame, len(activations)))
        
        return segments
    
    def predict_with_timing(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Predict melody with explicit timing information.
        
        This version returns notes with start times, useful for debugging
        or alternative use cases.
        
        Args:
            audio: Audio samples
            
        Returns:
            Dictionary with notes and timing info
        """
        mel_spec = self._preprocess_audio(audio)
        probabilities = self._run_inference(mel_spec)
        
        num_frames, num_notes = probabilities.shape
        activations = probabilities > self.threshold
        
        notes_with_timing = []
        
        for note_idx in range(num_notes):
            note_activations = activations[:, note_idx]
            segments = self._find_segments(note_activations)
            
            for start_frame, end_frame in segments:
                start_time = start_frame / self.frame_rate
                end_time = end_frame / self.frame_rate
                duration = end_time - start_time
                
                if duration < self.min_note_duration:
                    continue
                
                midi_note = note_idx + self.min_midi
                segment_probs = probabilities[start_frame:end_frame, note_idx]
                velocity = float(np.mean(segment_probs))
                
                notes_with_timing.append({
                    'pitch': int(midi_note),
                    'start_time': float(start_time),
                    'duration': float(duration),
                    'velocity': float(velocity),
                    'note_name': self._midi_to_note_name(midi_note)
                })
        
        # Sort by start time
        notes_with_timing.sort(key=lambda x: x['start_time'])
        
        return {
            'notes': notes_with_timing,
            'total_duration': num_frames / self.frame_rate,
            'num_notes': len(notes_with_timing)
        }
    
    def _midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI number to note name (e.g., 60 -> 'C4')."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = notes[midi % 12]
        return f"{note}{octave}"


def test_predictor():
    """Test predictor with dummy data."""
    import tempfile
    import soundfile as sf
    
    print("Testing MelodyPredictor...")
    
    # Create dummy audio (sine wave)
    sr = 16000
    duration = 2.0
    freq = 440  # A4
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * freq * t)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save audio
        audio_path = Path(tmpdir) / 'test.wav'
        sf.write(audio_path, audio, sr)
        
        # Create dummy checkpoint (for testing structure)
        checkpoint_path = Path(tmpdir) / 'model.pth'
        model = Hum2MelodyCRNN()
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 0,
            'val_loss': 0.5
        }, checkpoint_path)
        
        # Test predictor
        predictor = MelodyPredictor(str(checkpoint_path), device='cpu')
        
        # Test from file
        track = predictor.predict_from_file(str(audio_path))
        print(f"\nPredicted {len(track.notes) if track.notes else 0} notes")
        
        # Test from bytes
        audio_bytes = audio_path.read_bytes()
        track2 = predictor.predict_from_bytes(audio_bytes)
        print(f"From bytes: {len(track2.notes) if track2.notes else 0} notes")
        
        # Test with timing
        timing_result = predictor.predict_with_timing(audio)
        print(f"\nWith timing info:")
        print(f"  Total duration: {timing_result['total_duration']:.2f}s")
        print(f"  Num notes: {timing_result['num_notes']}")
        
        if timing_result['notes']:
            print(f"  First note: {timing_result['notes'][0]}")
        
        print("\n✓ Predictor test passed!")


if __name__ == '__main__':
    test_predictor()