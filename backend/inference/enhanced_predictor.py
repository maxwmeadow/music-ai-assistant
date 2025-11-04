"""
Enhanced Predictor for Multi-Task Hum2Melody Model

Works with EnhancedHum2MelodyModel (4 heads) or ImprovedHum2MelodyWithOnsets (2 heads)

Key features:
1. Automatically detects model architecture from checkpoint
2. Uses onset-informed decoding for better note boundaries
3. Compatible with both improved and enhanced models
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Optional

print("[ENHANCED_PREDICTOR] Starting import...")

# Import models - try both
try:
    from backend.models.enhanced_hum2melody_model import EnhancedHum2MelodyModel

    print("[ENHANCED_PREDICTOR] ✅ EnhancedHum2MelodyModel imported")
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"[ENHANCED_PREDICTOR] ⚠️ EnhancedHum2MelodyModel not available: {e}")
    ENHANCED_AVAILABLE = False

try:
    from backend.models.improved_hum2melody_model import ImprovedHum2MelodyWithOnsets

    print("[ENHANCED_PREDICTOR] ✅ ImprovedHum2MelodyWithOnsets imported")
    IMPROVED_AVAILABLE = True
except ImportError as e:
    print(f"[ENHANCED_PREDICTOR] ⚠️ ImprovedHum2MelodyWithOnsets not available: {e}")
    IMPROVED_AVAILABLE = False

# Import decoder
try:
    from backend.inference.onset_informed_decoder import OnsetInformedDecoder, Note

    print("[ENHANCED_PREDICTOR] ✅ OnsetInformedDecoder imported")
    DECODER_AVAILABLE = True
except ImportError as e:
    print(f"[ENHANCED_PREDICTOR] ⚠️ OnsetInformedDecoder not available: {e}")
    DECODER_AVAILABLE = False
    # Fallback Note class
    from dataclasses import dataclass


    @dataclass
    class Note:
        pitch: int
        start: float
        duration: float
        velocity: float


class EnhancedMelodyPredictor:
    """
    Predictor that works with multi-task models.

    Automatically detects model type from checkpoint and uses appropriate
    post-processing (onset-informed decoding when available).
    """

    def __init__(
            self,
            checkpoint_path: str,
            device: Optional[str] = None,
            frame_threshold: float = 0.4,
            onset_threshold: float = 0.5,
            min_note_duration: float = 0.12,
            use_onset_decoder: bool = True
    ):
        print("[ENHANCED_PREDICTOR] Initializing...")
        print(f"  Checkpoint: {checkpoint_path}")

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"  Device: {self.device}")

        # Audio parameters (must match training)
        self.sample_rate = 16000
        self.n_bins = 84  # CQT bins (7 octaves)
        self.bins_per_octave = 12
        self.target_frames = 500
        self.hop_length = 512
        self.min_midi = 21
        self.max_midi = 108
        self.num_notes = 88

        # Frame rate
        self.frame_rate = self.sample_rate / self.hop_length  # 31.25 fps

        # Thresholds
        self.frame_threshold = frame_threshold
        self.onset_threshold = onset_threshold
        self.min_note_duration = min_note_duration
        self.use_onset_decoder = use_onset_decoder and DECODER_AVAILABLE

        # Load model
        self.model, self.model_type = self._load_model()

        print(f"  Model type: {self.model_type}")
        print(f"  Use onset decoder: {self.use_onset_decoder}")
        print("[ENHANCED_PREDICTOR] ✅ Initialization complete")

    def _load_model(self):
        """Load model and detect type from checkpoint."""
        print("[ENHANCED_PREDICTOR] Loading checkpoint...")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Found training checkpoint (epoch {checkpoint.get('epoch', '?')})")
        else:
            state_dict = checkpoint
            print("  Found raw state dict")

        # Detect model type from keys
        keys = list(state_dict.keys())
        print(f"  Total keys: {len(keys)}")

        has_onset_head = any('onset_head' in k for k in keys)
        has_offset_head = any('offset_head' in k for k in keys)
        has_f0_head = any('f0_head' in k for k in keys)
        has_frame_head = any('frame_head' in k for k in keys)

        print(f"  Has frame_head: {has_frame_head}")
        print(f"  Has onset_head: {has_onset_head}")
        print(f"  Has offset_head: {has_offset_head}")
        print(f"  Has f0_head: {has_f0_head}")

        # Determine model type
        if has_f0_head and has_offset_head:
            model_type = "enhanced"
            print("  → Detected: EnhancedHum2MelodyModel (4 heads)")

            if not ENHANCED_AVAILABLE:
                raise ImportError("EnhancedHum2MelodyModel not available but checkpoint requires it")

            model = EnhancedHum2MelodyModel(
                n_bins=self.n_bins,
                hidden_size=256,
                num_notes=self.num_notes,
                use_attention=True
            )

        elif has_onset_head:
            model_type = "improved"
            print("  → Detected: ImprovedHum2MelodyWithOnsets (2 heads)")

            if not IMPROVED_AVAILABLE:
                raise ImportError("ImprovedHum2MelodyWithOnsets not available but checkpoint requires it")

            model = ImprovedHum2MelodyWithOnsets(
                n_bins=self.n_bins,
                hidden_size=256,
                num_notes=self.num_notes,
                use_attention=True
            )

        else:
            raise ValueError("Unknown model type - checkpoint doesn't match any known architecture")

        # Load weights
        print("  Loading state dict...")
        model.load_state_dict(state_dict)

        # Set to eval and move to device
        model.eval()
        model.to(self.device)

        print(f"  ✅ Model loaded ({model.count_parameters():,} parameters)")

        return model, model_type

    def predict_from_file(self, audio_path: str) -> List[Note]:
        """Predict melody from audio file."""
        print(f"[ENHANCED_PREDICTOR] Predicting from: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        print(f"  Audio: {len(audio)} samples, {sr} Hz")

        return self.predict_from_audio(audio)

    def predict_from_audio(self, audio: np.ndarray) -> List[Note]:
        """Predict melody from audio array."""
        print(f"[ENHANCED_PREDICTOR] Predicting from audio array: {audio.shape}")

        # Extract CQT
        cqt = self._extract_cqt(audio)
        print(f"  CQT shape: {cqt.shape}")

        # Run inference
        predictions = self._run_inference(cqt)
        print(f"  Predictions keys: {list(predictions.keys())}")
        for key, val in predictions.items():
            print(f"    {key}: {val.shape}")

        # Decode to notes
        if self.use_onset_decoder and 'onset' in predictions:
            print("  Using onset-informed decoder")
            notes = self._decode_with_onsets(predictions)
        else:
            print("  Using simple decoder (no onsets)")
            notes = self._decode_simple(predictions)

        print(f"  ✅ Generated {len(notes)} notes")
        return notes

    def _extract_cqt(self, audio: np.ndarray) -> torch.Tensor:
        """Extract CQT spectrogram."""
        # CQT
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=librosa.note_to_hz('C2')
        )

        # Convert to dB
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        # Normalize
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

        # To tensor: (n_bins, time) -> (1, time, n_bins) -> (1, 1, time, n_bins)
        cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)

        return cqt_tensor

    def _run_inference(self, cqt: torch.Tensor) -> Dict[str, np.ndarray]:
        """Run model inference."""
        cqt = cqt.to(self.device)

        with torch.no_grad():
            outputs = self.model(cqt)

        # Convert to probabilities and numpy
        predictions = {}

        for key in ['frame', 'onset', 'offset', 'f0']:
            if key in outputs:
                if key == 'f0':
                    # F0 has 2 channels: [log_f0, voicing]
                    predictions[key] = outputs[key].squeeze(0).cpu().numpy()
                else:
                    # Apply sigmoid for frame/onset/offset
                    probs = torch.sigmoid(outputs[key])
                    predictions[key] = probs.squeeze(0).cpu().numpy()

        return predictions

    def _decode_with_onsets(self, predictions: Dict[str, np.ndarray]) -> List[Note]:
        """Decode using onset-informed decoder."""
        frame_probs = predictions['frame']  # (time, 88)
        onset_probs = predictions['onset']  # (time, 1) or (time,)

        # Ensure onset is 1D
        if onset_probs.ndim == 2:
            onset_probs = onset_probs.squeeze(-1)

        # Create decoder
        decoder = OnsetInformedDecoder(
            min_midi=self.min_midi,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            min_note_duration=self.min_note_duration,
            frame_rate=self.frame_rate
        )

        # Decode
        notes = decoder.decode(frame_probs, onset_probs)

        # Convert confidence to velocity
        for note in notes:
            note.velocity = note.confidence

        return notes

    def _decode_simple(self, predictions: Dict[str, np.ndarray]) -> List[Note]:
        """Simple decoding without onsets."""
        frame_probs = predictions['frame']  # (time, 88)

        notes = []

        # For each frame, get most confident pitch
        max_probs = frame_probs.max(axis=1)
        max_pitches = frame_probs.argmax(axis=1)

        # Threshold
        active = max_probs > self.frame_threshold

        # Find continuous segments
        in_note = False
        note_start = 0
        note_pitch = 0
        note_confidences = []

        for i in range(len(active)):
            if active[i] and not in_note:
                # Start new note
                in_note = True
                note_start = i
                note_pitch = max_pitches[i]
                note_confidences = [max_probs[i]]

            elif active[i] and in_note:
                # Continue note (if same pitch)
                if max_pitches[i] == note_pitch:
                    note_confidences.append(max_probs[i])
                else:
                    # Pitch changed - save old note and start new
                    duration = (i - note_start) / self.frame_rate * 4  # Account for downsampling
                    if duration >= self.min_note_duration:
                        notes.append(Note(
                            pitch=int(note_pitch + self.min_midi),
                            start=float(note_start / self.frame_rate * 4),
                            duration=float(duration),
                            velocity=float(np.mean(note_confidences))
                        ))

                    # Start new note
                    note_start = i
                    note_pitch = max_pitches[i]
                    note_confidences = [max_probs[i]]

            elif not active[i] and in_note:
                # End note
                duration = (i - note_start) / self.frame_rate * 4
                if duration >= self.min_note_duration:
                    notes.append(Note(
                        pitch=int(note_pitch + self.min_midi),
                        start=float(note_start / self.frame_rate * 4),
                        duration=float(duration),
                        velocity=float(np.mean(note_confidences))
                    ))
                in_note = False

        # Handle final note
        if in_note:
            duration = (len(active) - note_start) / self.frame_rate * 4
            if duration >= self.min_note_duration:
                notes.append(Note(
                    pitch=int(note_pitch + self.min_midi),
                    start=float(note_start / self.frame_rate * 4),
                    duration=float(duration),
                    velocity=float(np.mean(note_confidences))
                ))

        return notes


def test_predictor():
    """Test the enhanced predictor."""
    print("\nTesting EnhancedMelodyPredictor...")

    import tempfile
    import soundfile as sf

    # Create test audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        audio_path = f.name

    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # A4
    sf.write(audio_path, audio, sr)

    try:
        # This will fail without a real checkpoint, but shows the API
        predictor = EnhancedMelodyPredictor(
            checkpoint_path="path/to/checkpoint.pth",
            frame_threshold=0.4,
            onset_threshold=0.5
        )

        notes = predictor.predict_from_file(audio_path)

        print(f"\n✅ Predicted {len(notes)} notes:")
        for note in notes:
            print(f"  MIDI {note.pitch} @ {note.start:.2f}s for {note.duration:.2f}s")

    except Exception as e:
        print(f"\n⚠️ Test failed (expected without checkpoint): {e}")

    finally:
        import os
        os.unlink(audio_path)

    print("\n✅ Predictor test complete!")


if __name__ == '__main__':
    test_predictor()