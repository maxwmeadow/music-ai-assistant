"""
Improved Predictor with better post-processing.

Key improvements:
1. Works with new 125-frame model
2. Better note merging
3. Overlap resolution (monophonic constraint)
4. Configurable thresholds
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Optional


from ..models.hum2melody_model import ImprovedHum2MelodyCRNN
from ..schemas import Track, Note


class ImprovedMelodyPredictor:
    """
    Improved inference engine with better post-processing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.4,  # Lower threshold
        min_note_duration: float = 0.12,  # Slightly shorter
        merge_tolerance: float = 0.08,  # Merge nearby notes
        confidence_threshold: float = 0.3  # Filter low confidence
    ):
        self.checkpoint_path = Path(checkpoint_path)

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[ImprovedPredictor] Using device: {self.device}")

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
        self.merge_tolerance = merge_tolerance
        self.confidence_threshold = confidence_threshold

        # Frame rate
        self.frame_rate = self.sample_rate / self.hop_length  # 31.25 fps

        # Load model
        self.model = self._load_model()

        print(f"[ImprovedPredictor] Model loaded successfully")
        print(f"  Threshold: {threshold}")
        print(f"  Min note duration: {min_note_duration}s")
        print(f"  Merge tolerance: {merge_tolerance}s")

    def _load_model(self) -> ImprovedHum2MelodyCRNN:
        """Load improved model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create model
        model = ImprovedHum2MelodyCRNN()

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_metrics = checkpoint.get('val_metrics', {})
            print(f"[ImprovedPredictor] Loaded from epoch {epoch}")
            if 'f1' in val_metrics:
                print(f"  Val F1: {val_metrics['f1']:.4f}")
        else:
            model.load_state_dict(checkpoint)

        # Set to eval mode and move to device
        model.eval()
        model.to(self.device)

        return model

    def predict_from_file(self, audio_path: str) -> Track:
        """Predict melody from audio file."""
        audio_path_obj = Path(audio_path)

        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path_obj, sr=self.sample_rate, mono=True)

        return self.predict_from_audio(audio)

    def predict_from_bytes(self, audio_bytes: bytes) -> Track:
        """Predict melody from audio bytes."""
        import io

        # Load audio from bytes
        audio_file = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)

        return self.predict_from_audio(audio)

    def predict_from_audio(self, audio: np.ndarray) -> Track:
        """Predict melody from audio array."""
        # Preprocess audio
        mel_spec = self._preprocess_audio(audio)

        # Run inference
        probabilities = self._run_inference(mel_spec)

        # Post-process to discrete notes
        notes = self._postprocess_predictions(probabilities)

        # Create Track
        track = Track(
            id="melody",
            instrument="lead_synth",
            notes=notes,
            samples=None
        )

        return track

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Preprocess audio to mel spectrogram."""
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

        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_normalized = (mel_spec_db + 80) / 80
        mel_spec_normalized = np.clip(mel_spec_normalized, 0, 1)

        # Pad or truncate
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
        """Run model inference."""
        mel_spec = mel_spec.to(self.device)

        with torch.no_grad():
            # Forward pass
            logits = self.model(mel_spec)  # (1, 125, 88)

            # Apply sigmoid
            probabilities = torch.sigmoid(logits)

            # Convert to numpy
            probabilities = probabilities.squeeze(0).cpu().numpy()  # (125, 88)

        return probabilities

    def _postprocess_predictions(self, probabilities: np.ndarray) -> List[Note]:
        """
        Post-process predictions with improved logic.

        Steps:
        1. Extract raw notes from probabilities
        2. Merge nearby notes of same pitch
        3. Merge tiny onset notes with sustained notes ← NEW!
        4. Resolve overlapping notes (monophonic constraint)
        5. Filter by confidence and duration
        """
        # Step 1: Extract raw notes
        raw_notes = self._extract_notes_from_probabilities(probabilities)

        # Step 2: Merge nearby notes of same pitch
        merged_notes = self._merge_nearby_notes(raw_notes)

        # Step 3: Merge onset notes ← ADD THIS LINE
        cleaned_notes = self._merge_onset_notes(merged_notes)

        # Step 4: Resolve overlaps (keep most confident)
        resolved_notes = self._resolve_overlapping_notes(cleaned_notes)  # ← Use cleaned_notes

        # Step 5: Filter by confidence and duration
        filtered_notes = self._filter_notes(resolved_notes)

        # Convert to Note objects
        note_objects = []
        for note in filtered_notes:
            note_objects.append(Note(
                pitch=int(note['pitch']),
                start=float(note['start']),
                duration=float(note['duration']),
                velocity=float(note['confidence'])
            ))

        # Sort by start time
        note_objects.sort(key=lambda x: x.start)

        return note_objects

    def _extract_notes_from_probabilities(
        self,
        probabilities: np.ndarray
    ) -> List[Dict]:
        """Extract raw notes from probability matrix."""
        num_frames, num_notes = probabilities.shape

        # Apply threshold
        activations = probabilities > self.threshold

        notes = []

        # For each MIDI note
        for note_idx in range(num_notes):
            note_activations = activations[:, note_idx]
            note_probs = probabilities[:, note_idx]

            # Find continuous segments
            segments = self._find_segments(note_activations)

            # Convert to notes
            for start_frame, end_frame in segments:
                start_time = start_frame / self.frame_rate * 4  # Account for 4x downsampling
                end_time = end_frame / self.frame_rate * 4
                duration = end_time - start_time

                # Calculate confidence
                confidence = float(np.mean(note_probs[start_frame:end_frame]))

                midi_note = note_idx + self.min_midi

                notes.append({
                    'pitch': midi_note,
                    'start': start_time,
                    'duration': duration,
                    'confidence': confidence
                })

        return notes

    def _find_segments(self, activations: np.ndarray) -> List[tuple]:
        """Find continuous segments where note is active."""
        segments = []
        in_segment = False
        start_frame = 0

        for frame_idx in range(len(activations)):
            if activations[frame_idx] and not in_segment:
                in_segment = True
                start_frame = frame_idx
            elif not activations[frame_idx] and in_segment:
                in_segment = False
                segments.append((start_frame, frame_idx))

        if in_segment:
            segments.append((start_frame, len(activations)))

        return segments

    def _merge_nearby_notes(self, notes: List[Dict]) -> List[Dict]:
        """Merge notes of same pitch that are close in time."""
        if not notes:
            return notes

        # Sort by pitch then start time
        notes.sort(key=lambda x: (x['pitch'], x['start']))

        merged = []
        i = 0

        while i < len(notes):
            current = notes[i].copy()

            # Look for mergeable notes
            j = i + 1
            while j < len(notes):
                next_note = notes[j]

                # Same pitch and close in time?
                if (next_note['pitch'] == current['pitch'] and
                    next_note['start'] - (current['start'] + current['duration']) < self.merge_tolerance):
                    # Merge
                    current['duration'] = (next_note['start'] + next_note['duration']) - current['start']
                    current['confidence'] = max(current['confidence'], next_note['confidence'])
                    j += 1
                else:
                    break

            merged.append(current)
            i = j if j > i + 1 else i + 1

        return merged

    def _resolve_overlapping_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Resolve overlapping notes by keeping most confident.
        This enforces monophonic constraint.
        """
        if not notes:
            return notes

        notes.sort(key=lambda x: x['start'])

        resolved = []
        i = 0

        while i < len(notes):
            current = notes[i]

            # Find all overlapping notes
            overlapping = [current]
            j = i + 1

            while j < len(notes):
                next_note = notes[j]

                # Check if overlaps
                if next_note['start'] < current['start'] + current['duration']:
                    overlapping.append(next_note)
                    j += 1
                else:
                    break

            # If multiple overlapping, keep most confident
            if len(overlapping) > 1:
                best = max(overlapping, key=lambda x: x['confidence'])
                resolved.append(best)
                i = j
            else:
                resolved.append(current)
                i += 1

        return resolved

    def _filter_notes(self, notes: List[Dict]) -> List[Dict]:
        """Filter notes by duration and confidence."""
        filtered = []

        for note in notes:
            # Filter by duration
            if note['duration'] < self.min_note_duration:
                continue

            # Filter by confidence
            if note['confidence'] < self.confidence_threshold:
                continue

            filtered.append(note)

        return filtered

    def _merge_onset_notes(self, notes: List[Dict]) -> List[Dict]:
        """
        Merge tiny 'onset' notes with the sustained note that follows.

        This fixes the issue where the model detects:
        - A tiny note (0.03s) at the attack
        - Followed immediately by the sustained note

        Example:
        Before: [A3 @ 0.288s for 0.032s], [B3 @ 0.32s for 0.352s]
        After:  [B3 @ 0.288s for 0.384s]
        """
        if not notes:
            return notes

        # Sort by start time
        notes.sort(key=lambda x: x['start'])

        merged = []
        i = 0

        while i < len(notes):
            current = notes[i]

            # Check if this is a tiny note (likely an onset detection)
            if (i < len(notes) - 1 and
                    current['duration'] < 0.08):  # Very short note

                next_note = notes[i + 1]

                # Calculate gap between notes
                gap = next_note['start'] - (current['start'] + current['duration'])

                # Calculate pitch difference in semitones
                pitch_diff = abs(next_note['pitch'] - current['pitch'])

                # Merge if:
                # 1. Gap is very small (< 0.1s)
                # 2. Pitches are close (within 3 semitones - could be vibrato/glide)
                if gap < 0.1 and pitch_diff <= 3:
                    # Merge into one note starting from onset
                    merged_note = {
                        'pitch': next_note['pitch'],  # Use sustained pitch
                        'start': current['start'],  # Start from onset
                        'duration': (next_note['start'] + next_note['duration']) - current['start'],
                        'confidence': max(current['confidence'], next_note['confidence'])
                    }
                    merged.append(merged_note)
                    i += 2  # Skip both notes
                    continue

            # If not merged, keep current note
            merged.append(current)
            i += 1

        return merged

    def _midi_to_note_name(self, midi: int) -> str:
        """Convert MIDI to note name."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = notes[midi % 12]
        return f"{note}{octave}"


if __name__ == '__main__':
    # Test predictor
    print("Testing ImprovedMelodyPredictor...")

    checkpoint = "checkpoints_v2/best_model.pth"
    if Path(checkpoint).exists():
        predictor = ImprovedMelodyPredictor(checkpoint)
        print("✅ Predictor loaded successfully!")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint}")
        print("   Train the model first!")