"""
Improved Predictor with EXTREME DEBUGGING

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
import sys
import traceback

print("[PREDICTOR.PY] ========================================")
print("[PREDICTOR.PY] Starting predictor.py import")
print(f"[PREDICTOR.PY] Python version: {sys.version}")
print(f"[PREDICTOR.PY] Current working directory: {Path.cwd()}")
print(f"[PREDICTOR.PY] sys.path: {sys.path[:3]}...")  # First 3 entries
print("[PREDICTOR.PY] ========================================")

# Try to import model
print("[PREDICTOR.PY] Attempting to import ImprovedHum2MelodyCRNN...")
try:
    print("[PREDICTOR.PY]   Trying: from backend.models.hum2melody_model import ImprovedHum2MelodyCRNN")
    from backend.models.hum2melody_model import ImprovedHum2MelodyCRNN
    print("[PREDICTOR.PY]   ✅ SUCCESS - ImprovedHum2MelodyCRNN imported")
except ImportError as e:
    print(f"[PREDICTOR.PY]   ❌ FAILED - ImportError: {e}")
    print(f"[PREDICTOR.PY]   Traceback:")
    traceback.print_exc()
    raise

# Try to import schemas
print("[PREDICTOR.PY] Attempting to import Track, Note...")
try:
    print("[PREDICTOR.PY]   Trying: from backend.schemas import Track, Note")
    from backend.schemas import Track, Note
    print("[PREDICTOR.PY]   ✅ SUCCESS - Track, Note imported")
except ImportError as e:
    print(f"[PREDICTOR.PY]   ❌ FAILED - ImportError: {e}")
    print(f"[PREDICTOR.PY]   Traceback:")
    traceback.print_exc()
    raise

print("[PREDICTOR.PY] All imports successful!")
print("[PREDICTOR.PY] ========================================")


class ImprovedMelodyPredictor:
    """
    Improved inference engine with better post-processing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.4,
        min_note_duration: float = 0.12,
        merge_tolerance: float = 0.08,
        confidence_threshold: float = 0.3
    ):
        print("[PREDICTOR.__init__] ========================================")
        print("[PREDICTOR.__init__] Initializing ImprovedMelodyPredictor")
        print(f"[PREDICTOR.__init__]   checkpoint_path: {checkpoint_path}")
        print(f"[PREDICTOR.__init__]   device: {device}")
        print(f"[PREDICTOR.__init__]   threshold: {threshold}")

        self.checkpoint_path = Path(checkpoint_path)
        print(f"[PREDICTOR.__init__]   checkpoint_path (Path): {self.checkpoint_path}")
        print(f"[PREDICTOR.__init__]   checkpoint exists: {self.checkpoint_path.exists()}")

        if self.checkpoint_path.exists():
            size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"[PREDICTOR.__init__]   checkpoint size: {size_mb:.1f} MB")
        else:
            print(f"[PREDICTOR.__init__]   ❌ CHECKPOINT NOT FOUND!")
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Setup device
        if device:
            self.device = torch.device(device)
            print(f"[PREDICTOR.__init__]   Using specified device: {self.device}")
        else:
            cuda_available = torch.cuda.is_available()
            print(f"[PREDICTOR.__init__]   CUDA available: {cuda_available}")
            self.device = torch.device('cuda' if cuda_available else 'cpu')
            print(f"[PREDICTOR.__init__]   Using device: {self.device}")

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
        self.frame_rate = self.sample_rate / self.hop_length
        print(f"[PREDICTOR.__init__]   frame_rate: {self.frame_rate} fps")

        # Load model
        print("[PREDICTOR.__init__]   Loading model...")
        try:
            self.model = self._load_model()
            print("[PREDICTOR.__init__]   ✅ Model loaded successfully")
        except Exception as e:
            print(f"[PREDICTOR.__init__]   ❌ Model loading failed: {e}")
            traceback.print_exc()
            raise

        print(f"[PREDICTOR.__init__]   Threshold: {threshold}")
        print(f"[PREDICTOR.__init__]   Min note duration: {min_note_duration}s")
        print(f"[PREDICTOR.__init__]   Merge tolerance: {merge_tolerance}s")
        print("[PREDICTOR.__init__] ========================================")

    def _load_model(self) -> ImprovedHum2MelodyCRNN:
        """Load improved model from checkpoint."""
        print("[PREDICTOR._load_model] ========================================")
        print(f"[PREDICTOR._load_model] Loading checkpoint from: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Create model
        print("[PREDICTOR._load_model] Creating ImprovedHum2MelodyCRNN instance...")
        try:
            model = ImprovedHum2MelodyCRNN()
            print("[PREDICTOR._load_model]   ✅ Model instance created")
        except Exception as e:
            print(f"[PREDICTOR._load_model]   ❌ Failed to create model: {e}")
            traceback.print_exc()
            raise

        # Load checkpoint
        print(f"[PREDICTOR._load_model] Loading checkpoint file...")
        try:
            # Use weights_only=False for compatibility with PyTorch 2.6+
            # This is safe because we trust our own trained checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            print(f"[PREDICTOR._load_model]   ✅ Checkpoint loaded")
            print(f"[PREDICTOR._load_model]   Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            print(f"[PREDICTOR._load_model]   ❌ Failed to load checkpoint: {e}")
            traceback.print_exc()
            raise

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            print("[PREDICTOR._load_model] Checkpoint format: training checkpoint with metadata")
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("[PREDICTOR._load_model]   ✅ State dict loaded")
            except Exception as e:
                print(f"[PREDICTOR._load_model]   ❌ Failed to load state dict: {e}")
                traceback.print_exc()
                raise

            epoch = checkpoint.get('epoch', 'unknown')
            val_metrics = checkpoint.get('val_metrics', {})
            print(f"[PREDICTOR._load_model]   Epoch: {epoch}")
            if 'f1' in val_metrics:
                print(f"[PREDICTOR._load_model]   Val F1: {val_metrics['f1']:.4f}")
        else:
            print("[PREDICTOR._load_model] Checkpoint format: raw state dict")
            try:
                model.load_state_dict(checkpoint)
                print("[PREDICTOR._load_model]   ✅ State dict loaded")
            except Exception as e:
                print(f"[PREDICTOR._load_model]   ❌ Failed to load state dict: {e}")
                traceback.print_exc()
                raise

        # Set to eval mode and move to device
        print(f"[PREDICTOR._load_model] Setting model to eval mode...")
        model.eval()
        print(f"[PREDICTOR._load_model] Moving model to device: {self.device}")
        model.to(self.device)
        print("[PREDICTOR._load_model]   ✅ Model ready")
        print("[PREDICTOR._load_model] ========================================")

        return model

    def predict_from_file(self, audio_path: str) -> Track:
        """Predict melody from audio file."""
        print(f"[PREDICTOR.predict_from_file] ========================================")
        print(f"[PREDICTOR.predict_from_file] Predicting from file: {audio_path}")

        audio_path_obj = Path(audio_path)
        print(f"[PREDICTOR.predict_from_file]   File exists: {audio_path_obj.exists()}")

        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        print(f"[PREDICTOR.predict_from_file] Loading audio...")
        try:
            audio, sr = librosa.load(audio_path_obj, sr=self.sample_rate, mono=True)
            print(f"[PREDICTOR.predict_from_file]   ✅ Audio loaded: {len(audio)} samples, {sr} Hz")
        except Exception as e:
            print(f"[PREDICTOR.predict_from_file]   ❌ Failed to load audio: {e}")
            traceback.print_exc()
            raise

        return self.predict_from_audio(audio)

    def predict_from_bytes(self, audio_bytes: bytes) -> Track:
        """Predict melody from audio bytes."""
        print(f"[PREDICTOR.predict_from_bytes] ========================================")
        print(f"[PREDICTOR.predict_from_bytes] Predicting from bytes: {len(audio_bytes)} bytes")

        import io

        # Load audio from bytes
        print(f"[PREDICTOR.predict_from_bytes] Loading audio from bytes...")
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            print(f"[PREDICTOR.predict_from_bytes]   ✅ Audio loaded: {len(audio)} samples, {sr} Hz")
        except Exception as e:
            print(f"[PREDICTOR.predict_from_bytes]   ❌ Failed to load audio: {e}")
            traceback.print_exc()
            raise

        return self.predict_from_audio(audio)

    def predict_from_audio(self, audio: np.ndarray) -> Track:
        """Predict melody from audio array."""
        print(f"[PREDICTOR.predict_from_audio] ========================================")
        print(f"[PREDICTOR.predict_from_audio] Predicting from audio array: {audio.shape}")

        # Preprocess audio
        print(f"[PREDICTOR.predict_from_audio] Preprocessing audio...")
        try:
            mel_spec = self._preprocess_audio(audio)
            print(f"[PREDICTOR.predict_from_audio]   ✅ Mel spec shape: {mel_spec.shape}")
        except Exception as e:
            print(f"[PREDICTOR.predict_from_audio]   ❌ Preprocessing failed: {e}")
            traceback.print_exc()
            raise

        # Run inference
        print(f"[PREDICTOR.predict_from_audio] Running inference...")
        try:
            probabilities = self._run_inference(mel_spec)
            print(f"[PREDICTOR.predict_from_audio]   ✅ Probabilities shape: {probabilities.shape}")
            print(f"[PREDICTOR.predict_from_audio]   Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        except Exception as e:
            print(f"[PREDICTOR.predict_from_audio]   ❌ Inference failed: {e}")
            traceback.print_exc()
            raise

        # Post-process to discrete notes
        print(f"[PREDICTOR.predict_from_audio] Post-processing predictions...")
        try:
            notes = self._postprocess_predictions(probabilities)
            print(f"[PREDICTOR.predict_from_audio]   ✅ Generated {len(notes)} notes")
        except Exception as e:
            print(f"[PREDICTOR.predict_from_audio]   ❌ Post-processing failed: {e}")
            traceback.print_exc()
            raise

        # Create Track
        print(f"[PREDICTOR.predict_from_audio] Creating Track object...")
        track = Track(
            id="melody",
            instrument="lead_synth",
            notes=notes,
            samples=None
        )
        print(f"[PREDICTOR.predict_from_audio]   ✅ Track created with {len(notes)} notes")
        print(f"[PREDICTOR.predict_from_audio] ========================================")

        return track

    def predict_from_audio_RAW(self, audio: np.ndarray) -> Track:
        """
        RAW prediction - NO POST-PROCESSING!
        Just converts model output directly to notes with minimal filtering.
        """
        print(f"[PREDICTOR.RAW] ========================================")
        print(f"[PREDICTOR.RAW] RAW PREDICTION MODE - NO POST-PROCESSING")
        print(f"[PREDICTOR.RAW] Audio shape: {audio.shape}")

        # Preprocess audio
        mel_spec = self._preprocess_audio(audio)
        print(f"[PREDICTOR.RAW] Mel spec shape: {mel_spec.shape}")

        # Run inference
        probabilities = self._run_inference(mel_spec)
        print(f"[PREDICTOR.RAW] Probabilities shape: {probabilities.shape}")
        print(f"[PREDICTOR.RAW] Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

        # RAW conversion - just threshold at 0.3 and convert directly to notes
        num_frames, num_notes = probabilities.shape
        threshold = 0.3  # Low threshold to see everything

        notes = []

        # For each frame, find notes above threshold
        print(f"[PREDICTOR.RAW] Converting to notes (threshold={threshold})...")
        for frame_idx in range(num_frames):
            frame_probs = probabilities[frame_idx]
            active_notes = np.where(frame_probs > threshold)[0]

            if len(active_notes) > 0:
                # Get the loudest note in this frame
                loudest_note_idx = active_notes[np.argmax(frame_probs[active_notes])]
                midi_note = loudest_note_idx + self.min_midi
                confidence = float(frame_probs[loudest_note_idx])

                # Time calculation (account for 4x downsampling)
                start_time = frame_idx / self.frame_rate * 4

                # Check if we should extend previous note or create new one
                if notes and notes[-1]['pitch'] == midi_note and start_time - (notes[-1]['start'] + notes[-1]['duration']) < 0.2:
                    # Extend previous note
                    notes[-1]['duration'] = start_time - notes[-1]['start'] + (1.0 / self.frame_rate * 4)
                    notes[-1]['confidence'] = max(notes[-1]['confidence'], confidence)
                else:
                    # New note
                    notes.append({
                        'pitch': midi_note,
                        'start': start_time,
                        'duration': 1.0 / self.frame_rate * 4,  # Single frame duration
                        'confidence': confidence
                    })

        print(f"[PREDICTOR.RAW] Generated {len(notes)} RAW notes")

        # Show first 10 notes for debugging
        print(f"[PREDICTOR.RAW] First 10 notes:")
        for i, note in enumerate(notes[:10]):
            note_name = self._midi_to_note_name(note['pitch'])
            print(f"[PREDICTOR.RAW]   {i+1}. {note_name} @ {note['start']:.3f}s for {note['duration']:.3f}s (conf: {note['confidence']:.3f})")

        # Convert to Note objects
        note_objects = []
        for note in notes:
            note_objects.append(Note(
                pitch=int(note['pitch']),
                start=float(note['start']),
                duration=float(note['duration']),
                velocity=float(note['confidence'])
            ))

        track = Track(
            id="melody",
            instrument="lead_synth",
            notes=note_objects,
            samples=None
        )

        print(f"[PREDICTOR.RAW] ========================================")
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
            logits = self.model(mel_spec)

            # Apply sigmoid
            probabilities = torch.sigmoid(logits)

            # Convert to numpy
            probabilities = probabilities.squeeze(0).cpu().numpy()

        return probabilities

    def _postprocess_predictions(self, probabilities: np.ndarray) -> List[Note]:
        """Post-process predictions."""
        # Extract raw notes
        raw_notes = self._extract_notes_from_probabilities(probabilities)
        print(f"[PREDICTOR._postprocess] Raw notes: {len(raw_notes)}")

        # Merge nearby notes of same pitch
        merged_notes = self._merge_nearby_notes(raw_notes)
        print(f"[PREDICTOR._postprocess] After merging nearby: {len(merged_notes)}")

        # Merge onset notes
        cleaned_notes = self._merge_onset_notes(merged_notes)
        print(f"[PREDICTOR._postprocess] After merging onsets: {len(cleaned_notes)}")

        # Resolve overlaps
        resolved_notes = self._resolve_overlapping_notes(cleaned_notes)
        print(f"[PREDICTOR._postprocess] After resolving overlaps: {len(resolved_notes)}")

        # Filter
        filtered_notes = self._filter_notes(resolved_notes)
        print(f"[PREDICTOR._postprocess] After filtering: {len(filtered_notes)}")

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

    def _extract_notes_from_probabilities(self, probabilities: np.ndarray) -> List[Dict]:
        """Extract raw notes from probability matrix."""
        num_frames, num_notes = probabilities.shape
        activations = probabilities > self.threshold
        notes = []

        for note_idx in range(num_notes):
            note_activations = activations[:, note_idx]
            note_probs = probabilities[:, note_idx]
            segments = self._find_segments(note_activations)

            for start_frame, end_frame in segments:
                start_time = start_frame / self.frame_rate * 4
                end_time = end_frame / self.frame_rate * 4
                duration = end_time - start_time
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

        notes.sort(key=lambda x: (x['pitch'], x['start']))
        merged = []
        i = 0

        while i < len(notes):
            current = notes[i].copy()
            j = i + 1

            while j < len(notes):
                next_note = notes[j]
                if (next_note['pitch'] == current['pitch'] and
                    next_note['start'] - (current['start'] + current['duration']) < self.merge_tolerance):
                    current['duration'] = (next_note['start'] + next_note['duration']) - current['start']
                    current['confidence'] = max(current['confidence'], next_note['confidence'])
                    j += 1
                else:
                    break

            merged.append(current)
            i = j if j > i + 1 else i + 1

        return merged

    def _resolve_overlapping_notes(self, notes: List[Dict]) -> List[Dict]:
        """Resolve overlapping notes by keeping most confident."""
        if not notes:
            return notes

        notes.sort(key=lambda x: x['start'])
        resolved = []
        i = 0

        while i < len(notes):
            current = notes[i]
            overlapping = [current]
            j = i + 1

            while j < len(notes):
                next_note = notes[j]
                if next_note['start'] < current['start'] + current['duration']:
                    overlapping.append(next_note)
                    j += 1
                else:
                    break

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
            if note['duration'] < self.min_note_duration:
                continue
            if note['confidence'] < self.confidence_threshold:
                continue
            filtered.append(note)

        return filtered

    def _merge_onset_notes(self, notes: List[Dict]) -> List[Dict]:
        """Merge tiny 'onset' notes with the sustained note that follows."""
        if not notes:
            return notes

        notes.sort(key=lambda x: x['start'])
        merged = []
        i = 0

        while i < len(notes):
            current = notes[i]

            if (i < len(notes) - 1 and current['duration'] < 0.08):
                next_note = notes[i + 1]
                gap = next_note['start'] - (current['start'] + current['duration'])
                pitch_diff = abs(next_note['pitch'] - current['pitch'])

                if gap < 0.1 and pitch_diff <= 3:
                    merged_note = {
                        'pitch': next_note['pitch'],
                        'start': current['start'],
                        'duration': (next_note['start'] + next_note['duration']) - current['start'],
                        'confidence': max(current['confidence'], next_note['confidence'])
                    }
                    merged.append(merged_note)
                    i += 2
                    continue

            merged.append(current)
            i += 1

        return merged


print("[PREDICTOR.PY] Class definition complete")
print("[PREDICTOR.PY] ========================================")