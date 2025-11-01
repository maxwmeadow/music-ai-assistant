"""
Melody Predictor - Production Inference Engine

Uses the combined Hum2Melody model for melody transcription from audio.
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

print("[PREDICTOR.PY] Initializing melody predictor module")

# Import model loader
try:
    from backend.models.combined_model_loader import load_combined_model
    print("[PREDICTOR.PY] ✅ Combined model loader imported")
except ImportError as e:
    print(f"[PREDICTOR.PY] ❌ Failed to import model loader: {e}")
    raise

# Import schemas
try:
    from backend.schemas import Track, Note, ChordEvent
    print("[PREDICTOR.PY] ✅ Schemas imported")
except ImportError as e:
    print(f"[PREDICTOR.PY] ❌ Failed to import schemas: {e}")
    raise

# Import music theory processor
try:
    from backend.music_theory import MusicTheoryProcessor
    print("[PREDICTOR.PY] ✅ Music theory processor imported")
except ImportError as e:
    print(f"[PREDICTOR.PY] ❌ Failed to import music theory processor: {e}")
    raise

print("[PREDICTOR.PY] All imports successful!")


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI number to note name."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"


class MelodyPredictor:
    """
    Production melody predictor using combined Hum2Melody model.

    Features:
        - CQT-based audio preprocessing
        - Combined pitch + onset/offset model
        - Configurable post-processing
        - Multiple prediction modes (standard, RAW)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        threshold: float = 0.4,
        onset_threshold: float = 0.5,
        min_note_duration: float = 0.12,
        merge_tolerance: float = 0.08,
        confidence_threshold: float = 0.3
    ):
        print(f"\n{'='*60}")
        print("MELODY PREDICTOR INITIALIZATION")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device or 'auto'}")
        print(f"Threshold: {threshold}")
        print(f"Onset threshold: {onset_threshold}")

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"✅ Checkpoint found ({self.checkpoint_path.stat().st_size / (1024*1024):.1f} MB)")

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Audio preprocessing parameters (must match training)
        self.sample_rate = 16000
        self.hop_length = 512
        self.n_bins = 88  # CQT bins for MIDI 21-108
        self.bins_per_octave = 12
        self.fmin = 27.5  # A0
        self.target_frames = 500
        self.min_midi = 21
        self.max_midi = 108

        # Post-processing parameters
        self.threshold = threshold
        self.onset_threshold = onset_threshold
        self.min_note_duration = min_note_duration
        self.merge_tolerance = merge_tolerance
        self.confidence_threshold = confidence_threshold

        # Frame rate (accounting for 4x CNN downsampling)
        self.frame_rate = (self.sample_rate / self.hop_length) / 4
        print(f"Frame rate: {self.frame_rate:.2f} fps (after CNN downsampling)")

        # Load combined model
        print("\nLoading combined model...")
        try:
            self.model = load_combined_model(
                str(self.checkpoint_path),
                device=str(self.device)
            )
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

        # Initialize music theory processor
        print("\nInitializing music theory processor...")
        try:
            self.music_processor = MusicTheoryProcessor()
            print("✅ Music theory processor initialized")
        except Exception as e:
            print(f"❌ Music theory processor initialization failed: {e}")
            raise

        print(f"{'='*60}\n")

    def predict_from_file(self, audio_path: str) -> Track:
        """Predict melody from audio file."""
        print(f"[predict_from_file] Loading: {audio_path}")

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path_obj, sr=self.sample_rate, mono=True)
        print(f"  Loaded {len(audio)} samples at {sr} Hz")

        return self.predict_from_audio(audio)

    def predict_from_audio(self, audio: np.ndarray) -> Track:
        """
        Predict melody from audio array with full music theory post-processing.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            Track with quantized Notes, metadata (key, tempo), and chord progression
        """
        print(f"[predict_from_audio] Processing {len(audio)} samples")

        # Preprocess to CQT
        cqt_tensor = self._preprocess_audio(audio)
        print(f"  CQT shape: {cqt_tensor.shape}")

        # Run inference
        with torch.no_grad():
            frame, onset, offset, f0 = self.model(cqt_tensor, None)

        # Convert to numpy
        frame_probs = torch.sigmoid(frame).cpu().numpy()[0]  # (time, 88)
        onset_probs = torch.sigmoid(onset).cpu().numpy()[0].squeeze()  # (time,)
        offset_probs = torch.sigmoid(offset).cpu().numpy()[0].squeeze()  # (time,)
        f0_output = f0.cpu().numpy()[0]  # (time, 2)

        print(f"  Frame probs: {frame_probs.shape}, range [{frame_probs.min():.3f}, {frame_probs.max():.3f}]")
        print(f"  Onset probs: {onset_probs.shape}, range [{onset_probs.min():.3f}, {onset_probs.max():.3f}]")
        print(f"  Offset probs: {offset_probs.shape}, range [{offset_probs.min():.3f}, {offset_probs.max():.3f}]")

        # ⭐ MUSIC THEORY PROCESSING
        # This pipeline:
        # 1. Cleans onsets/offsets (reduces ~60-70% over-prediction)
        # 2. Extracts raw notes
        # 3. Detects tempo
        # 4. Detects musical key (from raw, potentially off-key notes)
        # 5. Quantizes pitches to detected key scale
        # 6. Quantizes rhythm to tempo grid
        # 7. Infers chord progression
        print("\n⭐ Running music theory post-processing...")
        result = self.music_processor.process(
            frame_probs,
            onset_probs,
            offset_probs,
            frame_rate=self.frame_rate
        )

        # Convert to Note objects
        note_objects = [
            Note(
                pitch=int(n['pitch']),
                start=float(n['start']),
                duration=float(n['duration']),
                velocity=float(n.get('confidence', 0.8))
            )
            for n in result['notes']
        ]

        # Convert to ChordEvent objects
        chord_events = [
            ChordEvent(
                root=c['root'],
                quality=c['quality'],
                roman=c['roman'],
                start=float(c['start']),
                duration=float(c['duration'])
            )
            for c in result['harmony']
        ]

        print(f"\n✅ Final output:")
        print(f"  Notes: {len(note_objects)}")
        print(f"  Key: {result['metadata']['key']}")
        print(f"  Tempo: {result['metadata']['tempo']:.1f} BPM")
        print(f"  Chord progression: {len(chord_events)} chords")

        return Track(
            id="melody",
            instrument="lead_synth",
            notes=note_objects,
            samples=None,
            metadata=result['metadata'],
            harmony=chord_events
        )

    def predict_from_audio_RAW(self, audio: np.ndarray) -> Track:
        """
        RAW prediction with minimal post-processing.
        Useful for debugging and understanding model output.
        """
        print(f"[predict_from_audio_RAW] RAW MODE - minimal post-processing")

        # Preprocess
        cqt_tensor = self._preprocess_audio(audio)

        # Inference
        with torch.no_grad():
            frame, onset, offset, f0 = self.model(cqt_tensor, None)

        # Convert to numpy and apply sigmoid
        frame_probs = torch.sigmoid(frame).cpu().numpy()[0]
        onset_probs = torch.sigmoid(onset).cpu().numpy()[0].squeeze()

        # Simple thresholding
        num_frames = frame_probs.shape[0]
        notes = []

        for frame_idx in range(num_frames):
            # Get active notes in this frame
            active_notes = np.where(frame_probs[frame_idx] > self.threshold)[0]

            if len(active_notes) > 0:
                # Take the most confident note
                best_note_idx = active_notes[np.argmax(frame_probs[frame_idx][active_notes])]
                midi_note = best_note_idx + self.min_midi
                confidence = float(frame_probs[frame_idx][best_note_idx])

                # Time calculation
                start_time = frame_idx / self.frame_rate

                # Extend previous note or create new one
                if notes and notes[-1]['pitch'] == midi_note:
                    # Extend duration
                    notes[-1]['duration'] = start_time - notes[-1]['start'] + (1.0 / self.frame_rate)
                    notes[-1]['confidence'] = max(notes[-1]['confidence'], confidence)
                else:
                    # New note
                    notes.append({
                        'pitch': midi_note,
                        'start': start_time,
                        'duration': 1.0 / self.frame_rate,
                        'confidence': confidence
                    })

        print(f"  Generated {len(notes)} RAW notes")

        # Convert to Note objects
        note_objects = [
            Note(
                pitch=int(n['pitch']),
                start=float(n['start']),
                duration=float(n['duration']),
                velocity=float(n['confidence'])
            )
            for n in notes
        ]

        return Track(
            id="melody",
            instrument="lead_synth",
            notes=note_objects,
            samples=None
        )

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio to CQT spectrogram.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            CQT tensor (1, 1, time, n_bins)
        """
        # Extract CQT
        cqt = librosa.cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )

        # Convert to magnitude
        cqt_mag = np.abs(cqt)

        # Normalize (log scale)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        # Normalize to [0, 1]
        cqt_normalized = (cqt_db + 80) / 80
        cqt_normalized = np.clip(cqt_normalized, 0, 1)

        # Transpose to (time, freq) and pad/truncate
        cqt_normalized = cqt_normalized.T  # Now (time, n_bins)

        if cqt_normalized.shape[0] < self.target_frames:
            # Pad
            pad_width = self.target_frames - cqt_normalized.shape[0]
            cqt_normalized = np.pad(cqt_normalized, ((0, pad_width), (0, 0)), mode='constant')
        elif cqt_normalized.shape[0] > self.target_frames:
            # Truncate
            cqt_normalized = cqt_normalized[:self.target_frames, :]

        # Convert to tensor: (time, freq) -> (1, 1, time, freq)
        cqt_tensor = torch.FloatTensor(cqt_normalized).unsqueeze(0).unsqueeze(0)
        cqt_tensor = cqt_tensor.to(self.device)

        return cqt_tensor

    def _postprocess_predictions(
        self,
        frame_probs: np.ndarray,
        onset_probs: np.ndarray
    ) -> List[Note]:
        """
        Post-process model predictions to discrete notes.

        Args:
            frame_probs: (time, 88) frame probabilities
            onset_probs: (time,) onset probabilities

        Returns:
            List of Note objects
        """
        # Extract raw notes from frame predictions
        raw_notes = self._extract_notes_from_frames(frame_probs, onset_probs)
        print(f"    Raw notes: {len(raw_notes)}")

        # Merge nearby notes of same pitch
        merged_notes = self._merge_nearby_notes(raw_notes)
        print(f"    After merging: {len(merged_notes)}")

        # Filter by duration and confidence
        filtered_notes = self._filter_notes(merged_notes)
        print(f"    After filtering: {len(filtered_notes)}")

        # Resolve overlaps (monophonic constraint)
        resolved_notes = self._resolve_overlaps(filtered_notes)
        print(f"    After overlap resolution: {len(resolved_notes)}")

        # Convert to Note objects
        note_objects = [
            Note(
                pitch=int(n['pitch']),
                start=float(n['start']),
                duration=float(n['duration']),
                velocity=float(n['confidence'])
            )
            for n in resolved_notes
        ]

        # Sort by start time
        note_objects.sort(key=lambda x: x.start)

        return note_objects

    def _extract_notes_from_frames(
        self,
        frame_probs: np.ndarray,
        onset_probs: np.ndarray
    ) -> List[Dict]:
        """Extract note segments from frame-level predictions."""
        num_frames, num_notes = frame_probs.shape
        activations = frame_probs > self.threshold
        notes = []

        for note_idx in range(num_notes):
            note_activations = activations[:, note_idx]
            segments = self._find_segments(note_activations)

            for start_frame, end_frame in segments:
                start_time = start_frame / self.frame_rate
                end_time = end_frame / self.frame_rate
                duration = end_time - start_time
                confidence = float(np.mean(frame_probs[start_frame:end_frame, note_idx]))
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
                gap = next_note['start'] - (current['start'] + current['duration'])

                if (next_note['pitch'] == current['pitch'] and gap < self.merge_tolerance):
                    # Merge
                    current['duration'] = (next_note['start'] + next_note['duration']) - current['start']
                    current['confidence'] = max(current['confidence'], next_note['confidence'])
                    j += 1
                else:
                    break

            merged.append(current)
            i = j if j > i + 1 else i + 1

        return merged

    def _filter_notes(self, notes: List[Dict]) -> List[Dict]:
        """Filter notes by duration and confidence."""
        return [
            n for n in notes
            if n['duration'] >= self.min_note_duration
            and n['confidence'] >= self.confidence_threshold
        ]

    def _resolve_overlaps(self, notes: List[Dict]) -> List[Dict]:
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

            # Keep most confident note from overlapping group
            best = max(overlapping, key=lambda x: x['confidence'])
            resolved.append(best)
            i = j if j > i + 1 else i + 1

        return resolved


print("[PREDICTOR.PY] MelodyPredictor class defined successfully")
