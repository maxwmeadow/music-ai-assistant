"""
Model Server with EXTREME DEBUGGING for proper integration of trained Hum2Melody model
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import random
import sys
import traceback

print("[MODEL_SERVER.PY] ========================================")
print("[MODEL_SERVER.PY] Starting model_server.py import")
print(f"[MODEL_SERVER.PY] Python version: {sys.version}")
print(f"[MODEL_SERVER.PY] Current working directory: {Path.cwd()}")
print(f"[MODEL_SERVER.PY] sys.path (first 3): {sys.path[:3]}")
print("[MODEL_SERVER.PY] ========================================")

# Import schemas
print("[MODEL_SERVER.PY] Importing schemas...")
try:
    from .schemas import IR, Track, Note, SampleEvent
    print("[MODEL_SERVER.PY]   ✅ Schemas imported successfully")
except ImportError as e:
    print(f"[MODEL_SERVER.PY]   ❌ Failed to import schemas: {e}")
    traceback.print_exc()
    raise

# Import your trained model wrapper
print("[MODEL_SERVER.PY] Attempting to import MelodyPredictor...")
print("[MODEL_SERVER.PY]   Trying: from .inference.predictor import MelodyPredictor")

MelodyPredictor = None
try:
    from .inference.predictor import MelodyPredictor
    print("[MODEL_SERVER.PY]   ✅ SUCCESS - MelodyPredictor imported!")
    print(f"[MODEL_SERVER.PY]   MelodyPredictor type: {type(MelodyPredictor)}")
except ImportError as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - ImportError: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   MelodyPredictor will be None")
    MelodyPredictor = None
except Exception as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - Unexpected error: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   MelodyPredictor will be None")
    MelodyPredictor = None

print(f"[MODEL_SERVER.PY] Final MelodyPredictor status: {MelodyPredictor is not None}")
print("[MODEL_SERVER.PY] ========================================")


class ModelServer:
    """
    Serves AI model predictions for music generation.
    Now integrates the trained model if available, but falls back to mock logic.
    """

    def __init__(self):
        """Initialize model server and optionally load trained model"""
        print("[ModelServer.__init__] ========================================")
        print("[ModelServer.__init__] Initializing ModelServer")

        checkpoint_path = Path("backend/checkpoints/combined_hum2melody_full.pth")
        print(f"[ModelServer.__init__]   Checkpoint path: {checkpoint_path}")
        print(f"[ModelServer.__init__]   Checkpoint path (absolute): {checkpoint_path.absolute()}")
        print(f"[ModelServer.__init__]   Checkpoint exists: {checkpoint_path.exists()}")

        if checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"[ModelServer.__init__]   Checkpoint size: {size_mb:.1f} MB")

        print(f"[ModelServer.__init__]   MelodyPredictor is None: {MelodyPredictor is None}")
        print(f"[ModelServer.__init__]   Checkpoint exists: {checkpoint_path.exists()}")

        if MelodyPredictor is not None and checkpoint_path.exists():
            print("[ModelServer.__init__] ✅ Both conditions met - attempting to load model")
            try:
                print(f"[ModelServer.__init__]   Creating MelodyPredictor instance...")
                print(f"[ModelServer.__init__]   Args: checkpoint={checkpoint_path}, threshold=0.5, min_note_duration=0.1")

                self.predictor = MelodyPredictor(
                    str(checkpoint_path),
                    threshold=0.5,
                    min_note_duration=0.1
                )
                print(f"[ModelServer.__init__]   ✅ Model loaded successfully!")
                print(f"[ModelServer.__init__]   Predictor device: {self.predictor.device}")
            except Exception as e:
                print(f"[ModelServer.__init__]   ❌ Failed to load model: {e}")
                print("[ModelServer.__init__]   Full traceback:")
                traceback.print_exc()
                self.predictor = None
        else:
            print("[ModelServer.__init__] ❌ Conditions NOT met for loading model:")
            self.predictor = None
            if not checkpoint_path.exists():
                print(f"[ModelServer.__init__]   ❌ No trained model found at {checkpoint_path}")
            if MelodyPredictor is None:
                print("[ModelServer.__init__]   ❌ MelodyPredictor not available (import failed)")
            print("[ModelServer.__init__]   Using mock predictions")

        # Define scales for fallback
        self.c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]
        self.a_minor_scale = [57, 59, 60, 62, 64, 65, 67, 69]

        print(f"[ModelServer.__init__] Final predictor status: {self.predictor is not None}")
        print("[ModelServer.__init__] ========================================")

    async def predict_melody(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict melody from humming audio features.
        Uses trained model if available; otherwise falls back to mock predictions.
        """
        print("[ModelServer.predict_melody] ========================================")
        print(f"[ModelServer.predict_melody] Called with audio_features keys: {list(audio_features.keys())}")
        print(f"[ModelServer.predict_melody] Predictor is None: {self.predictor is None}")

        # Handle invalid or missing inputs
        if not audio_features:
            print("[ModelServer.predict_melody] ⚠️ Missing audio features, using mock.")
            return self._mock_melody(audio_features)

        # Try real model if available
        if self.predictor is not None:
            print("[ModelServer.predict_melody] ✅ Predictor available - attempting real inference")
            try:
                # Priority 1: Raw audio bytes (best for API uploads)
                audio_bytes = audio_features.get("audio_bytes")
                if audio_bytes:
                    print(f"[ModelServer.predict_melody]   Using raw audio bytes ({len(audio_bytes)} bytes)")
                    print(f"[ModelServer.predict_melody]   🔴 USING RAW PREDICTION MODE - NO POST-PROCESSING")

                    # Load audio from bytes
                    import io
                    import librosa
                    audio_file = io.BytesIO(audio_bytes)
                    audio, sr = librosa.load(audio_file, sr=16000, mono=True)

                    # Use RAW prediction
                    track = self.predictor.predict_from_audio_RAW(audio)
                    if track and getattr(track, "notes", None):
                        print(f"[ModelServer.predict_melody]   ✅ Generated {len(track.notes)} RAW notes from trained model")
                        return track

                # Priority 2: Audio array (if already loaded)
                audio = audio_features.get("audio")
                if audio is not None and isinstance(audio, np.ndarray):
                    print(f"[ModelServer.predict_melody]   Using audio array (shape: {audio.shape})")
                    track = self.predictor.predict_from_audio(audio)
                    if track and getattr(track, "notes", None):
                        print(f"[ModelServer.predict_melody]   ✅ Generated {len(track.notes)} notes from trained model")
                        return track

                # Priority 3: File path
                audio_path = audio_features.get("audio_path")
                if audio_path:
                    print(f"[ModelServer.predict_melody]   Using audio file path: {audio_path}")
                    track = self.predictor.predict_from_file(audio_path)
                    if track and getattr(track, "notes", None):
                        print(f"[ModelServer.predict_melody]   ✅ Generated {len(track.notes)} notes from trained model")
                        return track

                # If we get here, no valid audio format was provided
                print("[ModelServer.predict_melody]   ⚠️ No valid audio format (need audio_bytes, audio array, or audio_path)")
                return self._mock_melody(audio_features)

            except Exception as e:
                print(f"[ModelServer.predict_melody]   ❌ Model inference failed: {e}")
                print("[ModelServer.predict_melody]   Full traceback:")
                traceback.print_exc()
                return self._mock_melody(audio_features)
        else:
            # Fallback to mock predictions
            print("[ModelServer.predict_melody] ❌ No trained model loaded, using mock.")
            return self._mock_melody(audio_features)


    async def predict_drums(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict drum pattern from beatbox audio features (mock only for now).
        """
        onset_times = audio_features.get("onset_times", [])
        tempo = audio_features.get("tempo", 120)

        samples = []
        if not onset_times:
            beat_interval = 60.0 / tempo
            onset_times = [i * beat_interval for i in range(16)]

        for i, start_time in enumerate(onset_times):
            if i % 4 == 0:
                sample_type = "kick"
            elif i % 2 == 1:
                sample_type = "snare"
            else:
                sample_type = "hihat"
            samples.append(SampleEvent(sample=sample_type, start=float(start_time)))

        return Track(id="drums", instrument=None, notes=None, samples=samples)


    async def arrange_track(self, existing_ir: IR, style: str = "pop") -> IR:
        """
        Take existing melody and add accompanying tracks (bass, chords, etc).
        """
        melody_track = next((t for t in existing_ir.tracks if t.notes), None)
        if not melody_track or not melody_track.notes:
            return existing_ir

        # Generate bass (use start times from melody)
        bass_notes = []
        for i, note in enumerate(melody_track.notes[::2]):
            bass_notes.append(
                Note(
                    pitch=note.pitch - 12,
                    start=note.start,
                    duration=note.duration * 2,
                    velocity=0.8,
                )
            )

        bass_track = Track(
            id="bass", instrument="bass_synth", notes=bass_notes, samples=None
        )

        # Chords (every 2 seconds)
        chord_notes = []
        chord_progression = [60, 65, 67, 62]
        for i, chord_root in enumerate(chord_progression):
            start_time = i * 2.0
            for offset in [0, 4, 7]:
                chord_notes.append(
                    Note(
                        pitch=chord_root + offset,
                        start=start_time,
                        duration=2.0,
                        velocity=0.5,
                    )
                )

        chord_track = Track(
            id="chords", instrument="pad_synth", notes=chord_notes, samples=None
        )

        new_tracks = existing_ir.tracks + [bass_track, chord_track]
        return IR(metadata=existing_ir.metadata, tracks=new_tracks)


    def _mock_melody(self, audio_features: Dict[str, Any]) -> Track:
        """Fallback melody generator for testing."""
        print("[ModelServer._mock_melody] Using mock melody generator")

        onset_times = audio_features.get("onset_times", [])
        duration = audio_features.get("duration", 4.0)
        notes = []

        if not onset_times:
            onset_times = [i * 0.5 for i in range(8)]

        for i, start_time in enumerate(onset_times):
            pitch = self.a_minor_scale[i % len(self.a_minor_scale)]
            if i < len(onset_times) - 1:
                note_duration = onset_times[i + 1] - start_time
            else:
                note_duration = min(1.0, duration - start_time)
            velocity = random.uniform(0.6, 0.9)

            notes.append(Note(
                pitch=pitch,
                start=start_time,
                duration=note_duration,
                velocity=velocity
            ))

        print(f"[ModelServer._mock_melody]   Generated {len(notes)} mock notes")
        return Track(
            id="melody", instrument="guitar/rjs_guitar_new_strings", notes=notes, samples=None
        )


print("[MODEL_SERVER.PY] Class definition complete")
print("[MODEL_SERVER.PY] ========================================")