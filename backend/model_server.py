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

# Import the new hum2melody package
print("[MODEL_SERVER.PY] Attempting to import ChunkedHybridHum2Melody...")
print("[MODEL_SERVER.PY]   Trying: from .hum2melody.inference.hybrid_inference_chunked import ChunkedHybridHum2Melody")

ChunkedHybridHum2Melody = None
try:
    from .hum2melody.inference.hybrid_inference_chunked import ChunkedHybridHum2Melody
    print("[MODEL_SERVER.PY]   ✅ SUCCESS - ChunkedHybridHum2Melody imported!")
    print(f"[MODEL_SERVER.PY]   ChunkedHybridHum2Melody type: {type(ChunkedHybridHum2Melody)}")
except ImportError as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - ImportError: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   ChunkedHybridHum2Melody will be None")
    ChunkedHybridHum2Melody = None
except Exception as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - Unexpected error: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   ChunkedHybridHum2Melody will be None")
    ChunkedHybridHum2Melody = None

print(f"[MODEL_SERVER.PY] Final ChunkedHybridHum2Melody status: {ChunkedHybridHum2Melody is not None}")
print("[MODEL_SERVER.PY] ========================================")


class ModelServer:
    """
    Serves AI model predictions for music generation.
    Now integrates the trained model if available, but falls back to mock logic.
    """

    def __init__(self, preload: bool = True):
        """Initialize model server with optional preloading

        Args:
            preload: If True, load model immediately. If False, lazy-load on first use.
        """
        print("[ModelServer.__init__] ========================================")
        print(f"[ModelServer.__init__] Initializing ModelServer (preload={preload})")

        # Store checkpoint path but don't load the model yet
        self.checkpoint_path = Path("hum2melody/checkpoints/combined_hum2melody_full.pth")
        print(f"[ModelServer.__init__]   Checkpoint path: {self.checkpoint_path}")
        print(f"[ModelServer.__init__]   Checkpoint path (absolute): {self.checkpoint_path.absolute()}")
        print(f"[ModelServer.__init__]   Checkpoint exists: {self.checkpoint_path.exists()}")

        if self.checkpoint_path.exists():
            size_mb = self.checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"[ModelServer.__init__]   Checkpoint size: {size_mb:.1f} MB")

        # Model will be loaded on first use (lazy loading) or immediately if preload=True
        self.predictor = None
        self._model_loading_attempted = False

        # Define scales for fallback
        self.c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]
        self.a_minor_scale = [57, 59, 60, 62, 64, 65, 67, 69]

        if preload:
            print("[ModelServer.__init__] Preloading model on startup...")
            self._ensure_model_loaded()

        print("[ModelServer.__init__] ✅ Initialized")
        print("[ModelServer.__init__] ========================================")

    def _ensure_model_loaded(self):
        """Lazy-load the model on first use to save memory on startup"""
        if self._model_loading_attempted:
            return  # Already tried to load (success or failure)

        self._model_loading_attempted = True
        print("[ModelServer._ensure_model_loaded] ========================================")
        print("[ModelServer._ensure_model_loaded] Attempting lazy model load...")

        if ChunkedHybridHum2Melody is not None and self.checkpoint_path.exists():
            print("[ModelServer._ensure_model_loaded] ✅ Both conditions met - loading model")
            try:
                print(f"[ModelServer._ensure_model_loaded]   Creating ChunkedHybridHum2Melody instance...")
                print(f"[ModelServer._ensure_model_loaded]   Args: checkpoint={self.checkpoint_path}, device=cpu")

                self.predictor = ChunkedHybridHum2Melody(
                    checkpoint_path=str(self.checkpoint_path),
                    device='cpu',
                    onset_high=0.30,
                    onset_low=0.10,
                    offset_high=0.30,
                    offset_low=0.10
                )
                print(f"[ModelServer._ensure_model_loaded]   ✅ Model loaded successfully!")
                print(f"[ModelServer._ensure_model_loaded]   Predictor device: {self.predictor.device}")
            except Exception as e:
                print(f"[ModelServer._ensure_model_loaded]   ❌ Failed to load model: {e}")
                print("[ModelServer._ensure_model_loaded]   Full traceback:")
                traceback.print_exc()
                self.predictor = None
        else:
            print("[ModelServer._ensure_model_loaded] ❌ Conditions NOT met for loading model:")
            if not self.checkpoint_path.exists():
                print(f"[ModelServer._ensure_model_loaded]   ❌ No trained model found at {self.checkpoint_path}")
            if ChunkedHybridHum2Melody is None:
                print("[ModelServer._ensure_model_loaded]   ❌ ChunkedHybridHum2Melody not available (import failed)")
            print("[ModelServer._ensure_model_loaded]   Using mock predictions")

        print("[ModelServer._ensure_model_loaded] ========================================")

    async def predict_melody(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict melody from humming audio features.
        Uses trained model if available; otherwise falls back to mock predictions.
        """
        print("[ModelServer.predict_melody] ========================================")
        print(f"[ModelServer.predict_melody] Called with audio_features keys: {list(audio_features.keys())}")

        # Lazy-load model on first use
        self._ensure_model_loaded()

        print(f"[ModelServer.predict_melody] Predictor is None: {self.predictor is None}")

        # Handle invalid or missing inputs
        if not audio_features:
            print("[ModelServer.predict_melody] ⚠️ Missing audio features, using mock.")
            return self._mock_melody(audio_features)

        # Try real model if available
        if self.predictor is not None:
            print("[ModelServer.predict_melody] ✅ Predictor available - attempting real inference")
            try:
                import tempfile
                import os

                # Get audio file path or create temporary file
                audio_path = audio_features.get("audio_path")
                temp_file = None

                # If audio_bytes provided, save to temp file
                audio_bytes = audio_features.get("audio_bytes")
                if audio_bytes and not audio_path:
                    print(f"[ModelServer.predict_melody]   Saving audio bytes ({len(audio_bytes)} bytes) to temp file")
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.write(audio_bytes)
                    temp_file.close()
                    audio_path = temp_file.name
                    print(f"[ModelServer.predict_melody]   Temp file created: {audio_path}")

                if audio_path and os.path.exists(audio_path):
                    print(f"[ModelServer.predict_melody]   Using audio file: {audio_path}")

                    # Call predict_chunked with min_confidence=0.25 (recommended for production)
                    notes = self.predictor.predict_chunked(audio_path, min_confidence=0.25)

                    print(f"[ModelServer.predict_melody]   ✅ Generated {len(notes)} notes from trained model")

                    # Convert notes to Track format
                    track_notes = []
                    for note_dict in notes:
                        note = Note(
                            pitch=note_dict['midi'],
                            start=note_dict['start'],
                            duration=note_dict['duration'],
                            velocity=int(note_dict['confidence'] * 127)  # Convert confidence to velocity
                        )
                        track_notes.append(note)

                    # Clean up temp file if we created one
                    if temp_file:
                        try:
                            os.unlink(temp_file.name)
                            print(f"[ModelServer.predict_melody]   Cleaned up temp file")
                        except:
                            pass

                    if track_notes:
                        # Get instrument from audio_features or use default
                        instrument = audio_features.get('instrument', 'piano/grand_piano_k')
                        return Track(id='melody', instrument=instrument, notes=track_notes)
                    else:
                        print("[ModelServer.predict_melody]   ⚠️ No notes generated, using mock")
                        return self._mock_melody(audio_features)
                else:
                    print("[ModelServer.predict_melody]   ⚠️ No valid audio file path available")
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