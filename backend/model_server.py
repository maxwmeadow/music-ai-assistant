"""
Model Server with proper integration of trained Hum2Melody model
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import random

from .schemas import IR, Track, Note, SampleEvent

# Import your trained model wrapper
try:
    from inference.predictor import ImprovedMelodyPredictor as MelodyPredictor
except ImportError:
    MelodyPredictor = None


class ModelServer:
    """
    Serves AI model predictions for music generation.
    Now integrates the trained model if available, but falls back to mock logic.
    """

    def __init__(self):
        """Initialize model server and optionally load trained model"""
        checkpoint_path = Path("backend/checkpoints/best_model.pth")

        if MelodyPredictor is not None and checkpoint_path.exists():
            try:
                self.predictor = MelodyPredictor(
                    str(checkpoint_path),
                    threshold=0.5,  # Adjust this based on your validation results
                    min_note_duration=0.1  # Filter very short notes
                )
                print(f"✅ Loaded trained model from {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                self.predictor = None
        else:
            self.predictor = None
            if not checkpoint_path.exists():
                print(f"⚠️ No trained model found at {checkpoint_path}")
            if MelodyPredictor is None:
                print("⚠️ MelodyPredictor not available")
            print("Using mock predictions")

        # Define scales for fallback
        self.c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4–C5
        self.a_minor_scale = [57, 59, 60, 62, 64, 65, 67, 69]  # A3–A5

    async def predict_melody(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict melody from humming audio features.
        Uses trained model if available; otherwise falls back to mock predictions.
        """
        # Handle invalid or missing inputs
        if not audio_features:
            print("⚠️ Missing audio features, using mock.")
            return self._mock_melody(audio_features)

        # Try real model if available
        if self.predictor is not None:
            try:
                # Priority 1: Raw audio bytes (best for API uploads)
                audio_bytes = audio_features.get("audio_bytes")
                if audio_bytes:
                    print("[Model] Using raw audio bytes")
                    track = self.predictor.predict_from_bytes(audio_bytes)
                    if track and getattr(track, "notes", None):
                        print(f"✅ Generated {len(track.notes)} notes from trained model")
                        return track

                # Priority 2: Audio array (if already loaded)
                audio = audio_features.get("audio")
                if audio is not None and isinstance(audio, np.ndarray):
                    print("[Model] Using audio array")
                    track = self.predictor.predict_from_audio(audio)
                    if track and getattr(track, "notes", None):
                        print(f"✅ Generated {len(track.notes)} notes from trained model")
                        return track

                # Priority 3: File path
                audio_path = audio_features.get("audio_path")
                if audio_path:
                    print("[Model] Using audio file path")
                    track = self.predictor.predict_from_file(audio_path)
                    if track and getattr(track, "notes", None):
                        print(f"✅ Generated {len(track.notes)} notes from trained model")
                        return track

                # If we get here, no valid audio format was provided
                print("⚠️ No valid audio format (need audio_bytes, audio array, or audio_path)")
                return self._mock_melody(audio_features)

            except Exception as e:
                print(f"⚠️ Model inference failed: {e}")
                import traceback
                traceback.print_exc()
                return self._mock_melody(audio_features)
        else:
            # Fallback to mock predictions
            print("⚠️ No trained model loaded, using mock.")
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
        for i, note in enumerate(melody_track.notes[::2]):  # Every other note
            bass_notes.append(
                Note(
                    pitch=note.pitch - 12,
                    start=note.start,  # Use same start time
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
            
            # IMPORTANT: Include start time!
            notes.append(Note(
                pitch=pitch,
                start=start_time,  # Now included
                duration=note_duration,
                velocity=velocity
            ))

        return Track(
            id="melody", instrument="guitar/rjs_guitar_new_strings", notes=notes, samples=None
        )