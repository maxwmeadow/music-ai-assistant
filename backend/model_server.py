"""Model Server - AI model predictions for music generation"""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import random
import sys
import traceback

# Import schemas
try:
    from .schemas import IR, Track, Note, SampleEvent
except ImportError as e:
    print(f"❌ Failed to import schemas: {e}")
    raise

# Import hum2melody package
ChunkedHybridHum2Melody = None
try:
    from .hum2melody.inference.hybrid_inference_chunked import ChunkedHybridHum2Melody
    print("✅ ChunkedHybridHum2Melody loaded")
except ImportError as e:
    print(f"⚠️ ChunkedHybridHum2Melody unavailable: {e}")

# Import beatbox2drums package
Beatbox2DrumsPipeline = None
try:
    from .beatbox2drums.inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline
    print("✅ Beatbox2DrumsPipeline loaded")
except ImportError as e:
    print(f"⚠️ Beatbox2DrumsPipeline unavailable: {e}")


class ModelServer:
    """Serves AI model predictions for music generation."""

    def __init__(self):
        """Initialize model server with optional lazy loading"""
        import os

        self.checkpoint_path = Path("hum2melody/checkpoints/combined_hum2melody_full.pth")
        self._model_loading_attempted = False
        self._beatbox_loading_attempted = False
        self.lazy_load = os.getenv("LAZY_LOAD_MODELS", "false").lower() == "true"

        # Define scales for fallback
        self.c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]
        self.a_minor_scale = [57, 59, 60, 62, 64, 65, 67, 69]

        # Initialize model attributes
        self.predictor = None
        self.beatbox_predictor = None

        if self.lazy_load:
            print("[ModelServer.__init__] ⚡ Lazy loading enabled - models will load on first use")
        else:
            self._load_hum2melody_model()
            self._load_beatbox2drums_model()

    def _load_hum2melody_model(self):
        """Load the hum2melody model if not already loaded"""
        if self.predictor is not None or self._model_loading_attempted:
            return

        self._model_loading_attempted = True
        checkpoint_path = self.checkpoint_path

        if ChunkedHybridHum2Melody is not None and checkpoint_path.exists():
            try:
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                print(f"Loading Hum2Melody model ({size_mb:.1f} MB)...")
                self.predictor = ChunkedHybridHum2Melody(
                    checkpoint_path=str(checkpoint_path),
                    device='cpu',
                    onset_high=0.30,
                    onset_low=0.10,
                    offset_high=0.30,
                    offset_low=0.10
                )
                print("✅ Hum2Melody model ready")
            except Exception as e:
                print(f"❌ Failed to load Hum2Melody: {e}")
                self.predictor = None
        else:
            self.predictor = None
            if not checkpoint_path.exists():
                print(f"❌ Checkpoint not found: {checkpoint_path}")

    def _load_beatbox2drums_model(self):
        """Load the beatbox2drums model if not already loaded"""
        if self.beatbox_predictor is not None or self._beatbox_loading_attempted:
            return

        self._beatbox_loading_attempted = True

        onset_checkpoint_path = Path("beatbox2drums/checkpoints/onset_detector/best_onset_model.h5")
        classifier_checkpoint_path = Path("beatbox2drums/checkpoints/drum_classifier/best_model_multi_input.pth")
        feature_norm_path = Path("beatbox2drums/checkpoints/drum_classifier/feature_normalization.npz")

        if Beatbox2DrumsPipeline is not None and onset_checkpoint_path.exists() and classifier_checkpoint_path.exists():
            try:
                print("Loading Beatbox2Drums pipeline...")
                self.beatbox_predictor = Beatbox2DrumsPipeline(
                    onset_checkpoint_path=str(onset_checkpoint_path),
                    classifier_checkpoint_path=str(classifier_checkpoint_path),
                    onset_threshold=0.5,
                    onset_peak_delta=0.05,
                    classifier_confidence_threshold=0.3,
                    device='cpu',
                    use_multi_input=True,
                    feature_norm_path=str(feature_norm_path) if feature_norm_path.exists() else None
                )
                print("✅ Beatbox2Drums pipeline ready")
            except Exception as e:
                print(f"❌ Failed to load Beatbox2Drums: {e}")
                self.beatbox_predictor = None
        else:
            self.beatbox_predictor = None
            print("❌ Beatbox2Drums checkpoints not found")

    async def predict_melody(self, audio_features: Dict[str, Any]) -> Track:
        """Predict melody from humming audio features."""
        # Lazy load model if needed
        if self.lazy_load and self.predictor is None and not self._model_loading_attempted:
            self._load_hum2melody_model()

        # Handle invalid or missing inputs
        if not audio_features:
            return self._mock_melody(audio_features)

        # Try real model if available
        if self.predictor is not None:
            try:
                import tempfile
                import os

                # Get audio file path or create temporary file
                audio_path = audio_features.get("audio_path")
                temp_file = None

                # If audio_bytes provided, save to temp file
                audio_bytes = audio_features.get("audio_bytes")
                if audio_bytes and not audio_path:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.write(audio_bytes)
                    temp_file.close()
                    audio_path = temp_file.name

                if audio_path and os.path.exists(audio_path):
                    # Call predict_chunked with min_confidence=0.25
                    notes = self.predictor.predict_chunked(audio_path, min_confidence=0.25)

                    # Convert notes to Track format
                    track_notes = []
                    for note_dict in notes:
                        note = Note(
                            pitch=note_dict['midi'],
                            start=note_dict['start'],
                            duration=note_dict['duration'],
                            velocity=int(note_dict['confidence'] * 127)
                        )
                        track_notes.append(note)

                    # Clean up temp file if we created one
                    if temp_file:
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass

                    if track_notes:
                        instrument = audio_features.get('instrument', 'piano/grand_piano_k')
                        print(f"✅ Generated {len(track_notes)} notes from model")
                        return Track(id='melody', instrument=instrument, notes=track_notes)
                    else:
                        print("⚠️ No notes detected, using mock")
                        return self._mock_melody(audio_features)
                else:
                    return self._mock_melody(audio_features)

            except Exception as e:
                print(f"❌ Model inference failed: {e}")
                traceback.print_exc()
                return self._mock_melody(audio_features)
        else:
            print("⚠️ No trained model, using mock")
            return self._mock_melody(audio_features)


    async def predict_drums(self, audio_features: Dict[str, Any], return_visualization: bool = False):
        """Predict drum pattern from beatbox audio features."""
        # Lazy load model if needed
        if self.lazy_load and self.beatbox_predictor is None and not self._beatbox_loading_attempted:
            self._load_beatbox2drums_model()

        # Handle invalid or missing inputs
        if not audio_features:
            return self._mock_drums(audio_features), None

        # Try real model if available
        if self.beatbox_predictor is not None:
            try:
                import tempfile
                import os

                # Get audio file path or create temporary file
                temp_file = None
                audio_bytes = audio_features.get("audio_bytes")

                if audio_bytes:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.write(audio_bytes)
                    temp_file.close()
                    audio_path = temp_file.name
                else:
                    audio_path = audio_features.get("audio_path")

                if audio_path and os.path.exists(audio_path):
                    # Call pipeline to get drum hits
                    drum_hits = self.beatbox_predictor.predict(audio_path)

                    # Convert drum hits to Note format with MIDI pitches
                    # Map drum types to MIDI notes (matches runner/server.js)
                    DRUM_TO_MIDI = {
                        'kick': 36,           # C2
                        'snare': 38,          # D2
                        'snare_rimshot': 40,  # E2
                        'snare_buzz': 39,     # D#2
                        'hihat': 42,          # F#2
                        'hihat_closed': 42,   # F#2
                        'hihat_open': 46,     # A#2
                        'hihat_pedal': 44,    # G#2
                        'tom': 43,            # G2
                        'crash': 49,          # C#3
                        'ride': 51,           # D#3
                    }

                    # Duration by drum type (in seconds)
                    DRUM_DURATIONS = {
                        'kick': 0.4,
                        'snare': 0.3,
                        'snare_rimshot': 0.2,
                        'snare_buzz': 0.5,
                        'hihat': 0.15,
                        'hihat_closed': 0.15,
                        'hihat_open': 0.8,
                        'hihat_pedal': 0.2,
                        'tom': 0.5,
                        'crash': 1.5,
                        'ride': 0.4,
                    }

                    # Standard velocity by drum type
                    DRUM_VELOCITIES = {
                        'kick': 0.85,
                        'snare': 0.8,
                        'snare_rimshot': 0.75,
                        'snare_buzz': 0.7,
                        'hihat': 0.6,
                        'hihat_closed': 0.6,
                        'hihat_open': 0.65,
                        'hihat_pedal': 0.5,
                        'tom': 0.75,
                        'crash': 0.8,
                        'ride': 0.7,
                    }

                    notes = []
                    for i, hit in enumerate(drum_hits):
                        midi_pitch = DRUM_TO_MIDI.get(hit.drum_type, 36)  # Default to kick

                        # Get default duration for this drum type
                        default_duration = DRUM_DURATIONS.get(hit.drum_type, 0.5)

                        # Calculate duration based on time to next onset
                        if i < len(drum_hits) - 1:
                            time_to_next = float(drum_hits[i + 1].time) - float(hit.time)
                            # Use minimum of default duration and time to next hit
                            duration = min(default_duration, time_to_next * 0.9)  # 90% to avoid overlap
                        else:
                            duration = default_duration

                        # Get standard velocity for this drum type
                        velocity = DRUM_VELOCITIES.get(hit.drum_type, 0.8)

                        note = Note(
                            pitch=midi_pitch,
                            start=float(hit.time),
                            duration=duration,
                            velocity=velocity
                        )
                        notes.append(note)

                    # Build visualization data if requested
                    visualization_data = None
                    if return_visualization and drum_hits:
                        import librosa

                        # Load audio for waveform
                        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                        duration = len(audio) / sr

                        # Downsample waveform for visualization
                        downsample_factor = 100
                        downsampled_waveform = audio[::downsample_factor].tolist()

                        # Create drum hits data
                        hits_data = []
                        for hit in drum_hits:
                            hit_data = {
                                "time": float(hit.time),
                                "drum_type": hit.drum_type,
                                "confidence": float(hit.confidence)
                            }
                            if hit.probabilities:
                                hit_data["probabilities"] = {
                                    "kick": float(hit.probabilities.get('kick', 0)),
                                    "snare": float(hit.probabilities.get('snare', 0)),
                                    "hihat": float(hit.probabilities.get('hihat', 0))
                                }
                            hits_data.append(hit_data)

                        visualization_data = {
                            "waveform": {
                                "data": downsampled_waveform,
                                "sample_rate": sr // downsample_factor,
                                "duration": float(duration)
                            },
                            "drum_hits": hits_data,
                            "num_hits": len(drum_hits)
                        }

                    # Clean up temp file
                    if temp_file:
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass

                    if notes:
                        print(f"✅ Generated {len(notes)} drum hits from model")
                        # Use notes instead of samples for consistency, with drum instrument
                        return Track(id='drums', instrument='drums/bedroom_drums', notes=notes, samples=None), visualization_data
                    else:
                        print("⚠️ No drum hits detected, using mock")
                        return self._mock_drums(audio_features), None
                else:
                    return self._mock_drums(audio_features), None

            except Exception as e:
                print(f"❌ Drum inference failed: {e}")
                traceback.print_exc()
                return self._mock_drums(audio_features), None
        else:
            print("⚠️ No trained model, using mock")
            return self._mock_drums(audio_features), None

    def _mock_drums(self, audio_features: Dict[str, Any]) -> Track:
        """Fallback drum pattern generator for testing."""
        onset_times = audio_features.get("onset_times", [])
        tempo = audio_features.get("tempo", 120)

        # Drum MIDI mapping (matches runner/server.js)
        DRUM_TO_MIDI = {
            'kick': 36,
            'snare': 38,
            'hihat_closed': 42
        }

        # Duration and velocity by type
        DRUM_SETTINGS = {
            'kick': {'duration': 0.4, 'velocity': 0.85},
            'snare': {'duration': 0.3, 'velocity': 0.8},
            'hihat_closed': {'duration': 0.15, 'velocity': 0.6},
        }

        notes = []
        if not onset_times:
            beat_interval = 60.0 / tempo
            onset_times = [i * beat_interval for i in range(16)]

        for i, start_time in enumerate(onset_times):
            if i % 4 == 0:
                drum_type = "kick"
            elif i % 2 == 1:
                drum_type = "snare"
            else:
                drum_type = "hihat_closed"

            settings = DRUM_SETTINGS[drum_type]

            # Calculate duration based on time to next onset
            if i < len(onset_times) - 1:
                time_to_next = onset_times[i + 1] - start_time
                duration = min(settings['duration'], time_to_next * 0.9)
            else:
                duration = settings['duration']

            note = Note(
                pitch=DRUM_TO_MIDI[drum_type],
                start=float(start_time),
                duration=duration,
                velocity=settings['velocity']
            )
            notes.append(note)

        print(f"Mock: Generated {len(notes)} drum notes")
        return Track(id="drums", instrument='drums/bedroom_drums', notes=notes, samples=None)


    async def arrange_track(self, existing_ir: IR, style: str = "pop") -> IR:
        """Take existing melody and add accompanying tracks (bass, chords, etc)."""
        melody_track = next((t for t in existing_ir.tracks if t.notes), None)
        if not melody_track or not melody_track.notes:
            return existing_ir

        # Generate bass
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

        # Chords
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

            notes.append(Note(
                pitch=pitch,
                start=start_time,
                duration=note_duration,
                velocity=velocity
            ))

        print(f"Mock: Generated {len(notes)} notes")
        return Track(
            id="melody", instrument="guitar/rjs_guitar_new_strings", notes=notes, samples=None
        )
