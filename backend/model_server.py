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

# Import the new beatbox2drums package
print("[MODEL_SERVER.PY] Attempting to import Beatbox2DrumsPipeline...")
print("[MODEL_SERVER.PY]   Trying: from .beatbox2drums.inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline")

Beatbox2DrumsPipeline = None
try:
    from .beatbox2drums.inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline
    print("[MODEL_SERVER.PY]   ✅ SUCCESS - Beatbox2DrumsPipeline imported!")
    print(f"[MODEL_SERVER.PY]   Beatbox2DrumsPipeline type: {type(Beatbox2DrumsPipeline)}")
except ImportError as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - ImportError: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   Beatbox2DrumsPipeline will be None")
    Beatbox2DrumsPipeline = None
except Exception as e:
    print(f"[MODEL_SERVER.PY]   ❌ FAILED - Unexpected error: {e}")
    print("[MODEL_SERVER.PY]   Full traceback:")
    traceback.print_exc()
    print("[MODEL_SERVER.PY]   Beatbox2DrumsPipeline will be None")
    Beatbox2DrumsPipeline = None

print(f"[MODEL_SERVER.PY] Final Beatbox2DrumsPipeline status: {Beatbox2DrumsPipeline is not None}")
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

        # Initialize beatbox2drums pipeline
        print("[ModelServer.__init__] ========================================")
        print("[ModelServer.__init__] Initializing Beatbox2Drums Pipeline")

        onset_checkpoint_path = Path("beatbox2drums/checkpoints/onset_detector/best_onset_model.h5")
        classifier_checkpoint_path = Path("beatbox2drums/checkpoints/drum_classifier/best_model_multi_input.pth")
        feature_norm_path = Path("beatbox2drums/checkpoints/drum_classifier/feature_normalization.npz")

        print(f"[ModelServer.__init__]   Onset checkpoint: {onset_checkpoint_path}")
        print(f"[ModelServer.__init__]   Onset checkpoint exists: {onset_checkpoint_path.exists()}")
        print(f"[ModelServer.__init__]   Classifier checkpoint: {classifier_checkpoint_path}")
        print(f"[ModelServer.__init__]   Classifier checkpoint exists: {classifier_checkpoint_path.exists()}")
        print(f"[ModelServer.__init__]   Feature norm path: {feature_norm_path}")
        print(f"[ModelServer.__init__]   Feature norm exists: {feature_norm_path.exists()}")
        print(f"[ModelServer.__init__]   Beatbox2DrumsPipeline is None: {Beatbox2DrumsPipeline is None}")

        if Beatbox2DrumsPipeline is not None and onset_checkpoint_path.exists() and classifier_checkpoint_path.exists():
            print("[ModelServer.__init__] ✅ All conditions met - attempting to load beatbox2drums pipeline")
            try:
                print(f"[ModelServer.__init__]   Creating Beatbox2DrumsPipeline instance with multi-input model...")
                self.beatbox_predictor = Beatbox2DrumsPipeline(
                    onset_checkpoint_path=str(onset_checkpoint_path),
                    classifier_checkpoint_path=str(classifier_checkpoint_path),
                    onset_threshold=0.5,
                    onset_peak_delta=0.05,  # 50ms NMS window
                    classifier_confidence_threshold=0.3,
                    device='cpu',
                    use_multi_input=True,  # Use new multi-input model with spectral features
                    feature_norm_path=str(feature_norm_path) if feature_norm_path.exists() else None
                )
                print(f"[ModelServer.__init__]   ✅ Beatbox2Drums pipeline loaded successfully!")
                print(f"[ModelServer.__init__]   Pipeline device: {self.beatbox_predictor.device}")
                print(f"[ModelServer.__init__]   Multi-input mode: {self.beatbox_predictor.use_multi_input}")
            except Exception as e:
                print(f"[ModelServer.__init__]   ❌ Failed to load beatbox2drums pipeline: {e}")
                print("[ModelServer.__init__]   Full traceback:")
                traceback.print_exc()
                self.beatbox_predictor = None
        else:
            print("[ModelServer.__init__] ❌ Conditions NOT met for loading beatbox2drums pipeline:")
            self.beatbox_predictor = None
            if not onset_checkpoint_path.exists():
                print(f"[ModelServer.__init__]   ❌ Onset checkpoint not found at {onset_checkpoint_path}")
            if not classifier_checkpoint_path.exists():
                print(f"[ModelServer.__init__]   ❌ Classifier checkpoint not found at {classifier_checkpoint_path}")
            if Beatbox2DrumsPipeline is None:
                print("[ModelServer.__init__]   ❌ Beatbox2DrumsPipeline not available (import failed)")
            print("[ModelServer.__init__]   Using mock predictions for beatbox2drums")

        print(f"[ModelServer.__init__] Final beatbox_predictor status: {self.beatbox_predictor is not None}")
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

        print(f"[ModelServer._ensure_model_loaded] Final predictor status: {self.predictor is not None}")
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


    async def predict_drums(self, audio_features: Dict[str, Any], return_visualization: bool = False):
        """
        Predict drum pattern from beatbox audio features.
        Uses trained CNN pipeline if available; otherwise falls back to mock predictions.

        Returns:
            Tuple of (Track, Optional[visualization_data dict])
        """
        print("[ModelServer.predict_drums] ========================================")
        print(f"[ModelServer.predict_drums] Called with audio_features keys: {list(audio_features.keys())}")
        print(f"[ModelServer.predict_drums] Beatbox predictor is None: {self.beatbox_predictor is None}")

        # Handle invalid or missing inputs
        if not audio_features:
            print("[ModelServer.predict_drums] ⚠️ Missing audio features, using mock.")
            return self._mock_drums(audio_features), None

        # Try real model if available
        if self.beatbox_predictor is not None:
            print("[ModelServer.predict_drums] ✅ Beatbox predictor available - attempting real inference")
            try:
                import tempfile
                import os

                # Get audio file path or create temporary file
                temp_file = None
                audio_bytes = audio_features.get("audio_bytes")

                # Prefer audio_bytes if available (creates reliable temp file)
                if audio_bytes:
                    print(f"[ModelServer.predict_drums]   Saving audio bytes ({len(audio_bytes)} bytes) to temp file")
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.write(audio_bytes)
                    temp_file.close()
                    audio_path = temp_file.name
                    print(f"[ModelServer.predict_drums]   Temp file created: {audio_path}")
                else:
                    # Fall back to saved audio_path if audio_bytes not available
                    audio_path = audio_features.get("audio_path")
                    print(f"[ModelServer.predict_drums]   Using saved audio path: {audio_path}")

                if audio_path and os.path.exists(audio_path):
                    print(f"[ModelServer.predict_drums]   Using audio file: {audio_path}")

                    # Call pipeline to get drum hits
                    drum_hits = self.beatbox_predictor.predict(audio_path)

                    print(f"[ModelServer.predict_drums]   ✅ Generated {len(drum_hits)} drum hits from trained model")

                    # Convert drum hits to SampleEvent format
                    samples = []
                    for hit in drum_hits:
                        sample_event = SampleEvent(
                            sample=hit.drum_type,  # 'kick', 'snare', or 'hihat'
                            start=float(hit.time)
                        )
                        samples.append(sample_event)

                    # Build visualization data if requested
                    visualization_data = None
                    if return_visualization and drum_hits:
                        import librosa
                        import numpy as np

                        # Load audio for waveform
                        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                        duration = len(audio) / sr

                        # Downsample waveform for visualization (every 100 samples for 16kHz)
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
                            # Add probabilities if available
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
                                "sample_rate": sr // downsample_factor,  # Effective sample rate after downsampling
                                "duration": float(duration)
                            },
                            "drum_hits": hits_data,
                            "num_hits": len(drum_hits)
                        }

                        print(f"[ModelServer.predict_drums]   ✅ Generated visualization data with {len(drum_hits)} hits")

                    # Clean up temp file if we created one
                    if temp_file:
                        try:
                            os.unlink(temp_file.name)
                            print(f"[ModelServer.predict_drums]   Cleaned up temp file")
                        except:
                            pass

                    if samples:
                        print(f"[ModelServer.predict_drums]   Returning {len(samples)} drum samples")
                        return Track(id='drums', instrument=None, notes=None, samples=samples), visualization_data
                    else:
                        print("[ModelServer.predict_drums]   ⚠️ No drum hits generated, using mock")
                        return self._mock_drums(audio_features), None
                else:
                    print("[ModelServer.predict_drums]   ⚠️ No valid audio file path available")
                    return self._mock_drums(audio_features), None

            except Exception as e:
                print(f"[ModelServer.predict_drums]   ❌ Model inference failed: {e}")
                print("[ModelServer.predict_drums]   Full traceback:")
                traceback.print_exc()
                return self._mock_drums(audio_features), None
        else:
            # Fallback to mock predictions
            print("[ModelServer.predict_drums] ❌ No trained model loaded, using mock.")
            return self._mock_drums(audio_features), None

    def _mock_drums(self, audio_features: Dict[str, Any]) -> Track:
        """Fallback drum pattern generator for testing."""
        print("[ModelServer._mock_drums] Using mock drum pattern generator")

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

        print(f"[ModelServer._mock_drums]   Generated {len(samples)} mock drum samples")
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