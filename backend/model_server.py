import numpy as np
from typing import Dict, List, Any
from .schemas import IR, Track, Note, SampleEvent
import random

class ModelServer:
    """
    Serves AI model predictions for music generation.
    Currently returns mock predictions - will integrate real PyTorch models later.
    EMPHASIS ON MOCK FOR RIHGT NOW
    """
    
    def __init__(self):
        """Initialize model server (load models here in future)"""
        # Placeholder for future model loading
        self.hum2melody_model = None
        self.beatbox2drums_model = None
        self.arranger_model = None
        
        # MIDI note mappings for mock data
        self.c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        self.a_minor_scale = [57, 59, 60, 62, 64, 65, 67, 69]  # A3 to A5
        
    async def predict_melody(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict melody from humming audio features.
        
        Args:
            audio_features: Dictionary of processed audio features from AudioProcessor
            
        Returns:
            Track object containing melody notes in IR format
        """
        # Extract relevant features
        onset_times = audio_features.get("onset_times", [])
        duration = audio_features.get("duration", 4.0)
        
        # Generate mock melody notes based on onset times
        notes = []
        
        if not onset_times:
            # If no onsets detected, create a simple pattern
            onset_times = [i * 0.5 for i in range(8)]
        
        for i, start_time in enumerate(onset_times):
            # Pick notes from scale
            pitch = self.a_minor_scale[i % len(self.a_minor_scale)]
            
            # Calculate duration until next onset (or use default)
            if i < len(onset_times) - 1:
                note_duration = onset_times[i + 1] - start_time
            else:
                note_duration = min(1.0, duration - start_time)
            
            # Add some velocity variation
            velocity = random.uniform(0.6, 0.9)
            
            notes.append(Note(
                pitch=pitch,
                duration=note_duration,
                velocity=velocity
            ))
        
        return Track(
            id="melody",
            instrument="lead_synth",
            notes=notes,
            samples=None
        )
    
    async def predict_drums(self, audio_features: Dict[str, Any]) -> Track:
        """
        Predict drum pattern from beatbox audio features.
        
        Args:
            audio_features: Dictionary of processed audio features from AudioProcessor
            
        Returns:
            Track object containing drum samples in IR format
        """
        onset_times = audio_features.get("onset_times", [])
        tempo = audio_features.get("tempo", 120)
        spectral_centroid = audio_features.get("spectral_centroid", [])
        
        samples = []
        
        if not onset_times:
            # Generate default 4-on-the-floor pattern
            beat_interval = 60.0 / tempo  # seconds per beat
            onset_times = [i * beat_interval for i in range(16)]
        
        # Classify drum sounds based on spectral features (mock classification)
        for i, start_time in enumerate(onset_times):
            # Simple heuristic: low spectral centroid = kick, high = snare/hihat
            # In reality, you'd use ML model here
            
            if i % 4 == 0:
                # Kick on downbeats
                sample_type = "kick"
            elif i % 2 == 1:
                # Snare on backbeats
                sample_type = "snare"
            else:
                # Hi-hat on other beats
                sample_type = "hihat"
            
            samples.append(SampleEvent(
                sample=sample_type,
                start=float(start_time)
            ))
        
        return Track(
            id="drums",
            instrument=None,
            notes=None,
            samples=samples
        )
    
    async def arrange_track(self, existing_ir: IR, style: str = "pop") -> IR:
        """
        Take existing melody and add accompanying tracks (bass, chords, etc).
        
        Args:
            existing_ir: Existing IR with at least one track
            style: Musical style for arrangement ("pop", "jazz", "electronic")
            
        Returns:
            Enhanced IR with additional tracks
        """
        # Extract melody track
        melody_track = None
        for track in existing_ir.tracks:
            if track.notes:
                melody_track = track
                break
        
        if not melody_track or not melody_track.notes:
            # No melody to arrange around, return original
            return existing_ir
        
        # Generate bass line (root notes, octave below melody)
        bass_notes = []
        for i, note in enumerate(melody_track.notes[::2]):  # Every other note
            bass_pitch = note.pitch - 12  # One octave down
            bass_duration = note.duration * 2
            bass_notes.append(Note(
                pitch=bass_pitch,
                duration=bass_duration,
                velocity=0.8
            ))
        
        bass_track = Track(
            id="bass",
            instrument="bass_synth",
            notes=bass_notes,
            samples=None
        )
        
        # Generate chord progression (simplified)
        chord_notes = []
        chord_progression = [60, 65, 67, 62]  # I-IV-V-ii in C
        
        for i, chord_root in enumerate(chord_progression):
            start_time = i * 2.0  # 2 seconds per chord
            duration = 2.0
            
            # Add chord tones (root, third, fifth)
            for offset in [0, 4, 7]:
                chord_notes.append(Note(
                    pitch=chord_root + offset,
                    duration=duration,
                    velocity=0.5
                ))
        
        chord_track = Track(
            id="chords",
            instrument="pad_synth",
            notes=chord_notes,
            samples=None
        )
        
        # Add drums if not present
        has_drums = any(track.samples for track in existing_ir.tracks)
        
        new_tracks = existing_ir.tracks.copy()
        new_tracks.append(bass_track)
        new_tracks.append(chord_track)
        
        if not has_drums:
            # Add basic drum pattern
            drum_samples = []
            for i in range(16):
                if i % 4 == 0:
                    drum_samples.append(SampleEvent(sample="kick", start=float(i * 0.5)))
                if i % 4 == 2:
                    drum_samples.append(SampleEvent(sample="snare", start=float(i * 0.5)))
            
            drum_track = Track(
                id="drums",
                instrument=None,
                notes=None,
                samples=drum_samples
            )
            new_tracks.append(drum_track)
        
        return IR(
            metadata=existing_ir.metadata,
            tracks=new_tracks
        )
    
    def _predict_with_model(self, model, features: np.ndarray): # -> np.ndarray:
        """
        Helper method for future PyTorch model inference.
        
        Args:
            model: PyTorch model
            features: Input features as numpy array
            
        Returns:
            Model predictions as numpy array
        """
        # Placeholder for actual model inference
        # import torch
        # with torch.no_grad():
        #     tensor_features = torch.from_numpy(features).float()
        #     predictions = model(tensor_features)
        #     return predictions.numpy()
        pass