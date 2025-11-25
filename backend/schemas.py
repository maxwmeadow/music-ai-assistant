from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel


# ----- IR schema ----- #

class Note(BaseModel):
    pitch: int              # MIDI number
    start: float            # absolute start time in seconds
    duration: float         # seconds or beats
    velocity: float         # 0.0–1.0

class SampleEvent(BaseModel):
    sample: Literal["kick", "snare", "hihat", "clap", "perc"]  # extendable
    start: float            # when to trigger it

class ChordEvent(BaseModel):
    """Chord event for harmony track."""
    root: str              # Root note ("C", "D#", etc.)
    quality: str           # Chord quality ("major", "minor", "dom7", etc.)
    roman: str             # Roman numeral analysis ("I", "V7", "ii", etc.)
    start: float           # Start time in seconds
    duration: float        # Duration in seconds

class AudioClip(BaseModel):
    """Audio clip for vocal/audio tracks."""
    audio_data: str        # Base64 encoded audio data (WAV)
    start: float           # Start time in seconds
    duration: float        # Duration in seconds
    volume: float = 1.0    # Volume multiplier (0.0-1.0)

class Track(BaseModel):
    id: str
    instrument: Optional[str] = None
    notes: Optional[List[Note]] = None
    samples: Optional[List[SampleEvent]] = None
    audio: Optional[List[AudioClip]] = None    # Audio clips for vocal/audio tracks
    # Music theory metadata (from post-processing)
    metadata: Optional[Dict[str, Any]] = None      # key, tempo, grid_resolution, etc.
    harmony: Optional[List[ChordEvent]] = None     # Chord progression

class IR(BaseModel):
    metadata: Dict[str, Any]  # at least { "tempo": int }
    tracks: List[Track]

# ----- Request / Response ----- #

class GenerateTrackRequest(BaseModel):
    """Request for LLM-based track generation"""
    dsl_code: str                       # Current DSL code from editor
    track_type: str                     # bass, chords, pad, melody, counterMelody, arpeggio, drums
    genre: str                          # pop, jazz, electronic, etc.
    custom_request: Optional[str] = None  # Additional user instructions
    creativity: Optional[float] = 0.7    # Creativity level 0.0-1.0
    complexity: Optional[str] = "medium"  # Complexity level: simple, medium, complex

class RunBody(BaseModel):
    code: Optional[str] = None   # DSL string
    ir: Optional[IR] = None      # JSON IR

class RunnerEvalResponse(BaseModel):
    dsl: str
    meta: Optional[Dict[str, Any]] = None

