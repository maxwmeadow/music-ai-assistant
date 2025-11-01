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

class Track(BaseModel):
    id: str
    instrument: Optional[str] = None
    notes: Optional[List[Note]] = None
    samples: Optional[List[SampleEvent]] = None
    # Music theory metadata (from post-processing)
    metadata: Optional[Dict[str, Any]] = None      # key, tempo, grid_resolution, etc.
    harmony: Optional[List[ChordEvent]] = None     # Chord progression

class IR(BaseModel):
    metadata: Dict[str, Any]  # at least { "tempo": int }
    tracks: List[Track]

# ----- Request / Response ----- #

class RunBody(BaseModel):
    code: Optional[str] = None   # DSL string
    ir: Optional[IR] = None      # JSON IR

class RunnerEvalResponse(BaseModel):
    dsl: str
    meta: Optional[Dict[str, Any]] = None

