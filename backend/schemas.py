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

class Track(BaseModel):
    id: str
    instrument: Optional[str] = None
    notes: Optional[List[Note]] = None
    samples: Optional[List[SampleEvent]] = None

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

