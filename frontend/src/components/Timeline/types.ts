export interface TimelineNote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  isChord?: boolean;
  chordPitches?: number[]; // All pitches in the chord (includes root)
  isFromLoop?: boolean; // True if this note was generated from a loop block
}

export interface SelectedNote {
  trackId: string;
  noteIndex: number;
}

export interface DraggingNote {
  trackId: string;
  noteIndex: number;
  startX: number;
  initialStart: number;
  currentStart?: number; // Temporary visual state during drag
  selectedNotes?: SelectedNote[]; // For multi-selection dragging
  committed?: boolean; // True when mouse is released but visual state is still active
}

export interface ResizingNote {
  trackId: string;
  noteIndex: number;
  startX: number;
  initialDuration: number;
  currentDuration?: number; // Temporary visual state during resize
  committed?: boolean; // True when mouse is released but visual state is still active
}

export interface CopiedNote {
  note: TimelineNote;
  trackId: string;
}

export interface CopiedNotes {
  notes: Array<{ note: TimelineNote; trackId: string }>;
  minStart: number; // Store the earliest start time for relative positioning
}

export const SNAP_OPTIONS = [
  { label: '1/4 (Whole)', value: 4 },
  { label: '1/2 (Half)', value: 2 },
  { label: '1 (Quarter)', value: 1 },
  { label: '1/2 (8th)', value: 0.5 },
  { label: '1/4 (16th)', value: 0.25 },
  { label: '1/8 (32nd)', value: 0.125 },
];
