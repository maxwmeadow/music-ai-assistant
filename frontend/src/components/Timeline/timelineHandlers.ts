import { TimelineNote, SelectedNote, CopiedNote, CopiedNotes } from "./types";
import { parseNotesFromDSL, updateDSLWithNewNotes, secondsToBeats, snapToGrid } from "./timelineHelpers";

/**
 * Handle note deletion (supports multiple selections)
 */
export function handleDeleteNotes(
  selectedNotes: SelectedNote[],
  dslCode: string,
  tempo: number,
  onCodeChange: (code: string) => void,
  setSelectedNotes: (notes: SelectedNote[]) => void
): void {
  if (selectedNotes.length === 0) return;

  // Group deletions by track
  const byTrack = new Map<string, number[]>();
  selectedNotes.forEach(({ trackId, noteIndex }) => {
    if (!byTrack.has(trackId)) {
      byTrack.set(trackId, []);
    }
    byTrack.get(trackId)!.push(noteIndex);
  });

  let newCode = dslCode;

  // Delete notes from each track (sort indices descending to avoid index shifting)
  byTrack.forEach((indices, trackId) => {
    const notes = parseNotesFromDSL(newCode, trackId, tempo);
    indices.sort((a, b) => b - a).forEach(idx => {
      notes.splice(idx, 1);
    });
    newCode = updateDSLWithNewNotes(newCode, trackId, notes, tempo);
  });

  onCodeChange(newCode);
  setSelectedNotes([]);
}

/**
 * Handle note copy (supports multiple selections)
 */
export function handleCopyNotes(
  selectedNotes: SelectedNote[],
  dslCode: string,
  tempo: number,
  setCopiedNotes: (notes: CopiedNotes | null) => void
): void {
  if (selectedNotes.length === 0) return;

  const copiedData: Array<{ note: TimelineNote; trackId: string }> = [];
  let minStart = Infinity;

  selectedNotes.forEach(({ trackId, noteIndex }) => {
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const note = notes[noteIndex];
    if (note) {
      copiedData.push({
        note: { ...note },
        trackId
      });
      minStart = Math.min(minStart, note.start);
    }
  });

  if (copiedData.length > 0) {
    setCopiedNotes({
      notes: copiedData,
      minStart
    });
  }
}

/**
 * Handle note paste at playhead (supports multiple notes)
 */
export function handlePasteNotes(
  copiedNotes: CopiedNotes | null,
  currentTime: number,
  dslCode: string,
  tempo: number,
  snapValue: number,
  snapEnabled: boolean,
  onCodeChange: (code: string) => void,
  setSelectedNotes: (notes: SelectedNote[]) => void
): void {
  if (!copiedNotes || copiedNotes.notes.length === 0) return;

  const pastePosition = secondsToBeats(currentTime, tempo);
  const snappedPosition = snapToGrid(pastePosition, snapValue, snapEnabled);
  const offset = snappedPosition - copiedNotes.minStart;

  // Group notes by track
  const byTrack = new Map<string, TimelineNote[]>();
  copiedNotes.notes.forEach(({ note, trackId }) => {
    if (!byTrack.has(trackId)) {
      byTrack.set(trackId, parseNotesFromDSL(dslCode, trackId, tempo));
    }
    const newNote: TimelineNote = {
      ...note,
      start: note.start + offset
    };
    byTrack.get(trackId)!.push(newNote);
  });

  let newCode = dslCode;
  const newSelectedNotes: SelectedNote[] = [];

  // Update each track with pasted notes
  byTrack.forEach((notes, trackId) => {
    notes.sort((a, b) => a.start - b.start);
    newCode = updateDSLWithNewNotes(newCode, trackId, notes, tempo);

    // Select the newly pasted notes
    copiedNotes.notes.filter(n => n.trackId === trackId).forEach(({ note }) => {
      const newNoteIndex = notes.findIndex(n =>
        Math.abs(n.start - (note.start + offset)) < 0.001 && n.pitch === note.pitch
      );
      if (newNoteIndex !== -1) {
        newSelectedNotes.push({ trackId, noteIndex: newNoteIndex });
      }
    });
  });

  onCodeChange(newCode);
  setSelectedNotes(newSelectedNotes);
}

/**
 * Calculate new note position during drag (without applying to DSL)
 * Used for live visual feedback
 */
export function calculateNoteDrag(
  deltaX: number,
  zoom: number,
  initialStart: number,
  snapValue: number,
  snapEnabled: boolean
): number {
  const deltaBeats = deltaX / zoom;
  const rawStart = initialStart + deltaBeats;
  return snapToGrid(Math.max(0, rawStart), snapValue, snapEnabled);
}

/**
 * Calculate new note duration during resize (without applying to DSL)
 * Used for live visual feedback
 */
export function calculateNoteResize(
  deltaX: number,
  zoom: number,
  initialDuration: number,
  snapValue: number,
  snapEnabled: boolean
): number {
  const deltaBeats = deltaX / zoom;
  const rawDuration = initialDuration + deltaBeats;
  return Math.max(snapValue, snapToGrid(rawDuration, snapValue, snapEnabled));
}

/**
 * Handle note drag movement (final apply to DSL)
 */
export function handleNoteDrag(
  deltaX: number,
  zoom: number,
  draggingNote: { trackId: string; noteIndex: number; startX: number; initialStart: number },
  dslCode: string,
  tempo: number,
  snapValue: number,
  snapEnabled: boolean,
  onCodeChange: (code: string) => void
): void {
  const deltaBeats = deltaX / zoom;
  const rawStart = draggingNote.initialStart + deltaBeats;
  const newStart = snapToGrid(Math.max(0, rawStart), snapValue, snapEnabled);

  const notes = parseNotesFromDSL(dslCode, draggingNote.trackId, tempo);
  notes[draggingNote.noteIndex].start = newStart;

  const newCode = updateDSLWithNewNotes(dslCode, draggingNote.trackId, notes, tempo);
  onCodeChange(newCode);
}

/**
 * Handle multi-selection drag movement (final apply to DSL)
 */
export function handleMultiNoteDrag(
  deltaBeats: number,
  selectedNotes: Array<{ trackId: string; noteIndex: number }>,
  dslCode: string,
  tempo: number,
  snapValue: number,
  snapEnabled: boolean,
  onCodeChange: (code: string) => void
): void {
  // Group notes by track
  const byTrack = new Map<string, number[]>();
  selectedNotes.forEach(({ trackId, noteIndex }) => {
    if (!byTrack.has(trackId)) {
      byTrack.set(trackId, []);
    }
    byTrack.get(trackId)!.push(noteIndex);
  });

  let newCode = dslCode;

  // Update each track
  byTrack.forEach((indices, trackId) => {
    const notes = parseNotesFromDSL(newCode, trackId, tempo);
    indices.forEach(idx => {
      const rawStart = notes[idx].start + deltaBeats;
      notes[idx].start = snapToGrid(Math.max(0, rawStart), snapValue, snapEnabled);
    });
    newCode = updateDSLWithNewNotes(newCode, trackId, notes, tempo);
  });

  onCodeChange(newCode);
}

/**
 * Handle note resize
 */
export function handleNoteResize(
  deltaX: number,
  zoom: number,
  resizingNote: { trackId: string; noteIndex: number; startX: number; initialDuration: number },
  dslCode: string,
  tempo: number,
  snapValue: number,
  snapEnabled: boolean,
  onCodeChange: (code: string) => void
): void {
  const deltaBeats = deltaX / zoom;
  const rawDuration = resizingNote.initialDuration + deltaBeats;
  const newDuration = Math.max(snapValue, snapToGrid(rawDuration, snapValue, snapEnabled));

  const notes = parseNotesFromDSL(dslCode, resizingNote.trackId, tempo);
  notes[resizingNote.noteIndex].duration = newDuration;

  const newCode = updateDSLWithNewNotes(dslCode, resizingNote.trackId, notes, tempo);
  onCodeChange(newCode);
}
