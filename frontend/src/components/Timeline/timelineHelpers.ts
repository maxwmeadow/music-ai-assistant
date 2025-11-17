import { TimelineNote } from "./types";

/**
 * Extract tempo from DSL code
 */
export function getTempoFromDSL(dslCode: string): number {
  const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
  return tempoMatch ? parseInt(tempoMatch[1]) : 120;
}

/**
 * Convert beats to seconds based on tempo
 */
export function beatsToSeconds(beats: number, tempo: number): number {
  return (beats * 60) / tempo;
}

/**
 * Convert seconds to beats based on tempo
 */
export function secondsToBeats(seconds: number, tempo: number): number {
  return (seconds * tempo) / 60;
}

/**
 * Snap beats to grid based on snap value
 */
export function snapToGrid(beats: number, snapValue: number, snapEnabled: boolean): number {
  if (!snapEnabled) return beats;
  return Math.round(beats / snapValue) * snapValue;
}

/**
 * Determine grid subdivision level based on snap value (when enabled) or zoom
 */
export function getGridSubdivision(zoom: number, snapValue: number, snapEnabled: boolean): { subdivision: number } {
  // If snap is enabled, use the snap value to determine grid lines
  if (snapEnabled) {
    return { subdivision: snapValue };
  }

  // Otherwise, use zoom-based grid (original behavior)
  if (zoom < 60) {
    return { subdivision: 1 };
  } else if (zoom < 120) {
    return { subdivision: 0.5 };
  } else {
    return { subdivision: 0.25 };
  }
}

/**
 * Convert note name to MIDI pitch number
 */
export function noteToPitch(noteName: string): number {
  const noteMap: Record<string, number> = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
  };
  const match = noteName.match(/([A-G]#?)(\d+)/);
  if (!match) return 60;
  const [, note, octave] = match;
  return (parseInt(octave) + 1) * 12 + noteMap[note];
}

/**
 * Convert MIDI pitch number to note name
 */
export function pitchToNote(pitch: number): string {
  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(pitch / 12) - 1;
  const note = notes[pitch % 12];
  return `${note}${octave}`;
}

/**
 * Parse notes from DSL code for a specific track
 */
export function parseNotesFromDSL(dslCode: string, trackId: string, tempo: number): TimelineNote[] {
  const trackMatch = dslCode.match(new RegExp(`track\\("${trackId}"\\)\\s*{([^}]+)}`, 's'));
  if (!trackMatch) return [];

  const trackContent = trackMatch[1];
  const notes: TimelineNote[] = [];

  // Parse regular notes
  const noteMatches = trackContent.matchAll(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
  for (const match of noteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const pitch = noteToPitch(noteName);

    notes.push({
      pitch,
      start: secondsToBeats(parseFloat(start), tempo),
      duration: secondsToBeats(parseFloat(duration), tempo),
      velocity: parseFloat(velocity)
    });
  }

  // Parse chords
  const chordMatches = trackContent.matchAll(/chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g);
  for (const match of chordMatches) {
    const [, notesStr, start, duration, velocity] = match;
    const chordNotes = notesStr.split(',').map(n => n.trim().replace(/"/g, ''));
    const rootPitch = noteToPitch(chordNotes[0]);

    notes.push({
      pitch: rootPitch,
      start: secondsToBeats(parseFloat(start), tempo),
      duration: secondsToBeats(parseFloat(duration), tempo),
      velocity: parseFloat(velocity),
      isChord: true
    });
  }

  return notes.sort((a, b) => a.start - b.start);
}

/**
 * Update DSL code with new notes for a track
 */
export function updateDSLWithNewNotes(
  dslCode: string,
  trackId: string,
  updatedNotes: TimelineNote[],
  tempo: number
): string {
  const trackMatch = dslCode.match(new RegExp(`(track\\("${trackId}"\\)\\s*{)([^}]+)(})`, 's'));
  if (!trackMatch) return dslCode;

  const [fullMatch, opening, , closing] = trackMatch;

  const instrumentMatch = trackMatch[2].match(/instrument\("([^"]+)"\)/);
  const instrumentLine = instrumentMatch ? `  instrument("${instrumentMatch[1]}")\n` : '';

  // Generate new note lines, converting beats back to seconds
  const noteLines = updatedNotes.map(note => {
    const noteName = pitchToNote(note.pitch);
    const startSeconds = beatsToSeconds(note.start, tempo);
    const durationSeconds = beatsToSeconds(note.duration, tempo);

    if (note.isChord) {
      return `  chord(["${noteName}"], ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(1)})`;
    }
    return `  note("${noteName}", ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(1)})`;
  }).join('\n');

  const newTrackContent = `${opening}\n${instrumentLine}${noteLines}\n${closing}`;
  return dslCode.replace(fullMatch, newTrackContent);
}
