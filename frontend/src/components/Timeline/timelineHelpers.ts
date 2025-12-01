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
 * Convert MIDI pitch to drum name for drum tracks (matches runner/server.js)
 */
export function pitchToDrumName(pitch: number): string {
  const drumMap: Record<number, string> = {
    36: 'kick',          // C2
    38: 'snare',         // D2
    40: 'snare_rimshot', // E2
    39: 'snare_buzz',    // D#2
    42: 'hihat_closed',  // F#2
    46: 'hihat_open',    // A#2
    44: 'hihat_pedal',   // G#2
    43: 'tom',           // G2
    49: 'crash',         // C#3
    51: 'ride',          // D#3
  };

  return drumMap[pitch] || pitchToNote(pitch);
}

/**
 * Convert note name to MIDI pitch, supporting both drum names and regular notes
 */
export function noteNameToPitch(noteName: string): number {
  // Check if it's a drum name first (matches runner/server.js)
  const drumNameMap: Record<string, number> = {
    'kick': 36,
    'snare': 38,
    'snare_rimshot': 40,
    'snare_buzz': 39,
    'hihat_closed': 42,
    'hihat_open': 46,
    'hihat_pedal': 44,
    'tom': 43,
    'crash': 49,
    'ride': 51,
    // Aliases
    'hihat': 42,  // Default to closed
  };

  const lowerName = noteName.toLowerCase();
  if (drumNameMap[lowerName]) {
    return drumNameMap[lowerName];
  }

  // Otherwise, parse as regular note
  return noteToPitch(noteName);
}

/**
 * Check if a track is a drum track based on its ID or instrument
 */
export function isDrumTrack(trackId: string, instrument?: string): boolean {
  const lowerTrackId = trackId.toLowerCase();
  const lowerInstrument = instrument?.toLowerCase() || '';

  return lowerTrackId.includes('drum') ||
         lowerInstrument.includes('drum') ||
         lowerTrackId === 'drums';
}

/**
 * Expand loop constructs in DSL code
 * Supports time-based loops: loop(startTime, endTime) { note(pitch, relativeStart, duration, velocity) }
 */
function expandLoops(dslCode: string): string {
  let expandedCode = dslCode;
  let maxIterations = 100;
  let iteration = 0;

  // Pattern for time-based loops
  const loopPattern = /loop\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g;
  let match;

  while ((match = loopPattern.exec(expandedCode)) !== null && iteration < maxIterations) {
    const fullMatch = match[0];
    const startTime = parseFloat(match[1]);
    const endTime = parseFloat(match[2]);
    const loopContent = match[3];

    let expandedContent = '';

    // Time-based loop: parse notes inside to get pattern duration
    const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
    const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;

    const noteMatches = Array.from(loopContent.matchAll(notePattern));
    const chordMatches = Array.from(loopContent.matchAll(chordPattern));

    if (noteMatches.length === 0 && chordMatches.length === 0) {
      // No notes found, skip this loop
      expandedCode = expandedCode.replace(fullMatch, loopContent);
      loopPattern.lastIndex = 0;
      iteration++;
      continue;
    }

    // Find the pattern duration (max relativeStart + duration)
    let patternDuration = 0;

    noteMatches.forEach(noteMatch => {
      const relativeStart = parseFloat(noteMatch[2]);
      const duration = parseFloat(noteMatch[3]);
      patternDuration = Math.max(patternDuration, relativeStart + duration);
    });

    chordMatches.forEach(chordMatch => {
      const relativeStart = parseFloat(chordMatch[2]);
      const duration = parseFloat(chordMatch[3]);
      patternDuration = Math.max(patternDuration, relativeStart + duration);
    });

    // Generate repeated notes from startTime to endTime
    const loopDuration = endTime - startTime;
    const repetitions = Math.ceil(loopDuration / patternDuration);

    for (let rep = 0; rep < repetitions; rep++) {
      const repStartTime = startTime + (rep * patternDuration);

      // Only add notes that fit within the loop range
      if (repStartTime >= endTime) break;

      // Expand notes
      noteMatches.forEach(noteMatch => {
        const pitch = noteMatch[1];
        const relativeStart = parseFloat(noteMatch[2]);
        const duration = parseFloat(noteMatch[3]);
        const velocity = parseFloat(noteMatch[4]);

        const absoluteStart = repStartTime + relativeStart;

        // Only include notes that start before endTime
        if (absoluteStart < endTime) {
          expandedContent += `  note("${pitch}", ${absoluteStart}, ${duration}, ${velocity})\n`;
        }
      });

      // Expand chords
      chordMatches.forEach(chordMatch => {
        const notes = chordMatch[1];
        const relativeStart = parseFloat(chordMatch[2]);
        const duration = parseFloat(chordMatch[3]);
        const velocity = parseFloat(chordMatch[4]);

        const absoluteStart = repStartTime + relativeStart;

        // Only include chords that start before endTime
        if (absoluteStart < endTime) {
          expandedContent += `  chord([${notes}], ${absoluteStart}, ${duration}, ${velocity})\n`;
        }
      });
    }

    // Replace the loop with expanded content
    expandedCode = expandedCode.replace(fullMatch, expandedContent);
    loopPattern.lastIndex = 0;
    iteration++;
  }

  return expandedCode;
}

/**
 * Parse notes from DSL code for a specific track
 * Parses both directly-written notes AND loop-generated notes
 * Loop-generated notes are marked with isFromLoop: true (read-only in timeline)
 */
export function parseNotesFromDSL(dslCode: string, trackId: string, tempo: number): TimelineNote[] {
  const trackMatch = dslCode.match(new RegExp(`track\\("${trackId}"\\)\\s*\\{([\\s\\S]*?)\\n\\}`, 'm'));
  if (!trackMatch) return [];

  const trackContent = trackMatch[1];
  const notes: TimelineNote[] = [];

  // First, parse directly-written notes (outside loops)
  const contentWithoutLoops = trackContent.replace(/loop\s*\([^)]+\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g, '');

  const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
  const noteMatches = [...contentWithoutLoops.matchAll(notePattern)];

  let noteIdCounter = 0; // Counter for generating unique IDs

  for (const match of noteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const startVal = parseFloat(start);
    const durationVal = parseFloat(duration);
    const velocityVal = parseFloat(velocity);

    const pitch = noteNameToPitch(noteName);  // Use noteNameToPitch to handle drum names
    notes.push({
      id: `${trackId}-note-${noteIdCounter++}`,
      pitch,
      start: secondsToBeats(startVal, tempo),
      duration: secondsToBeats(durationVal, tempo),
      velocity: velocityVal,
      isFromLoop: false
    });
  }

  const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
  const chordMatches = [...contentWithoutLoops.matchAll(chordPattern)];

  for (const match of chordMatches) {
    const [, chordStr, start, duration, velocity] = match;
    const startVal = parseFloat(start);
    const durationVal = parseFloat(duration);
    const velocityVal = parseFloat(velocity);

    const chordNotes = chordStr.split(',').map(n => n.trim().replace(/"/g, ''));
    const rootPitch = noteNameToPitch(chordNotes[0]);  // Use noteNameToPitch to handle drum names
    const allPitches = chordNotes.map(n => noteNameToPitch(n));
    notes.push({
      id: `${trackId}-note-${noteIdCounter++}`,
      pitch: rootPitch,
      start: secondsToBeats(startVal, tempo),
      duration: secondsToBeats(durationVal, tempo),
      velocity: velocityVal,
      isChord: true,
      chordPitches: allPitches,
      isFromLoop: false
    });
  }

  // Now expand loops and parse their notes (marked as read-only)
  const expandedDSL = expandLoops(trackContent);
  const loopOnlyContent = expandedDSL.replace(/instrument\([^)]+\)/g, ''); // Remove instrument line

  const loopNoteMatches = [...loopOnlyContent.matchAll(notePattern)];
  const directNoteStarts = new Set(notes.map(n => n.start.toFixed(3)));

  for (const match of loopNoteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const startVal = parseFloat(start);
    const durationVal = parseFloat(duration);
    const velocityVal = parseFloat(velocity);
    const startBeats = secondsToBeats(startVal, tempo);

    // Skip if this note was already added as a direct note
    if (directNoteStarts.has(startBeats.toFixed(3))) continue;

    const pitch = noteNameToPitch(noteName);  // Use noteNameToPitch to handle drum names
    notes.push({
      pitch,
      start: startBeats,
      duration: secondsToBeats(durationVal, tempo),
      velocity: velocityVal,
      isFromLoop: true
    });
  }

  const loopChordMatches = [...loopOnlyContent.matchAll(chordPattern)];

  for (const match of loopChordMatches) {
    const [, chordStr, start, duration, velocity] = match;
    const startVal = parseFloat(start);
    const durationVal = parseFloat(duration);
    const velocityVal = parseFloat(velocity);
    const startBeats = secondsToBeats(startVal, tempo);

    // Skip if this chord was already added as a direct chord
    if (directNoteStarts.has(startBeats.toFixed(3))) continue;

    const chordNotes = chordStr.split(',').map(n => n.trim().replace(/"/g, ''));
    const rootPitch = noteNameToPitch(chordNotes[0]);  // Use noteNameToPitch to handle drum names
    const allPitches = chordNotes.map(n => noteNameToPitch(n));
    notes.push({
      pitch: rootPitch,
      start: startBeats,
      duration: secondsToBeats(durationVal, tempo),
      velocity: velocityVal,
      isChord: true,
      chordPitches: allPitches,
      isFromLoop: true
    });
  }

  // Sort notes by start time
  notes.sort((a, b) => a.start - b.start);

  return notes;
}

/**
 * Update DSL code with new notes for a track
 * IMPORTANT: Preserves loop structures - only updates directly-written notes
 */
/**
 * Extract a single track with proper brace matching
 */
function extractTrackWithBraceMatching(dslCode: string, trackId: string): {
  fullMatch: string;
  opening: string;
  trackContent: string;
  closing: string;
  startIndex: number;
  endIndex: number;
} | null {
  const trackPattern = new RegExp(`track\\("${trackId}"\\)\\s*\\{`);
  const match = trackPattern.exec(dslCode);
  if (!match) return null;

  const trackStart = match.index;
  const contentStart = match.index + match[0].length;

  // Find the matching closing brace using brace counting
  let braceCount = 1;
  let pos = contentStart;

  while (pos < dslCode.length && braceCount > 0) {
    if (dslCode[pos] === '{') braceCount++;
    if (dslCode[pos] === '}') braceCount--;
    pos++;
  }

  if (braceCount !== 0) {
    return null;
  }

  const trackEnd = pos;
  const fullMatch = dslCode.substring(trackStart, trackEnd);
  const opening = match[0]; // "track("id") {"
  const closing = '}';
  const trackContent = dslCode.substring(contentStart, trackEnd - 1);

  return {
    fullMatch,
    opening,
    trackContent,
    closing,
    startIndex: trackStart,
    endIndex: trackEnd
  };
}

export function updateDSLWithNewNotes(
  dslCode: string,
  trackId: string,
  updatedNotes: TimelineNote[],
  tempo: number
): string {
  const extracted = extractTrackWithBraceMatching(dslCode, trackId);
  if (!extracted) return dslCode;

  const { fullMatch, opening, trackContent, closing } = extracted;

  // Determine if this is a drum track
  const instrumentMatch = trackContent.match(/instrument\("([^"]+)"\)/);
  const instrument = instrumentMatch ? instrumentMatch[1] : undefined;
  const isDrum = isDrumTrack(trackId, instrument);

  // Filter out loop-generated notes - only update directly-written notes
  const directNotes = updatedNotes.filter(note => !note.isFromLoop);

  // Parse original notes to get their IDs
  const originalNotes = parseNotesFromDSL(dslCode, trackId, tempo).filter(note => !note.isFromLoop);

  // Build map of updated notes by ID
  const updatedNotesById = new Map<string, TimelineNote>();
  directNotes.forEach(note => {
    if (note.id) {
      updatedNotesById.set(note.id, note);
    }
  });

  // Build map of original notes by index (order they appear in DSL)
  const originalNotesByIndex = new Map<number, TimelineNote>();
  originalNotes.forEach((note, index) => {
    originalNotesByIndex.set(index, note);
  });

  // Detect indentation from existing note/chord lines
  const lines = trackContent.split('\n');
  let detectedIndent = '    '; // Default to 4 spaces
  for (const line of lines) {
    const noteChordMatch = line.trim().match(/^(note|chord)\(/);
    if (noteChordMatch) {
      const indentMatch = line.match(/^(\s*)/);
      if (indentMatch && indentMatch[1]) {
        detectedIndent = indentMatch[1];
        break;
      }
    }
  }

  // Track which IDs have been processed
  const processedIds = new Set<string>();

  // Process track content line by line, preserving structure
  const newLines: string[] = [];
  let noteCounter = 0; // Counter for notes/chords seen (matches parsing order)

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmedLine = line.trim();

    // Check if this line starts a loop block
    if (trimmedLine.match(/^loop\s*\(/)) {
      // Find the matching closing brace for this loop
      let loopBraceCount = 0;
      let loopStartIndex = i;
      let loopEndIndex = i;

      // Count braces to find the loop's end
      for (let j = i; j < lines.length; j++) {
        const currentLine = lines[j];
        for (let k = 0; k < currentLine.length; k++) {
          if (currentLine[k] === '{') loopBraceCount++;
          if (currentLine[k] === '}') loopBraceCount--;
        }

        if (loopBraceCount === 0 && j > i) {
          loopEndIndex = j;
          break;
        }
      }

      // Preserve all lines from loop start to loop end (inclusive)
      for (let j = loopStartIndex; j <= loopEndIndex; j++) {
        newLines.push(lines[j]);
      }

      // Skip to the line after the loop
      i = loopEndIndex + 1;
      continue;
    }

    // Check if this is a note or chord line (outside of loops)
    const noteMatch = trimmedLine.match(/^(note|chord)\(/);
    if (noteMatch) {
      // Get the original note by index (order it appears in DSL)
      const originalNote = originalNotesByIndex.get(noteCounter);

      if (originalNote && originalNote.id) {
        // Look up the updated version by ID
        const updatedNote = updatedNotesById.get(originalNote.id);

        if (updatedNote) {
          // Note exists in updated list - update it in place
          const indentMatch = line.match(/^(\s*)/);
          const indent = indentMatch ? indentMatch[1] : detectedIndent;

          // Generate the updated line
          const timeSeconds = beatsToSeconds(updatedNote.start, tempo);
          const durationSeconds = beatsToSeconds(updatedNote.duration, tempo);

          let updatedLine: string;
          if (updatedNote.isChord && updatedNote.chordPitches && updatedNote.chordPitches.length > 1) {
            const noteNames = updatedNote.chordPitches.map(pitch => {
              const name = isDrum ? pitchToDrumName(pitch) : pitchToNote(pitch);
              return `"${name}"`;
            }).join(', ');
            updatedLine = `${indent}chord([${noteNames}], ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${updatedNote.velocity.toFixed(2)})`;
          } else {
            const noteName = isDrum ? pitchToDrumName(updatedNote.pitch) : pitchToNote(updatedNote.pitch);
            updatedLine = `${indent}note("${noteName}", ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${updatedNote.velocity.toFixed(2)})`;
          }

          newLines.push(updatedLine);
          processedIds.add(originalNote.id);
        } else {
          // Note was deleted - don't add to newLines
        }
      }

      noteCounter++;
    } else {
      // Keep all non-note lines (comments, blank lines, instrument)
      newLines.push(line);
    }

    i++;
  }

  // Add any new notes that weren't in the original
  // This includes: notes without IDs (brand new) or notes with IDs not yet processed
  const newNotes: string[] = [];
  directNotes.forEach(note => {
    if (!note.id || !processedIds.has(note.id)) {
      // This is a new note (either no ID or ID not matched)
      const timeSeconds = beatsToSeconds(note.start, tempo);
      const durationSeconds = beatsToSeconds(note.duration, tempo);

      let line: string;
      if (note.isChord && note.chordPitches && note.chordPitches.length > 1) {
        const noteNames = note.chordPitches.map(pitch => {
          const name = isDrum ? pitchToDrumName(pitch) : pitchToNote(pitch);
          return `"${name}"`;
        }).join(', ');
        line = `${detectedIndent}chord([${noteNames}], ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(2)})`;
      } else {
        const noteName = isDrum ? pitchToDrumName(note.pitch) : pitchToNote(note.pitch);
        line = `${detectedIndent}note("${noteName}", ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(2)})`;
      }

      newNotes.push(line);
    }
  });

  // Insert new notes at the end before closing brace
  if (newNotes.length > 0) {
    let insertIndex = newLines.length - 1;
    while (insertIndex >= 0 && newLines[insertIndex].trim() === '') {
      insertIndex--;
    }
    newLines.splice(insertIndex + 1, 0, ...newNotes);
  }

  // Remove leading empty lines (to avoid double blank after opening brace)
  while (newLines.length > 0 && newLines[0].trim() === '') {
    newLines.shift();
  }

  // Remove trailing empty lines (to avoid blank line before closing brace)
  while (newLines.length > 0 && newLines[newLines.length - 1].trim() === '') {
    newLines.pop();
  }

  const newTrackContent = `${opening}\n${newLines.join('\n')}\n${closing}`;
  return dslCode.replace(fullMatch, newTrackContent);
}
