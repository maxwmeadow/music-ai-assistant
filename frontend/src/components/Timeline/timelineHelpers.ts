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

  for (const match of noteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const startVal = parseFloat(start);
    const durationVal = parseFloat(duration);
    const velocityVal = parseFloat(velocity);

    const pitch = noteNameToPitch(noteName);  // Use noteNameToPitch to handle drum names
    notes.push({
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
export function updateDSLWithNewNotes(
  dslCode: string,
  trackId: string,
  updatedNotes: TimelineNote[],
  tempo: number
): string {
  const trackMatch = dslCode.match(new RegExp(`(track\\("${trackId}"\\)\\s*\\{)([\\s\\S]*?)(\\n\\})`, 'm'));
  if (!trackMatch) return dslCode;

  const [fullMatch, opening, trackContent, closing] = trackMatch;

  // Determine if this is a drum track
  const instrumentMatch = trackContent.match(/instrument\("([^"]+)"\)/);
  const instrument = instrumentMatch ? instrumentMatch[1] : undefined;
  const isDrum = isDrumTrack(trackId, instrument);

  // First, extract original chord/note structures from DSL to preserve them
  const contentWithoutLoops = trackContent.replace(/loop\s*\([^)]+\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g, '');
  const originalNoteStructures: string[] = [];

  const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
  const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;

  // Collect original note/chord structures in order
  const allMatches: Array<{index: number, type: 'note'|'chord', pitches: string[]}> = [];

  for (const match of contentWithoutLoops.matchAll(notePattern)) {
    allMatches.push({
      index: match.index!,
      type: 'note',
      pitches: [match[1]]
    });
  }

  for (const match of contentWithoutLoops.matchAll(chordPattern)) {
    const chordNotes = match[1].split(',').map(n => n.trim().replace(/"/g, ''));
    allMatches.push({
      index: match.index!,
      type: 'chord',
      pitches: chordNotes
    });
  }

  // Sort by position in DSL
  allMatches.sort((a, b) => a.index - b.index);

  // Filter out loop-generated notes - only update directly-written notes
  const directNotes = updatedNotes.filter(note => !note.isFromLoop);

  // Build array of updated note lines preserving original chord structure
  const updatedNoteLines: string[] = [];
  directNotes.forEach((note, idx) => {
    const startSeconds = beatsToSeconds(note.start, tempo);
    const durationSeconds = beatsToSeconds(note.duration, tempo);

    // Use original chord/note structure if available
    const originalStructure = allMatches[idx];

    if (originalStructure && originalStructure.type === 'chord') {
      // Preserve original chord notes
      const chordNotesStr = originalStructure.pitches.map(p => `"${p}"`).join(', ');
      const noteLine = `chord([${chordNotesStr}], ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(2)})`;
      updatedNoteLines.push(noteLine);
    } else {
      // Single note
      const noteName = isDrum ? pitchToDrumName(note.pitch) : pitchToNote(note.pitch);
      const noteLine = `note("${noteName}", ${startSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(2)})`;
      updatedNoteLines.push(noteLine);
    }
  });

  // Process track content line by line, replacing notes by index
  const lines = trackContent.split('\n');
  const newLines: string[] = [];
  let noteIndex = 0; // Track which note/chord we're on
  let insideLoop = false; // Track if we're inside a loop block
  let loopDepth = 0; // Track nested braces

  for (const line of lines) {
    const trimmedLine = line.trim();

    // Track loop blocks to avoid modifying notes inside them
    if (trimmedLine.startsWith('loop(')) {
      insideLoop = true;
      loopDepth = 0;
    }

    if (insideLoop) {
      // Count braces to track loop nesting
      loopDepth += (line.match(/{/g) || []).length;
      loopDepth -= (line.match(/}/g) || []).length;

      // Keep all loop content as-is
      newLines.push(line);

      // Exit loop when we've closed all braces
      if (loopDepth === 0 && trimmedLine.includes('}')) {
        insideLoop = false;
      }
      continue;
    }

    // Check if this is a note or chord line (outside loops)
    const noteMatch = trimmedLine.match(/^(note|chord)\((?:"[^"]+"|(?:\[[^\]]+\])),\s*([0-9.]+)/);
    if (noteMatch) {
      // Replace with the corresponding updated note (by index, not time)
      if (noteIndex < updatedNoteLines.length) {
        // Preserve original indentation
        const indent = line.match(/^(\s*)/)?.[1] || '    ';
        newLines.push(`${indent}${updatedNoteLines[noteIndex]}`);
        noteIndex++;
      } else {
        // Shouldn't happen, but keep original if we run out of updated notes
        newLines.push(line);
      }
    } else {
      // Keep all non-note lines (comments, blank lines, instrument)
      newLines.push(line);
    }
  }

  // Add any remaining new notes that weren't in the original
  if (noteIndex < updatedNoteLines.length) {
    for (let i = noteIndex; i < updatedNoteLines.length; i++) {
      newLines.push(`    ${updatedNoteLines[i]}`);
    }
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
