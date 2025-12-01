import { TimelineNote } from "../Timeline/types";
import { beatsToSeconds, secondsToBeats, getTempoFromDSL, snapToGrid } from "../Timeline/timelineHelpers";

// Re-export Timeline helpers for convenience
export { beatsToSeconds, secondsToBeats, getTempoFromDSL, snapToGrid };

// Type alias for Piano Roll
export type PianoRollNote = TimelineNote;

export const MIDI_MIN = 36; // C2
export const MIDI_MAX = 108; // C8
export const NOTE_HEIGHT = 12; // pixels per semitone
export const PIANO_WIDTH = 60; // width of piano keys

// Pre-calculated gain compensation map (in dB)
export const INSTRUMENT_GAINS: Record<string, number> = {
  // Pianos
  'piano/steinway_grand': 0,
  'piano/bechstein_1911_upright': 0,
  'piano/fender_rhodes': 0,
  'piano/experience_ny_steinway': 0,
  // Harpsichords
  'harpsichord/harpsichord_english': -10,
  'harpsichord/harpsichord_flemish': -10,
  'harpsichord/harpsichord_french': -10,
  'harpsichord/harpsichord_italian': -15,
  'harpsichord/harpsichord_unk': -10,
  // Guitars
  'guitar/rjs_guitar_palm_muted_softly_strings': -5,
  'guitar/rjs_guitar_palm_muted_strings': -6,
  'synth/lead/ld_the_stack_guitar_chug': 0,
  'synth/lead/ld_the_stack_guitar': -1,
  'guitar/rjs_guitar_new_strings': 0,
  'guitar/rjs_guitar_old_strings': 4,
  // Bass
  'bass/funky_fingers': -10,
  'bass/low_fat_bass': -5,
  'bass/jp8000_sawbass': 2,
  'bass/jp8000_tribass': 2,
  // Strings
  'strings/nfo_chamber_strings_longs': 0,
  'strings/nfo_iso_celli_swells': 0,
  'strings/nfo_iso_viola_swells': 0,
  'strings/nfo_iso_violin_swells': 0,
  // Brass
  'brass/nfo_iso_brass_swells': 0,
  // Winds
  'winds/flute_violin': 0,
  'winds/subtle_clarinet': 0,
  'winds/decent_oboe': 0,
  'winds/tenor_saxophone': 0,
};

/**
 * Convert MIDI note name to pitch number
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
 * Convert pitch number to MIDI note name
 */
export function pitchToNote(pitch: number): string {
  const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(pitch / 12) - 1;
  const note = notes[pitch % 12];
  return `${note}${octave}`;
}

/**
 * Check if a pitch is a black key
 */
export function isBlackKey(pitch: number): boolean {
  const semitone = pitch % 12;
  return [1, 3, 6, 8, 10].includes(semitone); // C#, D#, F#, G#, A#
}

/**
 * Determine grid subdivision level based on zoom
 */
export function getGridSubdivision(zoom: number): {
  subdivision: number;
  showSubdivisions: boolean;
  showSixteenths: boolean
} {
  if (zoom < 60) {
    return { subdivision: 1, showSubdivisions: false, showSixteenths: false };
  } else if (zoom < 120) {
    return { subdivision: 0.5, showSubdivisions: true, showSixteenths: false };
  } else {
    return { subdivision: 0.25, showSubdivisions: true, showSixteenths: true };
  }
}

/**
 * Expand loop constructs in DSL code
 * Supports time-based loops: loop(startTime, endTime) { note(pitch, relativeStart, duration, velocity) }
 */
export function expandLoops(dslCode: string): string {
  let expandedCode = dslCode;
  let maxIterations = 100;
  let iteration = 0;

  const loopPattern = /loop\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g;
  let match;

  while ((match = loopPattern.exec(expandedCode)) !== null && iteration < maxIterations) {
    const fullMatch = match[0];
    const startTime = parseFloat(match[1]);
    const endTime = parseFloat(match[2]);
    const loopContent = match[3];

    let expandedContent = '';

    const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
    const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;

    const noteMatches = Array.from(loopContent.matchAll(notePattern));
    const chordMatches = Array.from(loopContent.matchAll(chordPattern));

    if (noteMatches.length === 0 && chordMatches.length === 0) {
      expandedCode = expandedCode.replace(fullMatch, loopContent);
      loopPattern.lastIndex = 0;
      iteration++;
      continue;
    }

    // Calculate pattern duration
    let patternDuration = 0;
    noteMatches.forEach(m => {
      const noteStart = parseFloat(m[2]);
      const noteDuration = parseFloat(m[3]);
      patternDuration = Math.max(patternDuration, noteStart + noteDuration);
    });
    chordMatches.forEach(m => {
      const chordStart = parseFloat(m[2]);
      const chordDuration = parseFloat(m[3]);
      patternDuration = Math.max(patternDuration, chordStart + chordDuration);
    });

    if (patternDuration === 0) patternDuration = 1;

    const loopDuration = endTime - startTime;
    const repetitions = Math.floor(loopDuration / patternDuration);

    // Generate repeated notes
    for (let i = 0; i < repetitions; i++) {
      const offset = startTime + (i * patternDuration);

      noteMatches.forEach(m => {
        const noteName = m[1];
        const noteStart = parseFloat(m[2]);
        const noteDuration = parseFloat(m[3]);
        const noteVelocity = parseFloat(m[4]);
        const absoluteStart = offset + noteStart;
        expandedContent += `  note("${noteName}", ${absoluteStart.toFixed(3)}, ${noteDuration.toFixed(3)}, ${noteVelocity})\n`;
      });

      chordMatches.forEach(m => {
        const chordNotes = m[1];
        const chordStart = parseFloat(m[2]);
        const chordDuration = parseFloat(m[3]);
        const chordVelocity = parseFloat(m[4]);
        const absoluteStart = offset + chordStart;
        expandedContent += `  chord([${chordNotes}], ${absoluteStart.toFixed(3)}, ${chordDuration.toFixed(3)}, ${chordVelocity})\n`;
      });
    }

    expandedCode = expandedCode.replace(fullMatch, expandedContent);
    loopPattern.lastIndex = 0;
    iteration++;
  }

  return expandedCode;
}

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
    console.error(`[updateDSL] Unmatched braces for track "${trackId}"`);
    return null;
  }

  const trackEnd = pos;
  const fullMatch = dslCode.substring(trackStart, trackEnd);
  const opening = match[0]; // "track("id") {"
  const closing = '}';
  const trackContent = dslCode.substring(contentStart, trackEnd - 1);

  console.log(`[updateDSL] Extracted track "${trackId}":`, {
    fullMatchLength: fullMatch.length,
    trackContentLength: trackContent.length,
    startIndex: trackStart,
    endIndex: trackEnd,
    trackContentPreview: trackContent.substring(0, 200) + '...'
  });

  return {
    fullMatch,
    opening,
    trackContent,
    closing,
    startIndex: trackStart,
    endIndex: trackEnd
  };
}

/**
 * Parse notes from DSL track content
 */
export function parseNotesFromDSL(dslCode: string, trackId: string, tempo: number): PianoRollNote[] {
  const extracted = extractTrackWithBraceMatching(dslCode, trackId);
  if (!extracted) {
    console.warn(`[parseNotes] Could not find track "${trackId}"`);
    return [];
  }

  const trackContent = extracted.trackContent;
  const notes: PianoRollNote[] = [];

  // Parse directly-written notes (outside loops) - these are editable
  const contentWithoutLoops = trackContent.replace(/loop\s*\([^)]+\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g, '');

  const notePattern = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
  const directNoteMatches = [...contentWithoutLoops.matchAll(notePattern)];

  for (const match of directNoteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const pitch = noteToPitch(noteName);

    notes.push({
      pitch,
      start: secondsToBeats(parseFloat(start), tempo),
      duration: secondsToBeats(parseFloat(duration), tempo),
      velocity: parseFloat(velocity),
      isFromLoop: false
    });
  }

  const chordPattern = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
  const directChordMatches = [...contentWithoutLoops.matchAll(chordPattern)];

  for (const match of directChordMatches) {
    const [, chordStr, start, duration, velocity] = match;
    const chordNotes = chordStr.split(',').map(n => n.trim().replace(/"/g, ''));

    chordNotes.forEach(noteName => {
      const pitch = noteToPitch(noteName);
      notes.push({
        pitch,
        start: secondsToBeats(parseFloat(start), tempo),
        duration: secondsToBeats(parseFloat(duration), tempo),
        velocity: parseFloat(velocity),
        isChord: true,
        isFromLoop: false
      });
    });
  }

  // Expand loops and parse their notes (marked as read-only)
  const expandedDSL = expandLoops(trackContent);
  const loopOnlyContent = expandedDSL.replace(/instrument\([^)]+\)/g, '');

  const loopNoteMatches = [...loopOnlyContent.matchAll(notePattern)];
  const directNoteStarts = new Set(notes.map(n => n.start.toFixed(3)));

  for (const match of loopNoteMatches) {
    const [, noteName, start, duration, velocity] = match;
    const startBeats = secondsToBeats(parseFloat(start), tempo);

    if (directNoteStarts.has(startBeats.toFixed(3))) continue;

    const pitch = noteToPitch(noteName);
    notes.push({
      pitch,
      start: startBeats,
      duration: secondsToBeats(parseFloat(duration), tempo),
      velocity: parseFloat(velocity),
      isFromLoop: true
    });
  }

  const loopChordMatches = [...loopOnlyContent.matchAll(chordPattern)];

  for (const match of loopChordMatches) {
    const [, chordStr, start, duration, velocity] = match;
    const startBeats = secondsToBeats(parseFloat(start), tempo);

    if (directNoteStarts.has(startBeats.toFixed(3))) continue;

    const chordNotes = chordStr.split(',').map(n => n.trim().replace(/"/g, ''));

    chordNotes.forEach(noteName => {
      const pitch = noteToPitch(noteName);
      notes.push({
        pitch,
        start: startBeats,
        duration: secondsToBeats(parseFloat(duration), tempo),
        velocity: parseFloat(velocity),
        isChord: true,
        isFromLoop: true
      });
    });
  }

  notes.sort((a, b) => a.start - b.start);
  return notes;
}

/**
 * Update DSL with new notes, preserving loop blocks
 */
export function updateDSLWithNewNotes(
  dslCode: string,
  trackId: string,
  updatedNotes: PianoRollNote[],
  tempo: number,
  onCodeChange: (newCode: string) => void
): void {
  console.log(`[updateDSL] Starting update for track "${trackId}"`, {
    totalNotes: updatedNotes.length,
    directNotes: updatedNotes.filter(n => !n.isFromLoop).length,
    loopNotes: updatedNotes.filter(n => n.isFromLoop).length
  });

  const extracted = extractTrackWithBraceMatching(dslCode, trackId);
  if (!extracted) {
    console.error(`[updateDSL] Could not extract track "${trackId}"`);
    return;
  }

  const { fullMatch, opening, trackContent, closing } = extracted;

  // Filter out loop-generated notes - only update directly-written notes
  const directNotes = updatedNotes.filter(note => !note.isFromLoop);

  console.log(`[updateDSL] Processing ${directNotes.length} direct notes`);

  // Group updated notes by time for chord detection
  const notesByTime = new Map<string, PianoRollNote[]>();
  directNotes.forEach(note => {
    const timeKey = note.start.toFixed(3);
    if (!notesByTime.has(timeKey)) {
      notesByTime.set(timeKey, []);
    }
    notesByTime.get(timeKey)!.push(note);
  });

  console.log(`[updateDSL] Grouped into ${notesByTime.size} time positions`);

  // Build a map of what each note/chord should look like
  const updatedNoteLines = new Map<string, string>();
  Array.from(notesByTime.entries()).forEach(([timeStr, notes]) => {
    const timeSeconds = beatsToSeconds(parseFloat(timeStr), tempo);
    const timeKey = timeSeconds.toFixed(3);

    if (notes.length === 1) {
      const note = notes[0];
      const noteName = pitchToNote(note.pitch);
      const durationSeconds = beatsToSeconds(note.duration, tempo);
      const line = `note("${noteName}", ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${note.velocity.toFixed(2)})`;
      updatedNoteLines.set(timeKey, line);
      console.log(`[updateDSL] Note at ${timeKey}s: ${line}`);
    } else {
      const noteNames = notes.map(n => `"${pitchToNote(n.pitch)}"`).join(', ');
      const avgDuration = notes.reduce((sum, n) => sum + n.duration, 0) / notes.length;
      const avgVelocity = notes.reduce((sum, n) => sum + n.velocity, 0) / notes.length;
      const durationSeconds = beatsToSeconds(avgDuration, tempo);
      const line = `chord([${noteNames}], ${timeSeconds.toFixed(3)}, ${durationSeconds.toFixed(3)}, ${avgVelocity.toFixed(2)})`;
      updatedNoteLines.set(timeKey, line);
      console.log(`[updateDSL] Chord at ${timeKey}s: ${line}`);
    }
  });

  // Process track content line by line, preserving structure
  const lines = trackContent.split('\n');
  const newLines: string[] = [];
  const processedTimes = new Set<string>();

  console.log(`[updateDSL] Processing ${lines.length} lines from track content`);
  console.log(`[updateDSL] First line: "${lines[0]}" (length: ${lines[0].length}, trimmed: "${lines[0].trim()}")`);
  console.log(`[updateDSL] Last line: "${lines[lines.length - 1]}" (length: ${lines[lines.length - 1].length}, trimmed: "${lines[lines.length - 1].trim()}")`);

  // Detect common indentation from existing note/chord lines
  let detectedIndent = '    '; // Default to 4 spaces
  for (const line of lines) {
    const noteChordMatch = line.trim().match(/^(note|chord)\(/);
    if (noteChordMatch) {
      const indentMatch = line.match(/^(\s*)/);
      if (indentMatch && indentMatch[1]) {
        detectedIndent = indentMatch[1];
        console.log(`[updateDSL] Detected indentation: ${detectedIndent.length} chars (from existing note/chord line)`);
        break;
      }
    }
  }

  // Helper to find matching time with tolerance
  const findMatchingTime = (targetTime: number): string | null => {
    const tolerance = 0.001; // 1ms tolerance
    for (const [key, _] of updatedNoteLines.entries()) {
      const keyTime = parseFloat(key);
      if (Math.abs(keyTime - targetTime) < tolerance) {
        return key;
      }
    }
    return null;
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmedLine = line.trim();

    // Check if this line starts a loop block
    if (trimmedLine.match(/^loop\s*\(/)) {
      console.log(`[updateDSL] Line ${i}: Found loop start, preserving entire loop block`);

      // Find the matching closing brace for this loop
      let loopBraceCount = 0;
      let loopStartIndex = i;
      let loopEndIndex = i;

      // Count braces to find the loop's end
      for (let j = i; j < lines.length; j++) {
        const currentLine = lines[j];
        // Count opening braces
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
        console.log(`[updateDSL] Line ${j}: Preserving loop content line`);
        newLines.push(lines[j]);
      }

      // Skip to the line after the loop
      i = loopEndIndex + 1;
      continue;
    }

    // Check if this is a note or chord line (outside of loops)
    // Match either: note("C4", 0.0, ...) or chord(["C4", "E4"], 0.0, ...)
    const noteMatch = trimmedLine.match(/^(note|chord)\((?:"[^"]+"|(?:\[[^\]]+\])),\s*([0-9.]+)/);
    if (noteMatch) {
      const timeSeconds = parseFloat(noteMatch[2]);

      // Detect indentation from original line
      const indentMatch = line.match(/^(\s*)/);
      const indent = indentMatch ? indentMatch[1] : '    '; // Default to 4 spaces

      // Find matching time key with tolerance
      const matchingKey = findMatchingTime(timeSeconds);

      // Check if we have an updated version of this note/chord
      if (matchingKey && !processedTimes.has(matchingKey)) {
        // Replace with updated version
        const updatedLine = `${indent}${updatedNoteLines.get(matchingKey)}`;
        console.log(`[updateDSL] Line ${i}: Replacing note/chord at ${timeSeconds}s with updated version (indent: ${indent.length} chars)`);
        newLines.push(updatedLine);
        processedTimes.add(matchingKey);
      } else if (!matchingKey) {
        // No update found - this note/chord was deleted, skip it
        console.log(`[updateDSL] Line ${i}: Deleting note/chord at ${timeSeconds}s (no notes at this time in updated list)`);
      } else {
        // Already processed - skip duplicate
        console.log(`[updateDSL] Line ${i}: Skipping duplicate note/chord at ${timeSeconds}s (already processed)`);
      }
    } else {
      // Keep all non-note lines (comments, blank lines, instrument)
      if (trimmedLine.includes('instrument(')) {
        console.log(`[updateDSL] Line ${i}: Preserving instrument: ${trimmedLine}`);
      } else if (trimmedLine.startsWith('//')) {
        console.log(`[updateDSL] Line ${i}: Preserving comment: ${trimmedLine}`);
      } else if (trimmedLine === '' || trimmedLine === '}') {
        console.log(`[updateDSL] Line ${i}: Preserving whitespace/brace`);
      } else {
        console.log(`[updateDSL] Line ${i}: Preserving other line: ${trimmedLine.substring(0, 50)}...`);
      }
      newLines.push(line);
    }

    i++;
  }

  console.log(`[updateDSL] Processed ${processedTimes.size} updated notes/chords`);

  // Add any new notes that weren't in the original
  const newNotes: string[] = [];
  for (const [timeKey, noteLine] of updatedNoteLines.entries()) {
    if (!processedTimes.has(timeKey)) {
      console.log(`[updateDSL] Adding new note/chord at ${timeKey}s: ${noteLine}`);
      newNotes.push(`${detectedIndent}${noteLine}`);
    }
  }

  console.log(`[updateDSL] Adding ${newNotes.length} new notes/chords`);

  // Insert new notes at the end before closing brace, maintaining structure
  if (newNotes.length > 0) {
    // Find the last non-empty line before the end
    let insertIndex = newLines.length - 1;
    while (insertIndex >= 0 && newLines[insertIndex].trim() === '') {
      insertIndex--;
    }
    console.log(`[updateDSL] Inserting new notes at index ${insertIndex + 1}`);
    newLines.splice(insertIndex + 1, 0, ...newNotes);
  }

  console.log(`[updateDSL] newLines array (before trim):`, {
    firstLine: `"${newLines[0]}"`,
    lastLine: `"${newLines[newLines.length - 1]}"`,
    totalLines: newLines.length
  });

  // Remove leading empty lines
  while (newLines.length > 0 && newLines[0].trim() === '') {
    console.log(`[updateDSL] Removing leading empty line`);
    newLines.shift();
  }

  // Remove trailing empty lines
  while (newLines.length > 0 && newLines[newLines.length - 1].trim() === '') {
    console.log(`[updateDSL] Removing trailing empty line`);
    newLines.pop();
  }

  console.log(`[updateDSL] newLines array (after trim):`, {
    firstLine: `"${newLines[0]}"`,
    lastLine: `"${newLines[newLines.length - 1]}"`,
    totalLines: newLines.length
  });

  const newTrackContent = `${opening}\n${newLines.join('\n')}\n${closing}`;
  const newDSL = dslCode.replace(fullMatch, newTrackContent);

  console.log(`[updateDSL] Rebuilt track:`, {
    originalLines: lines.length,
    newLines: newLines.length,
    newTrackContentLength: newTrackContent.length,
    newTrackContentPreview: newTrackContent.substring(0, 300) + '...',
    newTrackContentFirst100: newTrackContent.substring(0, 100).replace(/\n/g, '\\n')
  });

  console.log(`[updateDSL] Update complete for track "${trackId}"`);
  onCodeChange(newDSL);
}

/**
 * Parse instrument from DSL code
 */
export function getInstrumentFromDSL(dslCode: string, trackId: string): string | null {
  const trackMatch = dslCode.match(new RegExp(`track\\s*\\(\\s*["']${trackId}["']\\s*\\)\\s*\\{([\\s\\S]*?)\\}`, 'm'));
  if (!trackMatch) return null;
  const trackContent = trackMatch[1];
  const instrumentMatch = trackContent.match(/instrument\s*\(\s*["']([^"']+)["']\s*\)/);
  return instrumentMatch ? instrumentMatch[1] : null;
}

/**
 * Build sampler URLs from instrument mapping
 */
export function buildSamplerUrls(mapping: any, instrumentPath: string, cdnBase: string) {
  const urls: Record<string, string> = {};
  const baseUrl = `${cdnBase}/samples/${instrumentPath}/`;

  if (mapping.velocity_layers) {
    for (const [note, layers] of Object.entries(mapping.velocity_layers)) {
      const layerArray = layers as any[];

      const sampleLayer = layerArray.find((l: any) =>
        l.file.includes('Sustains') || l.file.includes('sus')
      ) || layerArray[0];

      if (!sampleLayer) continue;

      const pitchCenter = sampleLayer.pitch_center;

      if (pitchCenter !== undefined) {
        const noteName = pitchToNote(pitchCenter);
        urls[noteName] = sampleLayer.file.split('/').map(encodeURIComponent).join('/');
      }
    }
  } else if (mapping.samples) {
    for (const [note, file] of Object.entries(mapping.samples)) {
      urls[note] = (file as string).split('/').map(encodeURIComponent).join('/');
    }
  }

  return { urls, baseUrl };
}

/**
 * Analyze instrument gain levels
 */
export async function analyzeInstrumentGain(urls: Record<string, string>, baseUrl: string): Promise<number> {
  try {
    const Tone = await import('tone');
    const sampleKeys = Object.keys(urls).slice(0, Math.min(3, Object.keys(urls).length));
    if (sampleKeys.length === 0) return 0;

    const audioContext = Tone.context.rawContext;
    const peakLevels: number[] = [];

    for (const key of sampleKeys) {
      try {
        const sampleUrl = baseUrl + urls[key];
        const response = await fetch(sampleUrl);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        let maxPeak = 0;
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
          const channelData = audioBuffer.getChannelData(channel);
          for (let i = 0; i < channelData.length; i++) {
            maxPeak = Math.max(maxPeak, Math.abs(channelData[i]));
          }
        }

        peakLevels.push(maxPeak);
        console.log(`[PianoRoll Gain] Sample ${key}: peak = ${maxPeak.toFixed(6)} (${(20 * Math.log10(maxPeak)).toFixed(1)}dB)`);
      } catch (err) {
        console.warn(`[PianoRoll Gain] Failed to analyze ${key}:`, err);
      }
    }

    if (peakLevels.length === 0) return 0;

    const avgPeak = peakLevels.reduce((sum, p) => sum + p, 0) / peakLevels.length;
    const avgPeakDb = 20 * Math.log10(avgPeak);
    const targetDb = -6;
    const neededGain = targetDb - avgPeakDb;

    console.log(`[PianoRoll Gain] Average peak: ${avgPeak.toFixed(6)} (${avgPeakDb.toFixed(1)}dB)`);
    console.log(`[PianoRoll Gain] Recommended gain: ${neededGain.toFixed(1)}dB`);

    return Math.max(-20, Math.min(60, neededGain));
  } catch (error) {
    console.error('[PianoRoll Gain] Error:', error);
    return 0;
  }
}

/**
 * Coordinate conversion helpers
 */
export const timeToX = (beats: number, zoom: number) => beats * zoom;
export const xToTime = (x: number, zoom: number) => x / zoom;
export const pitchToY = (pitch: number) => (MIDI_MAX - pitch) * NOTE_HEIGHT;
export const yToPitch = (y: number) => Math.round(MIDI_MAX - y / NOTE_HEIGHT);