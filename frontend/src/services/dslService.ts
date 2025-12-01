/**
 * DSL Service - Handles music DSL parsing and analysis
 */

export class DSLService {
  /**
   * Expand loop constructs in DSL code
   * Supports time-based loops: loop(startTime, endTime) { note(pitch, relativeStart, duration, velocity) }
   */
  private static expandLoops(dslCode: string): string {
    let expandedCode = dslCode;
    const maxIterations = 100;
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
   * Extract track blocks from DSL with proper brace matching
   */
  private static extractTracks(dslCode: string): string[] {
    const tracks: string[] = [];
    const trackPattern = /track\("([^"]+)"\)\s*\{/g;
    let match;

    while ((match = trackPattern.exec(dslCode)) !== null) {
      const trackStart = match.index;
      const contentStart = trackPattern.lastIndex;

      // Find the matching closing brace
      let braceCount = 1;
      let pos = contentStart;

      while (pos < dslCode.length && braceCount > 0) {
        if (dslCode[pos] === '{') braceCount++;
        if (dslCode[pos] === '}') braceCount--;
        pos++;
      }

      if (braceCount === 0) {
        const trackBlock = dslCode.substring(trackStart, pos);
        tracks.push(trackBlock);
      }
    }

    return tracks;
  }

  /**
   * Calculate the maximum duration of music from DSL code
   */
  static calculateMaxDuration(code: string): number {
    // Expand loops first
    const expandedCode = this.expandLoops(code);

    let maxDuration = 0;
    const tracks = this.extractTracks(expandedCode);

    for (const trackBlock of tracks) {
      // Parse 4-parameter notes with absolute timing
      const noteRegex = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let noteMatch;

      while ((noteMatch = noteRegex.exec(trackBlock)) !== null) {
        const start = parseFloat(noteMatch[2]);
        const duration = parseFloat(noteMatch[3]);
        const endTime = start + duration;
        maxDuration = Math.max(maxDuration, endTime);
      }

      // Parse 4-parameter chords with absolute timing
      const chordRegex = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let chordMatch;

      while ((chordMatch = chordRegex.exec(trackBlock)) !== null) {
        const start = parseFloat(chordMatch[2]);
        const duration = parseFloat(chordMatch[3]);
        const endTime = start + duration;
        maxDuration = Math.max(maxDuration, endTime);
      }
    }

    return maxDuration;
  }

  /**
   * Extract tempo from DSL code
   */
  static getTempo(code: string): number {
    const tempoMatch = code.match(/tempo\((\d+)\)/);
    return tempoMatch ? parseInt(tempoMatch[1]) : 120;
  }

  /**
   * Extract all track IDs from DSL code
   */
  static getTrackIds(code: string): string[] {
    const trackIds: string[] = [];
    const trackRegex = /track\("([^"]+)"\)/g;
    let match;

    while ((match = trackRegex.exec(code)) !== null) {
      trackIds.push(match[1]);
    }

    return trackIds;
  }

  /**
   * Append a new track to existing DSL code
   * Preserves tempo and existing tracks
   */
  static appendTrack(existingDSL: string, newTrackDSL: string, trackName: string, instrument?: string): string {
    // Extract the new track block from the new DSL
    const trackBlocks = this.extractTracks(newTrackDSL);
    if (trackBlocks.length === 0) {
      throw new Error('No track found in new DSL');
    }

    // Take the first track and rename it
    let newTrack = trackBlocks[0];

    // Replace the track ID with the new name
    newTrack = newTrack.replace(/track\("([^"]+)"\)/, `track("${trackName}")`);

    // Replace instrument if specified
    if (instrument) {
      newTrack = newTrack.replace(/instrument\("([^"]+)"\)/, `instrument("${instrument}")`);
    }

    // If no existing DSL, just return the new track with tempo
    if (!existingDSL.trim()) {
      const tempo = this.getTempo(newTrackDSL);
      return `tempo(${tempo})\n\n${newTrack}`;
    }

    // Append to existing DSL (before the last closing brace if it exists, or just at the end)
    const trimmed = existingDSL.trim();
    return `${trimmed}\n\n${newTrack}`;
  }
}
