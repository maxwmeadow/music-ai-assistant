/**
 * DSL Service - Handles music DSL parsing and analysis
 */

export class DSLService {
  /**
   * Calculate the maximum duration of music from DSL code
   */
  static calculateMaxDuration(code: string): number {
    let maxDuration = 0;
    const trackRegex = /track\("([^"]+)"\)\s*\{([^}]+)\}/g;
    let trackMatch;

    while ((trackMatch = trackRegex.exec(code)) !== null) {
      const trackContent = trackMatch[2];

      // Parse notes
      const noteRegex = /note\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let noteMatch;

      while ((noteMatch = noteRegex.exec(trackContent)) !== null) {
        const start = parseFloat(noteMatch[2]);
        const duration = parseFloat(noteMatch[3]);
        const endTime = start + duration;
        maxDuration = Math.max(maxDuration, endTime);
      }

      // Parse chords
      const chordRegex = /chord\(\[([^\]]+)\],\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)/g;
      let chordMatch;

      while ((chordMatch = chordRegex.exec(trackContent)) !== null) {
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
}
