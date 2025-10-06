export interface ParsedTrack {
  id: string;
  instrument: string | null;
  lineStart: number;
  lineEnd: number;
}

export function parseTracksFromDSL(dsl: string): ParsedTrack[] {
  const tracks: ParsedTrack[] = [];
  const lines = dsl.split('\n');

  let currentTrack: Partial<ParsedTrack> | null = null;
  let braceDepth = 0;

  lines.forEach((line, index) => {
    const trackMatch = line.match(/track\("([^"]+)"\)/);
    if (trackMatch && braceDepth === 0) {
      currentTrack = {
        id: trackMatch[1],
        instrument: null,
        lineStart: index,
      };
    }

    if (currentTrack) {
      const instrumentMatch = line.match(/instrument\("([^"]+)"\)/);
      if (instrumentMatch) {
        currentTrack.instrument = instrumentMatch[1];
      }

      if (line.includes('{')) braceDepth++;
      if (line.includes('}')) {
        braceDepth--;
        if (braceDepth === 0) {
          currentTrack.lineEnd = index;
          tracks.push(currentTrack as ParsedTrack);
          currentTrack = null;
        }
      }
    }
  });

  return tracks;
}