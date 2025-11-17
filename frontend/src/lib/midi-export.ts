/**
 * MIDI Export functionality
 * Converts IR (Intermediate Representation) to standard MIDI format
 */

import { Midi } from '@tonejs/midi';

/**
 * General MIDI instrument mapping
 * Maps instrument categories to General MIDI program numbers
 */
const GENERAL_MIDI_PROGRAMS: Record<string, number> = {
  // Piano (0-7)
  'piano': 0,
  'grand_piano': 0,
  'electric_piano': 4,

  // Chromatic Percussion (8-15)
  'vibraphone': 11,
  'marimba': 12,

  // Organ (16-23)
  'organ': 16,
  'hammond': 16,

  // Guitar (24-31)
  'guitar': 24,
  'acoustic_guitar': 24,
  'electric_guitar': 27,
  'bass': 32,

  // Bass (32-39)
  'acoustic_bass': 32,
  'electric_bass': 33,
  'synth_bass': 38,

  // Strings (40-47)
  'violin': 40,
  'cello': 42,
  'strings': 48,

  // Ensemble (48-55)
  'string_ensemble': 48,
  'synth_strings': 50,

  // Brass (56-63)
  'trumpet': 56,
  'trombone': 57,
  'sax': 64,

  // Synth Lead (80-87)
  'synth_lead': 80,
  'synth': 80,

  // Synth Pad (88-95)
  'pad': 88,
  'synth_pad': 88,

  // Default
  'default': 0,
};

/**
 * Map instrument path to General MIDI program number
 */
function getGeneralMidiProgram(instrument: string | null | undefined): number {
  if (!instrument) return 0;

  const lowerInstrument = instrument.toLowerCase();

  // Check for exact matches first
  for (const [key, program] of Object.entries(GENERAL_MIDI_PROGRAMS)) {
    if (lowerInstrument.includes(key)) {
      return program;
    }
  }

  // Default to Acoustic Grand Piano
  return 0;
}

/**
 * Export IR to MIDI file
 * @param ir - Intermediate Representation object
 * @param tempo - Optional tempo override (defaults to IR metadata tempo or 120)
 * @returns Uint8Array of MIDI file data
 */
export function exportToMIDI(ir: any, tempo?: number): Uint8Array {
  try {
    // Create new MIDI file
    const midi = new Midi();

    // Set tempo (from IR metadata or parameter)
    const bpm = tempo || ir.metadata?.tempo || 120;
    midi.header.setTempo(bpm);

    // Set time signature if available
    if (ir.metadata?.timeSignature) {
      const [numerator, denominator] = ir.metadata.timeSignature.split('/').map(Number);
      if (numerator && denominator) {
        midi.header.timeSignatures = [{
          ticks: 0,
          timeSignature: [numerator, denominator],
          measures: 0,
        }];
      }
    }

    // Process each track
    const tracks = ir.tracks || [];
    tracks.forEach((track: any, index: number) => {
      // Only process tracks with notes (skip sample-based tracks like drums)
      if (!track.notes || track.notes.length === 0) {
        console.log(`[MIDI Export] Skipping track ${track.id} - no notes`);
        return;
      }

      // Create MIDI track
      const midiTrack = midi.addTrack();
      midiTrack.name = track.id || `Track ${index + 1}`;

      // Determine if this is a drum track (channel 10)
      const isDrumTrack = track.id?.toLowerCase().includes('drum') ||
                          track.instrument?.toLowerCase().includes('drum');

      if (isDrumTrack) {
        midiTrack.channel = 9; // MIDI channel 10 (0-indexed as 9)
      } else {
        // Assign program change (instrument)
        const program = getGeneralMidiProgram(track.instrument);
        midiTrack.instrument.number = program;
      }

      // Add notes
      track.notes.forEach((note: any) => {
        // Validate note data
        if (typeof note.pitch !== 'number' ||
            typeof note.start !== 'number' ||
            typeof note.duration !== 'number') {
          console.warn(`[MIDI Export] Invalid note data:`, note);
          return;
        }

        // Convert velocity (0.0-1.0) to MIDI velocity (0-127)
        let velocity = 64; // Default medium velocity
        if (typeof note.velocity === 'number') {
          if (note.velocity <= 1) {
            // Assume 0.0-1.0 range
            velocity = Math.round(note.velocity * 127);
          } else {
            // Already in 0-127 range
            velocity = Math.round(note.velocity);
          }
        }
        velocity = Math.max(1, Math.min(127, velocity)); // Clamp to valid range

        // Add note to MIDI track
        midiTrack.addNote({
          midi: note.pitch,
          time: note.start,
          duration: note.duration,
          velocity: velocity / 127, // @tonejs/midi expects 0-1
        });
      });

      console.log(`[MIDI Export] Added track "${midiTrack.name}" with ${track.notes.length} notes`);
    });

    // Convert to array buffer
    const arrayBuffer = midi.toArray();
    console.log(`[MIDI Export] Generated MIDI file (${arrayBuffer.length} bytes, ${tracks.length} tracks)`);

    return arrayBuffer;

  } catch (error) {
    console.error('[MIDI Export] Failed to export MIDI:', error);
    throw new Error('Failed to export MIDI file');
  }
}

/**
 * Download MIDI file
 * @param ir - Intermediate Representation
 * @param filename - Optional filename (without extension)
 */
export function downloadMIDI(ir: any, filename?: string): void {
  try {
    const midiData = exportToMIDI(ir);

    // Create blob (convert to regular Uint8Array for compatibility)
    const blob = new Blob([new Uint8Array(midiData)], { type: 'audio/midi' });

    // Generate filename
    const title = ir.metadata?.title || 'project';
    const finalFilename = filename ? `${filename}.mid` : `${title}.mid`;

    // Create download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = finalFilename;
    document.body.appendChild(link);
    link.click();

    // Cleanup
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`[MIDI Export] Downloaded: ${finalFilename}`);
  } catch (error) {
    console.error('[MIDI Export] Download failed:', error);
    throw error;
  }
}
