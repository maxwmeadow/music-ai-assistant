/**
 * MIDI Import functionality
 * Converts standard MIDI files to IR (Intermediate Representation)
 */

import { Midi } from '@tonejs/midi';

/**
 * Reverse mapping: General MIDI program number to instrument path
 * Maps to available instruments in the Music AI Assistant
 */
const PROGRAM_TO_INSTRUMENT: Record<number, string> = {
  // Piano (0-7)
  0: 'piano/steinway_grand',
  1: 'piano/steinway_grand',
  2: 'piano/steinway_grand',
  3: 'piano/steinway_grand',
  4: 'piano/steinway_grand',
  5: 'piano/steinway_grand',
  6: 'piano/steinway_grand',
  7: 'piano/steinway_grand',

  // Guitar (24-31)
  24: 'guitar/rjs_guitar_new_strings',
  25: 'guitar/rjs_guitar_new_strings',
  26: 'guitar/rjs_guitar_new_strings',
  27: 'guitar/rjs_guitar_new_strings',
  28: 'guitar/rjs_guitar_new_strings',
  29: 'guitar/rjs_guitar_new_strings',
  30: 'guitar/rjs_guitar_new_strings',
  31: 'guitar/rjs_guitar_new_strings',

  // Bass (32-39)
  32: 'bass/bass_synth',
  33: 'bass/bass_synth',
  34: 'bass/bass_synth',
  35: 'bass/bass_synth',
  36: 'bass/bass_synth',
  37: 'bass/bass_synth',
  38: 'bass/bass_synth',
  39: 'bass/bass_synth',
};

/**
 * Import MIDI file and convert to IR
 * @param file - MIDI file object
 * @returns IR object with tracks and metadata
 */
export async function importFromMIDI(file: File): Promise<any> {
  try {
    // Read file as ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();

    // Parse MIDI
    const midi = new Midi(arrayBuffer);

    console.log(`[MIDI Import] Parsing MIDI file: ${file.name}`);
    console.log(`[MIDI Import] Tracks: ${midi.tracks.length}, Duration: ${midi.duration}s`);

    // Extract tempo (use first tempo or default)
    const tempo = midi.header.tempos.length > 0
      ? Math.round(midi.header.tempos[0].bpm)
      : 120;

    // Extract time signature
    let timeSignature = '4/4';
    if (midi.header.timeSignatures.length > 0) {
      const ts = midi.header.timeSignatures[0].timeSignature;
      timeSignature = `${ts[0]}/${ts[1]}`;
    }

    // Convert tracks
    const tracks: any[] = [];
    midi.tracks.forEach((midiTrack, index) => {
      // Skip empty tracks
      if (midiTrack.notes.length === 0) {
        console.log(`[MIDI Import] Skipping empty track ${index}`);
        return;
      }

      // Determine instrument
      let instrument: string | null = null;
      const isDrumTrack = midiTrack.channel === 9; // MIDI channel 10 (0-indexed)

      if (!isDrumTrack && midiTrack.instrument) {
        const program = midiTrack.instrument.number;
        instrument = PROGRAM_TO_INSTRUMENT[program] || 'piano/steinway_grand';
      }

      // Convert notes
      const notes = midiTrack.notes.map(note => ({
        pitch: note.midi,
        start: note.time,
        duration: note.duration,
        velocity: Math.round(note.velocity * 127), // Convert back to 0-127
      }));

      // Create track object
      const trackId = midiTrack.name || `track_${index + 1}`;
      tracks.push({
        id: trackId,
        instrument: instrument,
        notes: notes,
        samples: null,
      });

      console.log(`[MIDI Import] Imported track "${trackId}": ${notes.length} notes, instrument: ${instrument || 'none'}`);
    });

    // Create IR object
    const ir = {
      metadata: {
        title: file.name.replace(/\.mid(i)?$/, ''),
        tempo: tempo,
        key: 'C', // MIDI doesn't reliably encode key
        timeSignature: timeSignature,
      },
      tracks: tracks,
    };

    console.log(`[MIDI Import] Successfully imported ${tracks.length} tracks`);
    return ir;

  } catch (error) {
    console.error('[MIDI Import] Failed to import MIDI:', error);
    throw new Error('Failed to import MIDI file. Please ensure it is a valid MIDI file.');
  }
}

/**
 * Check if file is a valid MIDI file
 */
export function isValidMIDIFile(file: File): boolean {
  const validExtensions = ['.mid', '.midi'];
  return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
}

/**
 * Get MIDI file info without full import
 * @param file - MIDI file
 * @returns Basic info about the MIDI file
 */
export async function getMIDIFileInfo(file: File): Promise<{
  name: string;
  duration: number;
  trackCount: number;
  tempo: number;
  noteCount: number;
}> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const midi = new Midi(arrayBuffer);

    const noteCount = midi.tracks.reduce((sum, track) => sum + track.notes.length, 0);
    const tempo = midi.header.tempos.length > 0
      ? Math.round(midi.header.tempos[0].bpm)
      : 120;

    return {
      name: file.name,
      duration: midi.duration,
      trackCount: midi.tracks.filter(t => t.notes.length > 0).length,
      tempo: tempo,
      noteCount: noteCount,
    };
  } catch (error) {
    console.error('[MIDI Import] Failed to get file info:', error);
    throw new Error('Failed to read MIDI file');
  }
}
