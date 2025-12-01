// Instrument catalog utilities
import catalogData from '../data/catalog.json';

export interface Instrument {
  name: string;
  category: string;
  path: string;
  samples: number;
  size_mb: number;
  preset_type?: string;
}

export interface CategorizedInstruments {
  [category: string]: Instrument[];
}

/**
 * Get all instruments grouped by category
 */
export function getInstrumentsByCategory(): CategorizedInstruments {
  const instruments = catalogData.instruments as Instrument[];
  const categorized: CategorizedInstruments = {};

  // Remove duplicates based on path
  const uniqueInstruments = instruments.filter(
    (instrument, index, self) =>
      index === self.findIndex((t) => t.path === instrument.path)
  );

  // Group by category
  uniqueInstruments.forEach((instrument) => {
    if (!categorized[instrument.category]) {
      categorized[instrument.category] = [];
    }
    categorized[instrument.category].push(instrument);
  });

  // Sort each category alphabetically by name
  Object.keys(categorized).forEach((category) => {
    categorized[category].sort((a, b) => a.name.localeCompare(b.name));
  });

  return categorized;
}

/**
 * Get all category names sorted
 */
export function getCategories(): string[] {
  const categorized = getInstrumentsByCategory();
  return Object.keys(categorized).sort();
}

/**
 * Search instruments by name or path
 */
export function searchInstruments(query: string): Instrument[] {
  const instruments = catalogData.instruments as Instrument[];
  const lowerQuery = query.toLowerCase();

  // Remove duplicates based on path
  const uniqueInstruments = instruments.filter(
    (instrument, index, self) =>
      index === self.findIndex((t) => t.path === instrument.path)
  );

  return uniqueInstruments.filter(
    (instrument) =>
      instrument.name.toLowerCase().includes(lowerQuery) ||
      instrument.path.toLowerCase().includes(lowerQuery)
  );
}

/**
 * Instrument catalog filtered by track type for AI arrangement
 * Organized by what each track type typically needs
 */
export const TRACK_TYPE_INSTRUMENTS: Record<string, string[]> = {
  // BASS - Low-register harmonic foundation
  bass: [
    // Acoustic bass
    "bass/jp8000_sawbass",
    "bass/jp8000_tribass",
    // Synth bass
    "synth/bass/bs_2010_house",
    "synth/bass/bs_another_analog_bass",
    "synth/bass/bs_corg_bass",
    "synth/bass/bs_deep_undertone",
    "synth/bass/bs_lead_bass_player",
    "synth/bass/bs_outrun_bass",
    "synth/bass/bs_thick_bass",
    "synth/bass/bs_ms20_bass"
  ],

  // CHORDS - Instruments good for playing chords
  chords: [
    // Pianos
    "piano/steinway_grand",
    "piano/grand_piano_s_model_b_1895",
    "piano/upright_piano_knight",
    "piano/upright_piano_y",
    // Guitars
    "guitar/rjs_guitar_new_strings",
    "guitar/rjs_guitar_old_strings",
    "guitar/rjs_guitar_palm_muted_softly_strings",
    "guitar/rjs_guitar_palm_muted_strings",
    // Synth keys
    "synth/keys/kb_dx_epiano",
    "synth/keys/kb_interstellar_on_a_budget",
    "synth/keys/kb_nord_string",
    "synth/keys/kb_rhode_less_traveled",
    "synth/keys/kb_synthetic_organ",
    "synth/keys/kb_synthetic_strings",
    "synth/keys/kb_the_organ_trail",
    // Harpsichords
    "harpsichord/harpsichord_english",
    "harpsichord/harpsichord_flemish",
    "harpsichord/harpsichord_french",
    "harpsichord/harpsichord_italian"
  ],

  // PAD - Sustained atmospheric textures
  pad: [
    // Synth pads
    "synth/pad/pd_airlock_leak",
    "synth/pad/pd_event_horizon",
    "synth/pad/pd_every_80s_movie_ever",
    "synth/pad/pd_fatness_pad",
    "synth/pad/pd_on_the_horizon",
    "synth/pad/pd_orion_belt",
    "synth/pad/pd_soft_and_padded",
    "synth/pad/pd_the_first_pad",
    "synth/pad/pd_timeless_movement",
    // Strings (work well as pads)
    "strings/nfo_chamber_strings_longs",
    "strings/nfo_iso_celli_swells",
    "strings/nfo_iso_viola_swells",
    "strings/nfo_iso_violin_swells",
    // Synth keys with pad qualities
    "synth/keys/kb_nord_string",
    "synth/keys/kb_synthetic_strings"
  ],

  // MELODY - Lead melodic lines
  melody: [
    // Pianos
    "piano/steinway_grand",
    "piano/grand_piano_s_model_b_1895",
    "piano/upright_piano_knight",
    "piano/upright_piano_y",
    // Guitars
    "guitar/rjs_guitar_new_strings",
    "guitar/rjs_guitar_old_strings",
    // Synth leads
    "synth/lead/ld_cs80-ish",
    "synth/lead/ld_classic_saws",
    "synth/lead/ld_crystaline_80s",
    "synth/lead/ld_fm_hard_lead",
    "synth/lead/ld_for_each_loop",
    "synth/lead/ld_forever_80s",
    "synth/lead/ld_french_house",
    "synth/lead/ld_imfamousoty",
    "synth/lead/ld_jp_patchlead",
    "synth/lead/ld_juno_was_is",
    "synth/lead/ld_legecy_lead",
    "synth/lead/ld_poptab",
    "synth/lead/ld_strangest_things",
    "synth/lead/ld_synthetic_brass",
    "synth/lead/ld_the_nord",
    "synth/lead/ld_the_stack_guitar",
    "synth/lead/ld_uberheim_legend",
    // Synth keys
    "synth/keys/kb_dx_epiano",
    "synth/keys/kb_outrun_pluck",
    // Brass & winds
    "brass/nfo_iso_brass_swells",
    "winds/nfo_iso_wind_swells"
  ],

  // COUNTER-MELODY - Secondary melodic lines
  counterMelody: [
    // Strings (great for counter-melodies)
    "strings/nfo_chamber_strings_longs",
    "strings/nfo_iso_celli_swells",
    "strings/nfo_iso_viola_swells",
    "strings/nfo_iso_violin_swells",
    // Synth leads
    "synth/lead/ld_cs80-ish",
    "synth/lead/ld_classic_saws",
    "synth/lead/ld_crystaline_80s",
    "synth/lead/ld_poptab",
    "synth/lead/ld_strangest_things",
    "synth/lead/ld_synthetic_brass",
    // Brass & winds
    "brass/nfo_iso_brass_swells",
    "winds/nfo_iso_wind_swells",
    // Synth keys with melodic qualities
    "synth/keys/kb_dx_epiano",
    "synth/keys/kb_outrun_pluck"
  ],

  // ARPEGGIO - Broken chord patterns
  arpeggio: [
    // Harpsichords (perfect for arpeggios)
    "harpsichord/harpsichord_english",
    "harpsichord/harpsichord_flemish",
    "harpsichord/harpsichord_french",
    "harpsichord/harpsichord_italian",
    // Pianos
    "piano/steinway_grand",
    "piano/grand_piano_s_model_b_1895",
    // Plucky synths
    "synth/keys/kb_outrun_pluck",
    "synth/lead/ld_poptab",
    // Sequences (designed for rhythmic patterns)
    "synth/seq/sq_rhythmic_seq_1_(100bpm)",
    "synth/seq/sq_rhythmic_seq_2_(100bpm)"
  ],

  // DRUMS - Rhythmic percussion
  drums: [
    "drums/bedroom_drums",
    "drums/lorenzos_drums"
  ]
};

/**
 * Get instruments suitable for a specific track type
 */
export function getInstrumentsForTrackType(trackType: string): string[] {
  return TRACK_TYPE_INSTRUMENTS[trackType] || [];
}

/**
 * Get available instruments from catalog that match the track type
 */
export function getAvailableInstrumentsForTrackType(trackType: string): Instrument[] {
  const targetPaths = TRACK_TYPE_INSTRUMENTS[trackType] || [];
  const instruments = catalogData.instruments as Instrument[];

  // Remove duplicates based on path
  const uniqueInstruments = instruments.filter(
    (instrument, index, self) =>
      index === self.findIndex((t) => t.path === instrument.path)
  );

  // Filter to only instruments that match the track type
  return uniqueInstruments.filter(instrument =>
    targetPaths.some(path => instrument.path.includes(path.split('/')[0]))
  );
}
