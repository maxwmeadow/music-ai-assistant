/**
 * Detailed descriptions for each instrument
 * Used to provide context to the LLM for better instrument selection
 */

export const INSTRUMENT_DESCRIPTIONS: Record<string, string> = {
  // === DRUMS ===
  "drums/lorenzos_drums":
    "Yamaha drum kit - punchy and snappy with individual mic channels (overhead, snare, kick). " +
    "Great for rock, metal, and most genres. Raw, authentic sound with 5 velocity layers. " +
    "DSL-supported sounds: kick, snare, hihat_closed, hihat_open, tom, crash, ride. " +
    "MISSING: snare_rimshot, snare_buzz, hihat_pedal (use bedroom_drums for complete kit).",

  "drums/bedroom_drums":
    "Sonor Martini Mini Kit - simple, organic drum sound. Punchy kick, snappy snare, bright cymbals. " +
    "Not over-produced, great for adding your own character. Complete kit with 3 velocity layers. " +
    "DSL-supported sounds: kick, snare, snare_rimshot, snare_buzz, hihat_closed, hihat_open, hihat_pedal, tom, crash, ride. " +
    "Complete DSL drum kit - supports ALL 10 standard drum names.",

  // === PIANOS ===
  "piano/steinway_grand":
    "Steinway Grand - tight, warm tone with rich, full-bodied sound. " +
    "Excellent dynamic range. Classic concert piano suitable for all genres. Range: F#0-F7.",

  "piano/grand_piano_s_model_b_1895":
    "Steinway Model B (1895) - vintage grand piano with warm, characterful tone. " +
    "Historic instrument with beautiful resonance.",

  "piano/upright_piano_knight":
    "Knight upright piano - intimate, closer mic'd sound. " +
    "Great for singer-songwriter and contemporary music.",

  "piano/upright_piano_y":
    "Yamaha upright piano - bright, articulate tone with consistent response. " +
    "Versatile for pop, rock, and classical.",

  // === HARPSICHORDS ===
  "harpsichord/harpsichord_english":
    "English harpsichord - bright, crisp plucked sound. " +
    "Perfect for baroque music and arpeggiated patterns.",

  "harpsichord/harpsichord_flemish":
    "Flemish harpsichord - fuller, rounder tone than English style. " +
    "Rich harmonic content, great for counterpoint.",

  "harpsichord/harpsichord_french":
    "French harpsichord - elegant, delicate plucked sound. " +
    "Classic baroque timbre.",

  "harpsichord/harpsichord_italian":
    "Italian harpsichord - light, transparent sound. " +
    "Perfect for early music and delicate passages.",

  // === GUITARS ===
  "guitar/rjs_guitar_new_strings":
    "Sampled Stratocaster with new strings - bright, clear tone. " +
    "Multiple articulations (vibrato, palm mute, dead notes). Great for leads and rhythm.",

  "guitar/rjs_guitar_old_strings":
    "Sampled Stratocaster with old strings - warm, mellower tone than new strings. " +
    "Multiple articulations. Works great in the mix.",

  "guitar/rjs_guitar_palm_muted_softly_strings":
    "Sampled Stratocaster palm muted (soft) - tight, percussive rhythm sound. " +
    "Perfect for clean rhythm parts and funk.",

  "guitar/rjs_guitar_palm_muted_strings":
    "Sampled Stratocaster palm muted (hard) - tight, aggressive rhythm sound. " +
    "Great for rock and metal rhythm parts.",

  // === BASS ===
  "bass/jp8000_sawbass":
    "Roland JP-8000 sawtooth bass - bright, cutting analog sound with rich harmonics. " +
    "Classic analog synth bass, great for electronic and dance music.",

  "bass/jp8000_tribass":
    "Roland JP-8000 triangle bass - smoother and warmer than sawtooth. " +
    "Hollow tone, good for mellow bass lines and deep sub-bass.",

  // === STRINGS ===
  "strings/nfo_chamber_strings_longs":
    "Northern Film Orchestra chamber strings - warm, cinematic ensemble sound. " +
    "Four mic positions (close, stereo, outriggers, room). Perfect for film scoring and layering.",

  "strings/nfo_iso_celli_swells":
    "Northern Film Orchestra cello swells - expressive dynamic swells with 5 layers. " +
    "Great for cinematic builds and emotional passages.",

  "strings/nfo_iso_viola_swells":
    "Northern Film Orchestra viola swells - rich mid-range swells with 5 dynamic layers. " +
    "Perfect for warm, emotional textures.",

  "strings/nfo_iso_violin_swells":
    "Northern Film Orchestra violin swells - soaring high strings with 5 dynamic layers. " +
    "Ideal for cinematic crescendos and lush textures.",

  // === BRASS ===
  "brass/nfo_iso_brass_swells":
    "Northern Film Orchestra brass accents - powerful brass hits with 5 dynamic layers. " +
    "Great for cinematic impact and orchestral power.",

  // === WINDS ===
  "winds/nfo_iso_wind_swells":
    "Northern Film Orchestra wind swells - airy, expressive woodwind textures. " +
    "5 dynamic layers, perfect for cinematic atmospheres.",

  // === SYNTH BASS ===
  "synth/bass/bs_2010_house":
    "House-style synth bass - punchy, electronic character. " +
    "Emulates classic Nord/Juno sounds. Great for house and electronic.",

  "synth/bass/bs_another_analog_bass":
    "Warm analog synth bass - vintage character emulating classic synths. " +
    "Versatile for electronic and pop production.",

  "synth/bass/bs_corg_bass":
    "Korg-style bass - fat analog tone with rich harmonics. " +
    "Classic synth bass sound for electronic music.",

  "synth/bass/bs_deep_undertone":
    "Deep sub-bass synth - heavy low-end for electronic and cinematic. " +
    "Massive sub-bass presence.",

  "synth/bass/bs_lead_bass_player":
    "Melodic synth bass - great for bass leads and prominent bass lines. " +
    "Cuts through the mix well.",

  "synth/bass/bs_outrun_bass":
    "Synthwave/outrun bass - 80s inspired analog sound. " +
    "Perfect for retro electronic and synthwave.",

  "synth/bass/bs_thick_bass":
    "Thick, fat-bodied synth bass - heavy, full sound. " +
    "Great for genres needing powerful low-end.",

  "synth/bass/bs_ms20_bass":
    "MS-20 style bass - aggressive, resonant analog character. " +
    "Great for electronic and experimental music.",

  // === SYNTH KEYS ===
  "synth/keys/kb_dx_epiano":
    "DX-style FM electric piano - metallic, bell-like Yamaha DX7 sound. " +
    "Classic 80s electric piano, great for pop and R&B.",

  "synth/keys/kb_interstellar_on_a_budget":
    "Cinematic synth keys - atmospheric, Hans Zimmer-inspired sound. " +
    "Perfect for film scoring and ambient music.",

  "synth/keys/kb_nord_string":
    "Nord-style string synth - lush, vintage string machine sound. " +
    "Great for pads and sustained chords.",

  "synth/keys/kb_outrun_pluck":
    "Synthwave pluck synth - bright, percussive 80s sound. " +
    "Perfect for arpeggios and retro electronic.",

  "synth/keys/kb_rhode_less_traveled":
    "Rhodes-style electric piano - warm, soulful vintage EP sound. " +
    "Classic for jazz, soul, and neo-soul.",

  "synth/keys/kb_synthetic_organ":
    "Synthetic organ - digital organ sound with vintage character. " +
    "Versatile for various genres.",

  "synth/keys/kb_synthetic_strings":
    "Synthetic string ensemble - classic string machine sound. " +
    "Great for pads and vintage textures.",

  "synth/keys/kb_the_organ_trail":
    "Vintage organ sound - classic tonewheel organ character. " +
    "Perfect for gospel, funk, and rock.",

  // === SYNTH LEADS ===
  "synth/lead/ld_cs80-ish":
    "Yamaha CS-80 style lead - lush, warm analog sound. " +
    "Classic for cinematic and electronic leads.",

  "synth/lead/ld_classic_saws":
    "Classic sawtooth lead - bright, cutting analog sound. " +
    "Versatile synth lead for all electronic genres.",

  "synth/lead/ld_crystaline_80s":
    "Crystalline 80s lead - bright, shimmering vintage sound. " +
    "Perfect for synthwave and retro pop.",

  "synth/lead/ld_fm_hard_lead":
    "FM hard lead - metallic, aggressive DX-style sound. " +
    "Great for cutting through dense mixes.",

  "synth/lead/ld_for_each_loop":
    "Looping synth lead - rhythmic, sequenced character. " +
    "Great for electronic and dance.",

  "synth/lead/ld_forever_80s":
    "Classic 80s lead - vintage synth pop character. " +
    "Perfect for retro productions.",

  "synth/lead/ld_french_house":
    "French house lead - filtered, disco-inspired sound. " +
    "Perfect for house and electronic dance.",

  "synth/lead/ld_imfamousoty":
    "Cinematic lead synth - atmospheric, score-ready sound. " +
    "Great for film and game music.",

  "synth/lead/ld_jp_patchlead":
    "Jupiter-8 style lead - fat, warm analog sound. " +
    "Classic for electronic and synthwave.",

  "synth/lead/ld_juno_was_is":
    "Juno-60 style lead - vintage chorus synth sound. " +
    "Classic warm analog character.",

  "synth/lead/ld_legecy_lead":
    "Legacy synth lead - vintage analog character. " +
    "Versatile for various electronic styles.",

  "synth/lead/ld_poptab":
    "Pop pluck lead - bright, percussive sound. " +
    "Great for melodies and arpeggios.",

  "synth/lead/ld_strangest_things":
    "Stranger Things style lead - dark, 80s horror synth. " +
    "Perfect for cinematic and retro scores.",

  "synth/lead/ld_synthetic_brass":
    "Synthetic brass lead - bold, brass-like synth sound. " +
    "Great for power and impact.",

  "synth/lead/ld_the_nord":
    "Nord Lead style - classic virtual analog sound. " +
    "Versatile for electronic and pop.",

  "synth/lead/ld_the_stack_guitar_chug":
    "Guitar-like synth chug - rhythmic, distorted sound. " +
    "Perfect for electronic rock.",

  "synth/lead/ld_the_stack_guitar":
    "Guitar-like synth lead - emulates guitar character. " +
    "Unique hybrid sound for creative production.",

  "synth/lead/ld_uberheim_legend":
    "Oberheim style lead - fat, multi-voice analog sound. " +
    "Classic vintage synth character.",

  // === SYNTH PADS ===
  "synth/pad/pd_airlock_leak":
    "Sci-fi atmospheric pad - spacey, cinematic texture. " +
    "Perfect for ambient and film scoring.",

  "synth/pad/pd_event_horizon":
    "Dark cinematic pad - ominous, deep atmosphere. " +
    "Great for tension and sci-fi.",

  "synth/pad/pd_every_80s_movie_ever":
    "Classic 80s pad - warm, nostalgic synth texture. " +
    "Perfect for synthwave and retro scores.",

  "synth/pad/pd_fatness_pad":
    "Fat, full-bodied pad - lush, warm synth texture. " +
    "Great for filling out arrangements.",

  "synth/pad/pd_on_the_horizon":
    "Evolving cinematic pad - dynamic, atmospheric texture. " +
    "Perfect for builds and transitions.",

  "synth/pad/pd_orion_belt":
    "Spacey pad - cosmic, ethereal atmosphere. " +
    "Great for ambient and sci-fi.",

  "synth/pad/pd_soft_and_padded":
    "Soft, gentle pad - warm, comforting texture. " +
    "Perfect for ballads and emotional music.",

  "synth/pad/pd_the_first_pad":
    "Classic synth pad - versatile, warm analog texture. " +
    "Great foundation for various genres.",

  "synth/pad/pd_timeless_movement":
    "Evolving pad - shifting, dynamic texture. " +
    "Perfect for cinematic and ambient music.",

  // === SYNTH SEQUENCES ===
  "synth/seq/sq_rhythmic_seq_1_(100bpm)":
    "Rhythmic sequence at 100 BPM - arpeggiated synth pattern. " +
    "Great for electronic and dance music foundations.",

  "synth/seq/sq_rhythmic_seq_2_(100bpm)":
    "Rhythmic sequence at 100 BPM - alternative arpeggiated pattern. " +
    "Perfect for adding movement to electronic tracks.",
};

/**
 * Get description for an instrument, with fallback to name
 */
export function getInstrumentDescription(path: string, fallbackName: string): string {
  return INSTRUMENT_DESCRIPTIONS[path] || fallbackName;
}

/**
 * Format instrument list with descriptions for LLM prompts
 */
export function formatInstrumentsForPrompt(instrumentPaths: string[]): string {
  return instrumentPaths
    .map(path => {
      const description = INSTRUMENT_DESCRIPTIONS[path];
      if (description) {
        return `- ${path}: ${description}`;
      }
      return `- ${path}`;
    })
    .join('\n');
}