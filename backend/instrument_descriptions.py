"""
Instrument descriptions for LLM context
Mirrors frontend/src/data/instrumentDescriptions.ts
"""

from typing import List, Dict

# Instrument descriptions from original sample packs
INSTRUMENT_DESCRIPTIONS: Dict[str, str] = {
    # === DRUMS ===
    "drums/lorenzos_drums":
        "Yamaha drum kit - punchy and snappy with individual mic channels (overhead, snare, kick). "
        "Great for rock, metal, and most genres. Raw, authentic sound with 5 velocity layers. "
        "DSL-supported sounds: kick, snare, hihat_closed, hihat_open, tom, crash, ride, splash. "
        "MISSING: snare_rimshot, snare_buzz, hihat_pedal (use bedroom_drums for complete kit).",

    "drums/bedroom_drums":
        "Sonor Martini Mini Kit - simple, organic drum sound. Punchy kick, snappy snare, bright cymbals. "
        "Not over-produced, great for adding your own character. Complete kit with 3 velocity layers. "
        "DSL-supported sounds: kick, snare, snare_rimshot, snare_buzz, hihat_closed, hihat_open, hihat_pedal, "
        "tom, crash, ride. Complete DSL drum kit - supports ALL 10 standard drum names.",

    # === PIANOS ===
        "piano/steinway_grand":
        "Steinway Grand - tight, warm tone with rich, full-bodied sound. Excellent dynamic range. Classic concert piano suitable for all genres. Range: F#0-F7.",

        "piano/experience_ny_steinway":
        "Steinway Model B - semi-concert grand with impeccable tone. Detailed velocity layers, action noise, and pedal mechanics for realistic performance. Professional studio sound. Range: C0-C7.",

        "piano/fender_rhodes":
        "Fender Rhodes Mark I - classic electric piano with bell-like, mellow tone. Perfect for jazz, soul, R&B, and vintage pop. Clean, unprocessed sound. Range: E1-E7.",

        "piano/bechstein_1911_upright":
        "Bechstein 1911 Upright - warm, characterful vintage upright with over a century of resonance. Historic instrument with intimate, close sound. Great for singer-songwriter and classical. Range: A0-A7.",

    # === HARPSICHORDS ===
    "harpsichord/harpsichord_english":
        "English harpsichord - bright, crisp plucked sound. "
        "Perfect for baroque music and arpeggiated patterns. Range: A#1-F6.",

    "harpsichord/harpsichord_flemish":
        "Flemish harpsichord - fuller, rounder tone than English style. "
        "Rich harmonic content, great for counterpoint. Range: F1-C6.",

    "harpsichord/harpsichord_french":
        "French harpsichord - elegant, delicate plucked sound. "
        "Classic baroque timbre. Range: C1-C6.",

    "harpsichord/harpsichord_italian":
        "Italian harpsichord - light, transparent sound. "
        "Perfect for early music and delicate passages. Range: F1-C6.",

    # === GUITARS ===
    "guitar/rjs_guitar_new_strings":
        "Sampled Stratocaster with new strings - bright, clear tone. "
        "Multiple articulations (vibrato, palm mute, dead notes). Great for leads and rhythm. Range: E2-E5.",

    "guitar/rjs_guitar_old_strings":
        "Sampled Stratocaster with old strings - warm, mellower tone than new strings. "
        "Multiple articulations. Works great in the mix. Range: E2-E5.",

    "guitar/rjs_guitar_palm_muted_softly_strings":
        "Sampled Stratocaster palm muted (soft) - tight, percussive rhythm sound. "
        "Perfect for clean rhythm parts and funk. Range: E2-E5.",

    "guitar/rjs_guitar_palm_muted_strings":
        "Sampled Stratocaster palm muted (hard) - tight, aggressive rhythm sound. "
        "Great for rock and metal rhythm parts. Range: E2-E5.",

    # === BASS ===
    "bass/jp8000_sawbass":
        "Roland JP-8000 sawtooth bass - bright, cutting analog sound with rich harmonics. "
        "Classic analog synth bass, great for electronic and dance music. Range: G2-G5.",

    "bass/jp8000_tribass":
        "Roland JP-8000 triangle bass - smoother and warmer than sawtooth. "
        "Hollow tone, good for mellow bass lines and deep sub-bass. Range: G2-G5.",

    "bass/funky_fingers":
        "Funky Fingers bass - warm fingerstyle electric bass with natural character. 3 velocity layers "
        "capture expressive dynamics. Great for funk, soul, R&B, and pop. Wide range B1-G#5.",

    "bass/low_fat_bass":
        "Low Fat Bass - clean fingerstyle electric bass. Long sustaining notes with even tone. "
        "Perfect for pop, indie, and contemporary styles. Range: E2-G4.",

    # === STRINGS ===
    "strings/nfo_chamber_strings_longs":
        "Northern Film Orchestra chamber strings - warm, cinematic ensemble sound. "
        "Four mic positions (close, stereo, outriggers, room). Perfect for film scoring and layering. Range: C1-F7.",

    "strings/nfo_iso_celli_swells":
        "Northern Film Orchestra cello swells - expressive dynamic swells with 5 layers. "
        "Great for cinematic builds and emotional passages. Range: C1-C5.",

    "strings/nfo_iso_viola_swells":
        "Northern Film Orchestra viola swells - rich mid-range swells with 5 dynamic layers. "
        "Perfect for warm, emotional textures. Range: A2-C6.",

    "strings/nfo_iso_violin_swells":
        "Northern Film Orchestra violin swells - soaring high strings with 5 dynamic layers. "
        "Ideal for cinematic crescendos and lush textures. Range: E3-G6.",

    # === BRASS ===
    "brass/nfo_iso_brass_swells":
        "Northern Film Orchestra brass accents - powerful brass hits with 5 dynamic layers. "
        "Great for cinematic impact and orchestral power. Range: C1-F#5.",

    # === WINDS ===
    "winds/flute_violin":
        "Flute + Violin hybrid - student violin layered with Native American flute, pitched down in lower registers. "
        "Ethereal sustain drone sound with dry and echo/reverb versions. Perfect for atmospheric and cinematic music. Range: D4-D5.",

    "winds/subtle_clarinet":
        "Subtle Clarinet - soft, gentle clarinet with ebb-and-flow dynamics and natural woody tone. "
        "Recorded in living room for intimate character. Great for jazz, classical, and delicate arrangements. Range: D#2-D#5.",

    "winds/decent_oboe":
        "Yamaha Oboe - expressive sustains, staccatos, and legato articulations with 2 round robins. "
        "Features vibrato, delay, and convolution reverb controls. Perfect for orchestral and solo passages. Range: C5-C#7.",

    "winds/tenor_saxophone":
        "Tenor Saxophone - multi-mic recording (close, mid-room, far-room) with 3 velocity layers and 3 round robins per layer. "
        "Includes key/pad sounds. Warm, rich tone for jazz, blues, soul, and R&B. Range: A3-A5.",

    # === SYNTH BASS ===
    "synth/bass/bs_2010_house":
        "House-style synth bass - punchy, electronic character. "
        "Emulates classic Nord/Juno sounds. Great for house and electronic. Range: C2-C6.",

    "synth/bass/bs_another_analog_bass":
        "Warm analog synth bass - vintage character emulating classic synths. "
        "Versatile for electronic and pop production. Range: C2-C6.",

    "synth/bass/bs_corg_bass":
        "Korg-style bass - fat analog tone with rich harmonics. "
        "Classic synth bass sound for electronic music. Range: C2-C6.",

    "synth/bass/bs_deep_undertone":
        "Deep sub-bass synth - heavy low-end for electronic and cinematic. "
        "Massive sub-bass presence. Range: C2-C6.",

    "synth/bass/bs_lead_bass_player":
        "Melodic synth bass - great for bass leads and prominent bass lines. "
        "Cuts through the mix well. Range: C2-C6.",

    "synth/bass/bs_outrun_bass":
        "Synthwave/outrun bass - 80s inspired analog sound. "
        "Perfect for retro electronic and synthwave. Range: C2-C6.",

    "synth/bass/bs_thick_bass":
        "Thick, fat-bodied synth bass - heavy, full sound. "
        "Great for genres needing powerful low-end. Range: C2-C6.",

    "synth/bass/bs_ms20_bass":
        "MS-20 style bass - aggressive, resonant analog character. "
        "Great for electronic and experimental music. Range: C2-C6.",

    # === SYNTH KEYS ===
    "synth/keys/kb_dx_epiano":
        "DX-style FM electric piano - metallic, bell-like Yamaha DX7 sound. "
        "Classic 80s electric piano, great for pop and R&B. Range: C2-C6.",

    "synth/keys/kb_interstellar_on_a_budget":
        "Cinematic synth keys - atmospheric, Hans Zimmer-inspired sound. "
        "Perfect for film scoring and ambient music. Range: C2-C6.",

    "synth/keys/kb_nord_string":
        "Nord-style string synth - lush, vintage string machine sound. "
        "Great for pads and sustained chords. Range: C2-C6.",

    "synth/keys/kb_outrun_pluck":
        "Synthwave pluck synth - bright, percussive 80s sound. "
        "Perfect for arpeggios and retro electronic. Range: C2-C6.",

    "synth/keys/kb_rhode_less_traveled":
        "Rhodes-style electric piano - warm, soulful vintage EP sound. "
        "Classic for jazz, soul, and neo-soul. Range: C2-C6.",

    "synth/keys/kb_synthetic_organ":
        "Synthetic organ - digital organ sound with vintage character. "
        "Versatile for various genres. Range: C2-C6.",

    "synth/keys/kb_synthetic_strings":
        "Synthetic string ensemble - classic string machine sound. "
        "Great for pads and vintage textures. Range: C2-C6.",

    "synth/keys/kb_the_organ_trail":
        "Vintage organ sound - classic tonewheel organ character. "
        "Perfect for gospel, funk, and rock. Range: C2-C6.",

    # === SYNTH LEADS ===
    "synth/lead/ld_cs80-ish":
        "Yamaha CS-80 style lead - lush, warm analog sound. "
        "Classic for cinematic and electronic leads. Range: C2-C6.",

    "synth/lead/ld_classic_saws":
        "Classic sawtooth lead - bright, cutting analog sound. "
        "Versatile synth lead for all electronic genres. Range: C2-C6.",

    "synth/lead/ld_crystaline_80s":
        "Crystalline 80s lead - bright, shimmering vintage sound. "
        "Perfect for synthwave and retro pop. Range: C2-C6.",

    "synth/lead/ld_fm_hard_lead":
        "FM hard lead - metallic, aggressive DX-style sound. "
        "Great for cutting through dense mixes. Range: C2-C6.",

    "synth/lead/ld_for_each_loop":
        "Looping synth lead - rhythmic, sequenced character. "
        "Great for electronic and dance. Range: C2-C6.",

    "synth/lead/ld_forever_80s":
        "Classic 80s lead - vintage synth pop character. "
        "Perfect for retro productions. Range: C2-C6.",

    "synth/lead/ld_french_house":
        "French house lead - filtered, disco-inspired sound. "
        "Perfect for house and electronic dance. Range: C2-C6.",

    "synth/lead/ld_imfamousoty":
        "Cinematic lead synth - atmospheric, score-ready sound. "
        "Great for film and game music. Range: C2-C6.",

    "synth/lead/ld_jp_patchlead":
        "Jupiter-8 style lead - fat, warm analog sound. "
        "Classic for electronic and synthwave. Range: C2-C6.",

    "synth/lead/ld_juno_was_is":
        "Juno-60 style lead - vintage chorus synth sound. "
        "Classic warm analog character. Range: C2-C6.",

    "synth/lead/ld_legecy_lead":
        "Legacy synth lead - vintage analog character. "
        "Versatile for various electronic styles. Range: C2-C6.",

    "synth/lead/ld_poptab":
        "Pop pluck lead - bright, percussive sound. "
        "Great for melodies and arpeggios. Range: C2-C6.",

    "synth/lead/ld_strangest_things":
        "Stranger Things style lead - dark, 80s horror synth. "
        "Perfect for cinematic and retro scores. Range: C2-C6.",

    "synth/lead/ld_synthetic_brass":
        "Synthetic brass lead - bold, brass-like synth sound. "
        "Great for power and impact. Range: C2-C6.",

    "synth/lead/ld_the_nord":
        "Nord Lead style - classic virtual analog sound. "
        "Versatile for electronic and pop. Range: C2-C6.",

    "synth/lead/ld_the_stack_guitar_chug":
        "Guitar-like synth chug - rhythmic, distorted sound. "
        "Perfect for electronic rock. Range: C2-C6.",

    "synth/lead/ld_the_stack_guitar":
        "Guitar-like synth lead - emulates guitar character. "
        "Unique hybrid sound for creative production. Range: C2-C6.",

    "synth/lead/ld_uberheim_legend":
        "Oberheim style lead - fat, multi-voice analog sound. "
        "Classic vintage synth character. Range: C2-C6.",

    # === SYNTH PADS ===
    "synth/pad/pd_airlock_leak":
        "Sci-fi atmospheric pad - spacey, cinematic texture. "
        "Perfect for ambient and film scoring. Range: C2-C6.",

    "synth/pad/pd_event_horizon":
        "Dark cinematic pad - ominous, deep atmosphere. "
        "Great for tension and sci-fi. Range: C2-C6.",

    "synth/pad/pd_every_80s_movie_ever":
        "Classic 80s pad - warm, nostalgic synth texture. "
        "Perfect for synthwave and retro scores. Range: C2-C6.",

    "synth/pad/pd_fatness_pad":
        "Fat, full-bodied pad - lush, warm synth texture. "
        "Great for filling out arrangements. Range: C2-C6.",

    "synth/pad/pd_on_the_horizon":
        "Evolving cinematic pad - dynamic, atmospheric texture. "
        "Perfect for builds and transitions. Range: C2-C6.",

    "synth/pad/pd_orion_belt":
        "Spacey pad - cosmic, ethereal atmosphere. "
        "Great for ambient and sci-fi. Range: C2-C6.",

    "synth/pad/pd_soft_and_padded":
        "Soft, gentle pad - warm, comforting texture. "
        "Perfect for ballads and emotional music. Range: C2-C6.",

    "synth/pad/pd_the_first_pad":
        "Classic synth pad - versatile, warm analog texture. "
        "Great foundation for various genres. Range: C2-C6.",

    "synth/pad/pd_timeless_movement":
        "Evolving pad - shifting, dynamic texture. "
        "Perfect for cinematic and ambient music. Range: C2-C6.",

    # === SYNTH SEQUENCES ===
    "synth/seq/sq_rhythmic_seq_1_(100bpm)":
        "Rhythmic sequence at 100 BPM - arpeggiated synth pattern. "
        "Great for electronic and dance music foundations. Range: C2-C6.",

    "synth/seq/sq_rhythmic_seq_2_(100bpm)":
        "Rhythmic sequence at 100 BPM - alternative arpeggiated pattern. "
        "Perfect for adding movement to electronic tracks. Range: C2-C6.",
}

# Track type to instrument mapping
# Mirrors frontend/src/lib/instrumentCatalog.ts
TRACK_TYPE_INSTRUMENTS: Dict[str, List[str]] = {
    "bass": [
        "bass/jp8000_sawbass",
        "bass/jp8000_tribass",
        "bass/funky_fingers",
        "bass/low_fat_bass",
        "synth/bass/bs_2010_house",
        "synth/bass/bs_another_analog_bass",
        "synth/bass/bs_corg_bass",
        "synth/bass/bs_deep_undertone",
        "synth/bass/bs_lead_bass_player",
        "synth/bass/bs_outrun_bass",
        "synth/bass/bs_thick_bass",
        "synth/bass/bs_ms20_bass"
    ],
    "chords": [
        "piano/steinway_grand",
        "piano/experience_ny_steinway",
        "piano/fender_rhodes",
        "piano/bechstein_1911_upright",
        "guitar/rjs_guitar_new_strings",
        "guitar/rjs_guitar_old_strings",
        "guitar/rjs_guitar_palm_muted_softly_strings",
        "guitar/rjs_guitar_palm_muted_strings",
        "synth/keys/kb_dx_epiano",
        "synth/keys/kb_interstellar_on_a_budget",
        "synth/keys/kb_nord_string",
        "synth/keys/kb_rhode_less_traveled",
        "synth/keys/kb_synthetic_organ",
        "synth/keys/kb_synthetic_strings",
        "synth/keys/kb_the_organ_trail",
        "harpsichord/harpsichord_english",
        "harpsichord/harpsichord_flemish",
        "harpsichord/harpsichord_french",
        "harpsichord/harpsichord_italian"
    ],
    "pad": [
        "synth/pad/pd_airlock_leak",
        "synth/pad/pd_event_horizon",
        "synth/pad/pd_every_80s_movie_ever",
        "synth/pad/pd_fatness_pad",
        "synth/pad/pd_on_the_horizon",
        "synth/pad/pd_orion_belt",
        "synth/pad/pd_soft_and_padded",
        "synth/pad/pd_the_first_pad",
        "synth/pad/pd_timeless_movement",
        "strings/nfo_chamber_strings_longs",
        "strings/nfo_iso_celli_swells",
        "strings/nfo_iso_viola_swells",
        "strings/nfo_iso_violin_swells",
        "synth/keys/kb_nord_string",
        "synth/keys/kb_synthetic_strings"
    ],
    "melody": [
        "piano/steinway_grand",
        "piano/experience_ny_steinway",
        "piano/fender_rhodes",
        "piano/bechstein_1911_upright",
        "guitar/rjs_guitar_new_strings",
        "guitar/rjs_guitar_old_strings",
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
        "synth/keys/kb_dx_epiano",
        "synth/keys/kb_outrun_pluck",
        "brass/nfo_iso_brass_swells",
        "winds/flute_violin",
        "winds/subtle_clarinet",
        "winds/decent_oboe",
        "winds/tenor_saxophone"
    ],
    "counterMelody": [
        "strings/nfo_chamber_strings_longs",
        "strings/nfo_iso_celli_swells",
        "strings/nfo_iso_viola_swells",
        "strings/nfo_iso_violin_swells",
        "synth/lead/ld_cs80-ish",
        "synth/lead/ld_classic_saws",
        "synth/lead/ld_crystaline_80s",
        "synth/lead/ld_poptab",
        "synth/lead/ld_strangest_things",
        "synth/lead/ld_synthetic_brass",
        "brass/nfo_iso_brass_swells",
        "synth/keys/kb_dx_epiano",
        "synth/keys/kb_outrun_pluck"
    ],
    "arpeggio": [
        "harpsichord/harpsichord_english",
        "harpsichord/harpsichord_flemish",
        "harpsichord/harpsichord_french",
        "harpsichord/harpsichord_italian",
        "piano/steinway_grand",
        "piano/experience_ny_steinway",
        "synth/keys/kb_outrun_pluck",
        "synth/lead/ld_poptab",
        "synth/seq/sq_rhythmic_seq_1_(100bpm)",
        "synth/seq/sq_rhythmic_seq_2_(100bpm)"
    ],
    "drums": [
        "drums/bedroom_drums",
        "drums/lorenzos_drums"
    ]
}


def get_instrument_description(path: str, fallback_name: str = "") -> str:
    """Get description for an instrument, with fallback to name."""
    return INSTRUMENT_DESCRIPTIONS.get(path, fallback_name)


def format_instruments_for_prompt(instrument_paths: List[str]) -> str:
    """
    Format instrument list with descriptions for LLM prompts.

    Args:
        instrument_paths: List of instrument paths

    Returns:
        Formatted string with instrument descriptions
    """
    lines = []
    for path in instrument_paths:
        description = INSTRUMENT_DESCRIPTIONS.get(path)
        if description:
            lines.append(f"- {path}: {description}")
        else:
            lines.append(f"- {path}")

    return "\n".join(lines)


def get_instruments_for_track_type(track_type: str) -> List[str]:
    """
    Get instrument paths for a specific track type.

    Args:
        track_type: Track type (bass, chords, pad, etc.)

    Returns:
        List of instrument paths
    """
    return TRACK_TYPE_INSTRUMENTS.get(track_type, [])