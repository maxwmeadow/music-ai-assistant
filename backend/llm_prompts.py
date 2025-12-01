"""
LLM Prompt templates for music arrangement generation
These prompts will be cached to reduce token costs
"""

# This prompt will be cached using Claude's prompt caching featureThere

DSL_DOCUMENTATION = """You are a professional music arranger generating complementary tracks in our custom DSL format.

<role>
Your task: Generate a SINGLE track that complements existing music.
Output: Pure DSL code only - no explanations, no markdown.
</role>

<dsl_syntax>
## Track Definition
track("name") { ... }
- Defines a musical track
- Name should be descriptive: "bass", "chords", "melody", "pad", etc.

## Instrument Selection
instrument("category/name")
- MUST be first command inside track
- Example: instrument("piano/steinway_grand")
- Use ONLY instruments from the provided catalog

## Notes
note("pitch", start, duration, velocity)
- pitch: Note name + octave OR drum name
  - Regular notes: "C4", "F#3", "Bb5" (Middle C = C4)
  - Drum names: "kick", "snare", "hihat_closed", "hihat_open", "crash", "ride"
  - Use either sharps (C#, D#, F#, G#, A#) or flats (Db, Eb, Gb, Ab, Bb)
- start: Time in seconds (float) → 0.0, 1.5, 2.25
- duration: Length in seconds (float) → 0.5, 1.0, 2.0
- velocity: Volume 0.0-1.0 → 0.0=silent, 1.0=loudest

Examples:
note("C4", 0.0, 1.0, 0.8)      // Middle C at 0s, 1s duration, loud
note("F#3", 2.5, 0.25, 0.6)    // F# below middle C, short note
note("kick", 0.0, 0.3, 0.9)    // Kick drum (for drum tracks)
note("snare", 1.0, 0.3, 0.8)   // Snare drum (for drum tracks)

## Chords
chord(["note1", "note2", ...], start, duration, velocity)
- Array of 2-6 notes played simultaneously
- Same timing/velocity rules as notes

Examples:
chord(["C4", "E4", "G4"], 0.0, 2.0, 0.7)           // C major triad
chord(["F3", "A3", "C4", "E4"], 2.0, 1.0, 0.6)     // F major 7th

## Loops
loop(start_time, end_time) { ... }
- Repeats content between absolute times
- Content inside uses relative timing (starts at 0.0)
- Duration determines how many times pattern repeats

Example:
loop(8.0, 16.0) {
    // This 4s pattern repeats from 8-16s (2 times)
    note("C2", 0.0, 1.0, 0.7)
    note("G2", 1.0, 1.0, 0.7)
    note("F2", 2.0, 1.0, 0.7)
    note("C2", 3.0, 1.0, 0.7)
}

## Comments
// Comment text
- Use for section labels: // Intro, // Verse, // Chorus
</dsl_syntax>

<examples>
## Example 1: Simple Track
track("pad") {
    instrument("synth/pad/pd_soft_and_padded")

    // Long sustained chords
    chord(["C3", "E3", "G3"], 0.0, 4.0, 0.5)
    chord(["F3", "A3", "C4"], 4.0, 4.0, 0.5)
    chord(["G3", "B3", "D4"], 8.0, 4.0, 0.5)
}

## Example 2: Using Loops
track("bass") {
    instrument("bass/jp8000_sawbass")

    // Intro (0-8s)
    note("C2", 0.0, 2.0, 0.7)
    note("F2", 2.0, 2.0, 0.7)
    note("G2", 4.0, 2.0, 0.7)
    note("C2", 6.0, 2.0, 0.7)

    // Repeating section (8-16s)
    loop(8.0, 16.0) {
        note("C2", 0.0, 1.0, 0.8)
        note("E2", 1.0, 1.0, 0.7)
        note("G2", 2.0, 1.0, 0.8)
        note("C2", 3.0, 1.0, 0.7)
    }
}
</examples>

<output_rules>
CRITICAL: Your response must be PURE DSL CODE ONLY.

REQUIRED:
- Start immediately with: track("name") {
- End with: }
- First character of response: 't'
- Last character of response: '}'
- Include instrument() as first command
- Use only valid DSL syntax
- Use only instruments from provided catalog
- Match the requested musical style

FORBIDDEN:
- NO markdown code fences (no ``` anywhere)
- NO explanatory text before/after code
- NO introductions ("Here's a..." / "This creates...")
- NO tempo() command (already set)
- NO multiple tracks (only ONE)
- NO velocity outside 0.0-1.0 range
- NO negative start times
</output_rules>"""


def get_style_guidance(track_type: str, genre: str) -> str:
    """
    Generate dynamic style guidance and examples based on track type and genre.

    Args:
        track_type: Type of track (bass, chords, pad, melody, counterMelody, arpeggio, drums)
        genre: Musical genre/style

    Returns:
        Style-specific guidance string
    """
    # Normalize inputs
    track_type = track_type.lower()
    genre = genre.lower()

    # Style guidance templates
    guidance = []

    # === BASS GUIDANCE ===
    if track_type == "bass":
        guidance.append("<bass_guidance>")
        guidance.append("Register: C1-C3 (lower notes for foundation)")

        if "pop" in genre or "rock" in genre:
            guidance.append("Style: Simple root notes on beat 1, whole/half notes, steady pulse")
            guidance.append("Example:")
            guidance.append('note("C2", 0.0, 2.0, 0.7)  // Root on downbeat')
            guidance.append('note("F2", 2.0, 2.0, 0.7)  // Sustained notes')

        elif "jazz" in genre:
            guidance.append("Style: Walking bass - quarter notes, chromatic approach tones, chord tones")
            guidance.append("Example:")
            guidance.append('note("C2", 0.0, 0.5, 0.8)  // Root')
            guidance.append('note("D2", 0.5, 0.5, 0.7)  // Passing tone')
            guidance.append('note("E2", 1.0, 0.5, 0.8)  // Approach')
            guidance.append('note("F2", 1.5, 0.5, 0.8)  // Target')

        elif "electronic" in genre or "edm" in genre or "house" in genre:
            guidance.append("Style: Syncopated, 16th notes, rhythmic accents, use rests")
            guidance.append("Example:")
            guidance.append('note("C2", 0.0, 0.25, 1.0)   // Strong accent')
            guidance.append('note("C2", 0.25, 0.25, 0.8)  // Light')
            guidance.append('note("E2", 0.5, 0.5, 0.9)    // Syncopation')

        else:
            guidance.append("Style: Root notes on strong beats, follow chord progression")
            guidance.append("Typical range: C2, D2, E2, F2, G2, A2, B2")

        guidance.append("</bass_guidance>")

    # === CHORDS GUIDANCE ===
    elif track_type == "chords":
        guidance.append("<chords_guidance>")
        guidance.append("Register: C3-C5 (mid-range, leave room for bass and melody)")

        if "pop" in genre or "rock" in genre:
            guidance.append("Style: Block chords (triads), sustained, clear rhythm")
            guidance.append("Example:")
            guidance.append('chord(["C4", "E4", "G4"], 0.0, 2.0, 0.7)     // C major triad')
            guidance.append('chord(["F4", "A4", "C5"], 2.0, 2.0, 0.7)     // F major')

        elif "jazz" in genre:
            guidance.append("Style: 7th chords, extensions (9th, 11th), voicings with space")
            guidance.append("Example:")
            guidance.append('chord(["C4", "E4", "G4", "B4"], 0.0, 1.0, 0.7)     // Cmaj7')
            guidance.append('chord(["D4", "F4", "A4", "C5"], 1.0, 1.0, 0.7)     // Dm7')

        elif "electronic" in genre:
            guidance.append("Style: Triads or power chords, sustained or rhythmic stabs")
            guidance.append("Velocity: 0.6-0.9 for punch")

        else:
            guidance.append("Style: Triads or 7th chords, align with harmonic progression")
            guidance.append("Common voicings: Root position, 1st inversion for smooth voice leading")

        guidance.append("</chords_guidance>")

    # === PAD GUIDANCE ===
    elif track_type == "pad":
        guidance.append("<pad_guidance>")
        guidance.append("Register: C3-C5 (mid-range)")
        guidance.append("Style: Long sustained chords, low velocity (0.4-0.6), create atmosphere")
        guidance.append("Duration: 4-8 seconds per chord, slow harmonic rhythm")
        guidance.append("Example:")
        guidance.append('chord(["C3", "E3", "G3"], 0.0, 8.0, 0.5)   // Soft, sustained')
        guidance.append('chord(["F3", "A3", "C4"], 8.0, 8.0, 0.5)   // Gentle transition')
        guidance.append("</pad_guidance>")

    # === MELODY GUIDANCE ===
    elif track_type == "melody":
        guidance.append("<melody_guidance>")
        guidance.append("Register: C4-C6 (higher range, above chords)")

        if "pop" in genre:
            guidance.append("Style: Singable phrases, step-wise motion, memorable hooks")
            guidance.append("Rhythm: Mix of quarter and eighth notes, clear phrases")

        elif "jazz" in genre:
            guidance.append("Style: Chromatic approach, bebop scales, swing rhythm")
            guidance.append("Use chord extensions and alterations")

        elif "electronic" in genre:
            guidance.append("Style: Repetitive motifs, arpeggiated patterns, rhythmic interest")

        else:
            guidance.append("Style: Lyrical phrases, follow harmonic progression, clear contour")

        guidance.append("Phrasing: Use rests between phrases for breathing")
        guidance.append("</melody_guidance>")

    # === COUNTER MELODY GUIDANCE ===
    elif track_type == "countermelody":
        guidance.append("<countermelody_guidance>")
        guidance.append("Register: C4-C5 (between melody and chords)")
        guidance.append("Style: Answer or complement main melody, different rhythm")
        guidance.append("Use rests when main melody is active, fill spaces")
        guidance.append("Typically lower velocity (0.5-0.7) to stay in background")
        guidance.append("</countermelody_guidance>")

    # === ARPEGGIO GUIDANCE ===
    elif track_type == "arpeggio":
        guidance.append("<arpeggio_guidance>")
        guidance.append("Register: C4-C6")
        guidance.append("Style: Sequential chord notes, fast rhythm (8th or 16th notes)")
        guidance.append("Pattern: Up, down, up-down, or melodic shapes")
        guidance.append("Example:")
        guidance.append('note("C4", 0.0, 0.25, 0.8)   // Root')
        guidance.append('note("E4", 0.25, 0.25, 0.8)  // Third')
        guidance.append('note("G4", 0.5, 0.25, 0.8)   // Fifth')
        guidance.append('note("C5", 0.75, 0.25, 0.9)  // Octave')
        guidance.append("</arpeggio_guidance>")

    # === DRUMS GUIDANCE ===
    elif track_type == "drums":
        guidance.append("<drums_guidance>")
        guidance.append("CRITICAL: Use drum names in note() calls, NOT MIDI notes!")
        guidance.append("")
        guidance.append("Available drum types:")
        guidance.append('- note("kick", start, duration, velocity)       // Bass drum')
        guidance.append('- note("snare", start, duration, velocity)      // Snare drum')
        guidance.append('- note("snare_rimshot", start, duration, velocity) // Rimshot')
        guidance.append('- note("snare_buzz", start, duration, velocity)     // Buzz roll')
        guidance.append('- note("hihat_closed", start, duration, velocity)   // Closed hi-hat')
        guidance.append('- note("hihat_open", start, duration, velocity)     // Open hi-hat')
        guidance.append('- note("hihat_pedal", start, duration, velocity)    // Pedal hi-hat')
        guidance.append('- note("tom", start, duration, velocity)        // Tom')
        guidance.append('- note("crash", start, duration, velocity)      // Crash cymbal')
        guidance.append('- note("ride", start, duration, velocity)       // Ride cymbal')
        guidance.append("")
        guidance.append("Duration guidelines:")
        guidance.append("- kick: 0.3-0.5s (short and punchy)")
        guidance.append("- snare: 0.2-0.4s")
        guidance.append("- hihat_closed: 0.1-0.2s (very short)")
        guidance.append("- hihat_open: 0.5-1.0s (let it ring)")
        guidance.append("- crash: 1.0-2.0s (sustained)")
        guidance.append("- ride: 0.3-0.6s")
        guidance.append("")
        guidance.append("Velocity tips:")
        guidance.append("- Accented beats: 0.9-1.0")
        guidance.append("- Normal hits: 0.7-0.8")
        guidance.append("- Ghost notes: 0.3-0.5")
        guidance.append("")

        if "pop" in genre or "rock" in genre:
            guidance.append("Style: Kick on 1 and 3, snare on 2 and 4, hi-hats on 8ths")
            guidance.append("Example:")
            guidance.append('loop(0.0, 8.0) {')
            guidance.append('    note("kick", 0.0, 0.3, 0.9)')
            guidance.append('    note("hihat_closed", 0.0, 0.1, 0.6)')
            guidance.append('    note("hihat_closed", 0.5, 0.1, 0.6)')
            guidance.append('    note("snare", 1.0, 0.3, 0.8)')
            guidance.append('    note("hihat_closed", 1.0, 0.1, 0.6)')
            guidance.append('    note("hihat_closed", 1.5, 0.1, 0.6)')
            guidance.append('}')
        elif "electronic" in genre:
            guidance.append("Style: Four-on-floor kick (every beat), hi-hats on off-beats")
            guidance.append("Example:")
            guidance.append('loop(0.0, 8.0) {')
            guidance.append('    note("kick", 0.0, 0.3, 1.0)  // Every beat')
            guidance.append('    note("kick", 0.5, 0.3, 1.0)')
            guidance.append('    note("kick", 1.0, 0.3, 1.0)')
            guidance.append('    note("kick", 1.5, 0.3, 1.0)')
            guidance.append('    note("hihat_closed", 0.25, 0.1, 0.7)  // Off-beats')
            guidance.append('    note("hihat_closed", 0.75, 0.1, 0.7)')
            guidance.append('    note("snare", 1.0, 0.3, 0.9)  // On 2 and 4')
            guidance.append('}')
        elif "jazz" in genre:
            guidance.append("Style: Swing hi-hats/ride, sparse kick/snare, syncopated rhythms")
            guidance.append("Use ride cymbal for swing patterns")

        guidance.append("</drums_guidance>")

    return "\n".join(guidance) if guidance else ""


def build_arrangement_prompt(
    current_dsl: str,
    track_type: str,
    genre: str,
    available_instruments: str,
    tempo: int,
    user_request: str = None,
    music_context: dict = None,
    complexity: str = "medium"
) -> str:
    """
    Build the complete prompt for arrangement generation.

    Args:
        current_dsl: The existing DSL code
        track_type: Type of track to generate (bass, chords, pad, etc.)
        genre: Musical genre/style
        available_instruments: Formatted list of instruments with descriptions
        tempo: Current tempo in BPM
        user_request: Optional custom user request
        music_context: Optional dict with key, chords, duration, etc.
        complexity: Complexity level - "simple", "medium", or "complex"

    Returns:
        Complete prompt string for the user message
    """

    # Build context section
    context_lines = ["## Musical Context", f"- Tempo: {tempo} BPM"]

    if music_context:
        if 'key' in music_context:
            context_lines.append(f"- Key: {music_context['key']}")
        if 'chords' in music_context and music_context['chords']:
            context_lines.append(f"- Chord progression: {' → '.join(music_context['chords'])}")
        if 'duration' in music_context:
            context_lines.append(f"- Current duration: {music_context['duration']:.1f} seconds")
        if 'density' in music_context:
            context_lines.append(f"- Existing density: {music_context['density']} ({music_context.get('notes_per_second', 0):.1f} notes/sec)")
        if 'register' in music_context:
            context_lines.append(f"- Existing register: {music_context['register']}")
        if 'voice_leading_suggestion' in music_context:
            context_lines.append(f"- Voice leading: {music_context['voice_leading_suggestion']}")

    context_lines.append("")
    context_str = "\n".join(context_lines)

    # Get style-specific guidance
    style_guidance = get_style_guidance(track_type, genre)

    # Build complexity guidance
    complexity_guidance = {
        "simple": "- Complexity: SIMPLE - Use basic rhythms, few notes, clear patterns, minimal variation",
        "medium": "- Complexity: MODERATE - Balance simplicity with some variation, standard rhythmic patterns",
        "complex": "- Complexity: INTRICATE - Rich harmonies, varied rhythms, detailed patterns, use extensions and alterations"
    }
    complexity_line = complexity_guidance.get(complexity, complexity_guidance["medium"])

    # Build the full user message
    user_message = f"""## Task

Generate a **{track_type}** track in **{genre}** style to complement the existing music.

{context_str}## Available Instruments for {track_type}

{available_instruments}

## Style Guidance

{style_guidance}

## Current Music (DSL)

```
{current_dsl}
```

## Requirements

- Track type: {track_type}
- Genre/Style: {genre}
{complexity_line}
{f"- Additional instructions: {user_request}" if user_request else ""}
- Must complement existing tracks harmonically and rhythmically
- Follow {genre} style conventions
- Choose ONE instrument from the available list above

## Output

Generate ONLY the DSL code for the new track. Start with `track("{track_type}") {{`. No explanations, no markdown fences."""

    return user_message
