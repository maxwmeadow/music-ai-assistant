from .schemas import IR

"""
FILE compiler_stub.py
Temporary file to provide some backend testing without
the actual compiler
"""


def json_ir_to_dsl(ir: IR) -> str:
    tempo = ir.metadata.get("tempo", 120)
    lines = [f"tempo({tempo})", ""]

    for track in ir.tracks:
        lines.append(f'track("{track.id}") {{')

        if track.instrument:
            lines.append(f'  instrument("{track.instrument}")')

        if track.notes:
            for note in track.notes:
                note_name = midi_to_note(note.pitch)
                lines.append(f'  note("{note_name}", {note.start}, {note.duration}, {note.velocity})')

        if track.samples:
            for sample in track.samples:
                lines.append(f'  {sample.sample}({sample.start})')

        lines.append("}\n")

    return "\n".join(lines)

def compile_scale_to_dsl() -> str:
    """Demo/test stub DSL - Complex multi-track song with loops."""
    return """tempo(120)

track("chords") {
    instrument("piano/grand_piano_k")

    // Intro (0-8s)
    chord(["C4", "E4", "G4"], 0.0, 2.0, 0.6)
    chord(["F4", "A4", "C5"], 2.0, 2.0, 0.6)
    chord(["G4", "B4", "D5"], 4.0, 2.0, 0.6)
    chord(["C4", "E4", "G4"], 6.0, 2.0, 0.6)

    // Verse chord progression (8-24s)
    loop(8.0, 24.0) {
        chord(["C4", "E4", "G4"], 0.0, 2.0, 0.7)
        chord(["A3", "C4", "E4"], 2.0, 2.0, 0.7)
        chord(["F3", "A3", "C4"], 4.0, 2.0, 0.7)
        chord(["G3", "B3", "D4"], 6.0, 2.0, 0.7)
    }

    // Chorus (24-40s)
    loop(24.0, 40.0) {
        chord(["C4", "E4", "G4", "B4"], 0.0, 1.0, 0.9)
        chord(["F4", "A4", "C5"], 1.0, 1.0, 0.9)
        chord(["G4", "B4", "D5"], 2.0, 1.0, 0.9)
        chord(["E4", "G4", "B4"], 3.0, 1.0, 0.9)
    }

    // Verse 2 (40-56s)
    loop(40.0, 56.0) {
        chord(["C4", "E4", "G4"], 0.0, 2.0, 0.7)
        chord(["A3", "C4", "E4"], 2.0, 2.0, 0.7)
        chord(["F3", "A3", "C4"], 4.0, 2.0, 0.7)
        chord(["G3", "B3", "D4"], 6.0, 2.0, 0.7)
    }

    // Outro (56-64s)
    chord(["C4", "E4", "G4"], 56.0, 8.0, 0.5)
}

track("bass") {
    instrument("bass/jp8000_sawbass")

    // Intro (0-8s)
    note("C2", 0.0, 2.0, 0.7)
    note("F2", 2.0, 2.0, 0.7)
    note("G2", 4.0, 2.0, 0.7)
    note("C2", 6.0, 2.0, 0.7)

    // Verse bass loop (8-24s)
    loop(8.0, 24.0) {
        note("C2", 0.0, 0.5, 0.9)
        note("C2", 0.5, 0.25, 0.7)
        note("C2", 0.75, 0.25, 0.6)
        note("E2", 1.0, 0.5, 0.8)
        note("A2", 1.5, 0.5, 0.9)
        note("A2", 2.0, 0.25, 0.7)
        note("A2", 2.25, 0.25, 0.6)
        note("G2", 2.5, 0.5, 0.8)
        note("F2", 3.0, 0.5, 0.9)
        note("F2", 3.5, 0.25, 0.7)
        note("F2", 3.75, 0.25, 0.6)
        note("A2", 4.0, 0.5, 0.8)
        note("G2", 4.5, 0.5, 0.9)
        note("G2", 5.0, 0.25, 0.7)
        note("G2", 5.25, 0.25, 0.6)
        note("B2", 5.5, 0.5, 0.8)
    }

    // Chorus bass - more energetic (24-40s)
    loop(24.0, 40.0) {
        note("C2", 0.0, 0.25, 1.0)
        note("C2", 0.25, 0.25, 0.8)
        note("E2", 0.5, 0.5, 0.9)
        note("F2", 1.0, 0.25, 0.9)
        note("F2", 1.25, 0.25, 1.0)
        note("A2", 1.5, 0.25, 0.8)
        note("G2", 1.75, 0.5, 0.9)
        note("B2", 2.25, 0.25, 0.8)
    }

    // Verse 2 (40-56s)
    loop(40.0, 56.0) {
        note("C2", 0.0, 0.5, 0.9)
        note("C2", 0.5, 0.25, 0.7)
        note("C2", 0.75, 0.25, 0.6)
        note("E2", 1.0, 0.5, 0.8)
        note("A2", 1.5, 0.5, 0.9)
        note("A2", 2.0, 0.25, 0.7)
        note("A2", 2.25, 0.25, 0.6)
        note("G2", 2.5, 0.5, 0.8)
        note("F2", 3.0, 0.5, 0.9)
        note("F2", 3.5, 0.25, 0.7)
        note("F2", 3.75, 0.25, 0.6)
        note("A2", 4.0, 0.5, 0.8)
        note("G2", 4.5, 0.5, 0.9)
        note("G2", 5.0, 0.25, 0.7)
        note("G2", 5.25, 0.25, 0.6)
        note("B2", 5.5, 0.5, 0.8)
    }

    // Outro (56-64s)
    note("C2", 56.0, 8.0, 0.6)
}

track("drums") {
    instrument("drums/bedroom_drums")

    // Intro - light hihat (0-8s)
    loop(0.0, 8.0) {
        note("F#2", 0.0, 0.5, 0.4)
    }

    // Verse drums (8-24s) - kick on 1,3 snare on 2,4
    loop(8.0, 24.0) {
        note("C2", 0.0, 0.25, 0.9)
        note("F#2", 0.25, 0.25, 0.6)
        note("F#2", 0.5, 0.5, 0.4)
        note("D2", 1.0, 0.25, 0.8)
        note("F#2", 1.25, 0.25, 0.6)
        note("F#2", 1.5, 0.5, 0.4)
    }

    // Chorus - more intense (24-40s)
    loop(24.0, 40.0) {
        note("C2", 0.0, 0.25, 1.0)
        note("F#2", 0.25, 0.25, 0.8)
        note("F#2", 0.5, 0.25, 0.5)
        note("F#2", 0.75, 0.25, 0.6)
        note("D2", 1.0, 0.25, 0.9)
        note("F#2", 1.25, 0.25, 0.8)
        note("C2", 1.5, 0.25, 0.7)
        note("F#2", 1.75, 0.25, 0.5)
    }

    // Verse 2 (40-56s)
    loop(40.0, 56.0) {
        note("C2", 0.0, 0.25, 0.9)
        note("F#2", 0.25, 0.25, 0.6)
        note("F#2", 0.5, 0.5, 0.4)
        note("D2", 1.0, 0.25, 0.8)
        note("F#2", 1.25, 0.25, 0.6)
        note("F#2", 1.5, 0.5, 0.4)
    }
}

track("melody") {
    instrument("piano/grand_piano_k")

    // Intro melody (0-8s)
    note("G5", 0.0, 0.5, 0.7)
    note("E5", 0.5, 0.5, 0.7)
    note("C5", 1.0, 1.0, 0.8)
    note("A5", 2.0, 0.5, 0.7)
    note("F5", 2.5, 0.5, 0.7)
    note("C5", 3.0, 1.0, 0.8)
    note("B5", 4.0, 0.5, 0.7)
    note("G5", 4.5, 0.5, 0.7)
    note("D5", 5.0, 1.0, 0.8)
    note("E5", 6.0, 2.0, 0.9)

    // Verse melody (8-24s)
    loop(8.0, 24.0) {
        note("E5", 0.0, 0.5, 0.8)
        note("G5", 0.5, 0.5, 0.8)
        note("C6", 1.0, 1.0, 0.9)
        note("B5", 2.0, 0.5, 0.7)
        note("A5", 2.5, 0.5, 0.8)
        note("G5", 3.0, 0.5, 0.7)
        note("E5", 3.5, 1.0, 0.8)
        note("D5", 4.5, 0.5, 0.7)
        note("C5", 5.0, 1.5, 0.9)
        note("C5", 6.5, 0.5, 0.6)
    }

    // Chorus melody (24-40s)
    loop(24.0, 40.0) {
        note("C6", 0.0, 0.5, 1.0)
        note("B5", 0.5, 0.5, 0.9)
        note("G5", 1.0, 0.5, 0.9)
        note("C6", 1.5, 0.5, 1.0)
        note("D6", 2.0, 1.0, 1.0)
        note("B5", 3.0, 0.5, 0.9)
        note("G5", 3.5, 0.5, 0.8)
    }

    // Verse 2 (40-56s)
    loop(40.0, 56.0) {
        note("E5", 0.0, 0.5, 0.8)
        note("G5", 0.5, 0.5, 0.8)
        note("C6", 1.0, 1.0, 0.9)
        note("B5", 2.0, 0.5, 0.7)
        note("A5", 2.5, 0.5, 0.8)
        note("G5", 3.0, 0.5, 0.7)
        note("E5", 3.5, 1.0, 0.8)
        note("D5", 4.5, 0.5, 0.7)
        note("C5", 5.0, 1.5, 0.9)
        note("C5", 6.5, 0.5, 0.6)
    }

    // Outro (56-64s)
    note("E5", 56.0, 2.0, 0.7)
    note("C5", 58.0, 6.0, 0.5)
}"""

# --- helpers --- #

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTES[midi % 12]
    return f"{note}{octave}"