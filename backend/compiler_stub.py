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
    """Demo/test stub DSL."""
    return """tempo(128)

track("melody") {
    instrument("piano/grand_piano_k")
    note("E4", 0.0, 0.5, 0.8)
    note("D4", 0.5, 0.5, 0.7)
    note("C4", 1.5, 0.5, 0.8)
    note("D4", 2.5, 0.5, 0.7)
    note("E4", 3.0, 0.5, 0.8)
    note("E4", 3.5, 0.5, 0.8)
    note("E4", 4.0, 1.0, 0.9)
}
  
track("guitar") {
    instrument("guitar/rjs_guitar_new_strings")
}"""

# --- helpers --- #

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTES[midi % 12]
    return f"{note}{octave}"