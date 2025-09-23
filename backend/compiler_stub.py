from .schemas import IR

"""
FILE compiler_stub.py
Temporary file to provide some backend testing without
the actual compiler
"""


def json_ir_to_dsl(ir: IR) -> str:
    """
    Convert JSON IR into DSL code (stub version).
    """
    tempo = ir.metadata.get("tempo", 120)
    lines = [f"tempo({tempo})", ""]

    for track in ir.tracks:
        lines.append(f'track("{track.id}") {{')

        if track.instrument:
            lines.append(f'  instrument("{track.instrument}")')

        if track.notes:
            for note in track.notes:
                # We could later add a MIDI→note name converter like in generator.js
                lines.append(f'  note("{midi_to_note(note.pitch)}", {note.duration}, {note.velocity})')

        if track.samples:
            for sample in track.samples:
                lines.append(f'  {sample.sample}({sample.start})')

        lines.append("}\n")

    return "\n".join(lines)

def compile_scale_to_dsl() -> str:
    """Demo/test stub DSL."""
    return """tempo(128)

track("drums") {
  kick(0.0)
  snare(1.0)
}

track("bass") {
  instrument("fm_bass")
  note("A2", 2.0, 0.9)
}
"""

# --- helpers --- #

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTES[midi % 12]
    return f"{note}{octave}"