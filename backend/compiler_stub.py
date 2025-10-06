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

track("melody") {
  instrument("piano/grand_piano_k")
  note("E4", 0.5, 0.8)
  note("D4", 0.5, 0.7)
  note("C4", 0.5, 0.8)
  note("D4", 0.5, 0.7)
  note("E4", 0.5, 0.8)
  note("E4", 0.5, 0.8)
  note("E4", 1.0, 0.9)
}

track("chords") {
  instrument("synth/pad/pd_fatness_pad")
  chord(["C4", "E4", "G4"], 2.0, 0.6)
  chord(["F4", "A4", "C5"], 2.0, 0.6)
  chord(["G4", "B4", "D5"], 2.0, 0.6)
  chord(["C4", "E4", "G4"], 2.0, 0.6)
}

track("bass") {
  instrument("bass/jp8000_sawbass")
  note("C2", 2.0, 0.9)
  note("F2", 2.0, 0.9)
  note("G2", 2.0, 0.9)
  note("C2", 2.0, 0.9)
}

track("drums") {
  instrument("drums/bedroom_drums")
  note("C2", 0.5, 1.0)
  note("F#2", 0.5, 0.7)
  note("D2", 0.5, 1.0)
  note("F#2", 0.5, 0.7)
  note("C2", 0.5, 1.0)
  note("F#2", 0.5, 0.7)
  note("D2", 0.5, 1.0)
  note("F#2", 0.5, 0.7)
}
"""

# --- helpers --- #

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTES[midi % 12]
    return f"{note}{octave}"