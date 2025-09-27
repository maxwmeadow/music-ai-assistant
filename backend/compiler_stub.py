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
    return """tempo(100)

    track("drums") {
      kick(0.0)
      kick(2.0)
      kick(4.0)
      kick(6.0)
      kick(8.0)
      kick(10.0)
      kick(12.0)
      kick(14.0)
      snare(1.0)
      snare(3.0)
      snare(5.0)
      snare(7.0)
      snare(9.0)
      snare(11.0)
      snare(13.0)
      snare(15.0)
    }

    track("melody") {
      instrument("synth")
      note("C4", 1.0, 0.8)
      note("C4", 1.0, 0.8)
      note("G4", 1.0, 0.8)
      note("G4", 1.0, 0.8)
      note("A4", 1.0, 0.8)
      note("A4", 1.0, 0.8)
      note("G4", 2.0, 0.8)
      note("F4", 1.0, 0.8)
      note("F4", 1.0, 0.8)
      note("E4", 1.0, 0.8)
      note("E4", 1.0, 0.8)
      note("D4", 1.0, 0.8)
      note("D4", 1.0, 0.8)
      note("C4", 2.0, 0.8)
    }

    track("bass") {
      instrument("fm_bass")
      note("C2", 4.0, 0.9)
      note("F2", 4.0, 0.9)
      note("C2", 4.0, 0.9)
      note("G2", 4.0, 0.9)
    }

    track("arpeggios") {
      instrument("bell")
      note("C5", 0.5, 0.6)
      note("E5", 0.5, 0.6)
      note("G5", 0.5, 0.6)
      note("C6", 0.5, 0.6)
      note("G5", 0.5, 0.6)
      note("E5", 0.5, 0.6)
      note("C5", 0.5, 0.6)
      note("E5", 0.5, 0.6)
    }

    track("chord_stabs") {
      instrument("pad")
      note("C4", 0.5, 0.5)
      note("G4", 2.0, 0.3)
      note("F4", 2.0, 0.3)
      note("C4", 2.0, 0.3)
    }
"""

# --- helpers --- #

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note(midi: int) -> str:
    octave = (midi // 12) - 1
    note = NOTES[midi % 12]
    return f"{note}{octave}"