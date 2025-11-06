import argparse
import json
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from typing import List, Dict, Any, Tuple
import pretty_midi
import numpy as np
from tqdm import tqdm
import traceback
import sys

# ---------- CONFIG ----------
DEFAULT_SR = 16000
GRID_STEP_DEFAULT = 0.125         # seconds (default quantization step). Use 0.0625 for 32nd notes.
DURATION_MIN = 0.05               # seconds
DURATION_MAX = 4.0                # seconds
PITCH_MIN, PITCH_MAX = 0, 127
VELOCITY_MIN, VELOCITY_MAX = 0, 1
INSTRUMENT_MAP = {
    'melody': 0,
    'chords': 1,
    'bass': 2,
    'pads': 3
}
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_folder", type=str, help="Folder containing MIDI (.mid/.midi) or JSON sequence files")
    p.add_argument("--out", "-o", type=str, default="data/preprocessed", help="Output folder for preprocessed JSON")
    p.add_argument("--grid", type=float, default=GRID_STEP_DEFAULT, help="Temporal grid step (seconds)")
    p.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    p.add_argument("--tempo", type=float, default=120.0, help="Fallback tempo (BPM) for time grid if needed")
    return p.parse_args()

# ---------------- Utility functions ----------------

def load_midi_notes(path: Path) -> Tuple[List[Dict[str, Any]], float]:
    """Load notes from MIDI file. Returns list of note dicts and total duration."""
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for inst in pm.instruments:
        # pretty_midi uses channel 9 (index 9) for drums often; we still read them
        for n in inst.notes:
            notes.append({
                "pitch": int(n.pitch),
                "start": float(n.start),
                "duration": float(n.end - n.start),
                "velocity": float(n.velocity) / 127.0,  # normalize here (0-1)
                "instrument_name": inst.name or f"program_{inst.program}"
            })
    duration = pm.get_end_time() if pm.instruments else 0.0
    return notes, float(duration)

def load_json_sequence(path: Path) -> Tuple[List[Dict[str, Any]], float]:
    """Load notes from a JSON file. Expect list-like structure or IR format."""
    j = json.loads(path.read_text())
    notes = []
    duration = j.get("duration", 0.0)
    # Accept either IR format with "tracks" or a single "notes" list
    if isinstance(j.get("tracks"), list):
        for t in j["tracks"]:
            inst = t.get("instrument", "melody")
            for n in t.get("notes", []):
                notes.append({
                    "pitch": int(n["pitch"]),
                    "start": float(n["start"]),
                    "duration": float(n["duration"]),
                    "velocity": float(n.get("velocity", 1.0)),
                    "instrument_name": inst
                })
    elif isinstance(j.get("notes"), list):
        for n in j["notes"]:
            notes.append({
                "pitch": int(n["pitch"]),
                "start": float(n["start"]),
                "duration": float(n["duration"]),
                "velocity": float(n.get("velocity", 1.0)),
                "instrument_name": j.get("instrument", "melody")
            })
    else:
        # If JSON is a plain array of notes
        if isinstance(j, list):
            for n in j:
                notes.append({
                    "pitch": int(n["pitch"]),
                    "start": float(n["start"]),
                    "duration": float(n["duration"]),
                    "velocity": float(n.get("velocity", 1.0)),
                    "instrument_name": n.get("instrument", "melody")
                })
    return notes, float(duration)

def quantize_time(value: float, grid_step: float) -> float:
    return round(value / grid_step) * grid_step

def normalize_pitch(pitch: int) -> float:
    return (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)

def normalize_velocity(v: float) -> float:
    # Input might already be 0-1 (from MIDI), or 0-127 (from raw)
    if v > 1.0:
        v = v / 127.0
    return float(max(0.0, min(1.0, v)))

def clip_duration(d: float) -> float:
    return float(max(DURATION_MIN, min(DURATION_MAX, d)))

def instrument_heuristic(inst_name: str) -> str:
    name = (inst_name or "").lower()
    if "bass" in name or "synth bass" in name:
        return "bass"
    if "pad" in name or "pad" in name:
        return "pads"
    if "chord" in name or "piano" in name or "rhodes" in name:
        return "chords"
    # default to melody
    return "melody"

def notes_to_tracks(notes: List[Dict[str, Any]], grid_step: float, tempo: float) -> List[Dict[str, Any]]:
    """
    Group notes into tracks by instrument heuristic, quantize and normalize.
    Returns list of tracks in IR format.
    """
    grouped = {}
    for n in notes:
        inst = instrument_heuristic(n.get("instrument_name", "melody"))
        grouped.setdefault(inst, []).append(n)

    tracks = []
    for inst, nlist in grouped.items():
        processed_notes = []
        for n in nlist:
            start_q = quantize_time(n["start"], grid_step)
            dur_q = quantize_time(n["duration"], grid_step)
            dur_q = clip_duration(dur_q)
            pitch = int(np.clip(n["pitch"], PITCH_MIN, PITCH_MAX))
            vel = normalize_velocity(n.get("velocity", 1.0))

            processed_notes.append({
                "pitch": pitch,
                "start": float(start_q),
                "duration": float(dur_q),
                "velocity": float(vel)
            })

        # sort by start time
        processed_notes.sort(key=lambda x: x["start"])
        tracks.append({
            "instrument": inst,
            "embedding_id": INSTRUMENT_MAP.get(inst, 0),  # placeholder embedding id
            "notes": processed_notes
        })
    return tracks

def preprocess_single(file_path: Path, out_folder: Path, grid_step: float, tempo: float) -> Dict[str, Any]:
    """Load, process and save one file. Returns a status dict."""
    try:
        ext = file_path.suffix.lower()
        if ext in [".mid", ".midi"]:
            notes, duration = load_midi_notes(file_path)
            if duration == 0:
                # fallback: compute duration from notes
                duration = max((n["start"] + n["duration"]) for n in notes) if notes else 0.0
        elif ext in [".json"]:
            notes, duration = load_json_sequence(file_path)
        else:
            # unknown extension
            raise RuntimeError(f"Unsupported file extension: {ext}")

        # If no notes, return empty but save metadata
        tracks = notes_to_tracks(notes, grid_step, tempo)

        out = {
            "original_filename": str(file_path.name),
            "duration": float(duration),
            "note_count": sum(len(t["notes"]) for t in tracks),
            "tracks": tracks
        }

        out_folder.mkdir(parents=True, exist_ok=True)
        out_path = out_folder / (file_path.stem + "_preprocessed.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        return {"file": str(file_path.name), "ok": True, "out_path": str(out_path)}

    except Exception as e:
        # write error log next to file
        err_path = file_path.with_name(file_path.stem + "_preprocess_error.log")
        with open(err_path, "w", encoding="utf-8") as efile:
            efile.write(f"Error processing {file_path}:\n")
            efile.write(traceback.format_exc())
        return {"file": str(file_path.name), "ok": False, "error": str(e)}

def process_batch(file_paths: List[Path], out_folder: Path, grid_step: float, workers: int, tempo: float):
    fn = partial(preprocess_single, out_folder=out_folder, grid_step=grid_step, tempo=tempo)
    results = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap_unordered(fn, file_paths), total=len(file_paths)):
            results.append(r)
    return results

def find_inputs(folder: Path) -> List[Path]:
    exts = ["*.mid", "*.midi", "*.json"]
    files = []
    for e in exts:
        files.extend(sorted(folder.glob(e)))
    return files

def main():
    args = parse_args()
    in_folder = Path(args.input_folder)
    out_folder = Path(args.out)
    grid_step = float(args.grid)
    workers = int(args.workers)
    tempo = float(args.tempo)

    files = find_inputs(in_folder)
    if not files:
        print("No input files found in", in_folder)
        sys.exit(1)

    print(f"Processing {len(files)} files with {workers} workers to {out_folder} (grid={grid_step}s)")
    results = process_batch(files, out_folder, grid_step, workers, tempo)

    ok = sum(1 for r in results if r["ok"])
    fail = len(results) - ok
    print(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        print("Check *_preprocess_error.log files for details.")

if __name__ == "__main__":
    main()