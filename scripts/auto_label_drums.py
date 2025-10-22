import librosa
import numpy as np

import json
import traceback
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import sys

# === PARAMETERS ===
SR = 16000
HOP_LENGTH = 512
WINDOW_BEFORE = 0.02  # seconds
WINDOW_AFTER = 0.08   # seconds
N_WORKERS = 8


def classify_hit(segment: np.ndarray, sr: int):
    """Heuristic drum type classifier based on spectral + temporal features."""
    if len(segment) == 0:
        return None, 0.0

    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
    rms = librosa.feature.rms(y=segment).mean()
    zcr = librosa.feature.zero_crossing_rate(y=segment).mean()

    velocity = float(np.clip(rms * 10, 0.0, 1.0))  # normalize roughly to 0–1

    if centroid < 200:  # low frequency → kick
        drum_type = "kick"
    elif centroid > 4000 or zcr > 0.2:  # bright or noisy → hihat
        drum_type = "hihat"
    else:
        drum_type = "snare"

    return drum_type, velocity


def process_file(audio_path: Path):
    """Process a single audio file and save JSON label file."""
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Detect onsets in time units
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units='time', hop_length=HOP_LENGTH, backtrack=True
        )

        drum_hits = {'kick': [], 'snare': [], 'hihat': []}

        for onset_time in onsets:
            start_sample = int(max(0, (onset_time - WINDOW_BEFORE) * sr))
            end_sample = int(min(len(y), (onset_time + WINDOW_AFTER) * sr))
            segment = y[start_sample:end_sample]

            drum_type, velocity = classify_hit(segment, sr)
            if drum_type:
                drum_hits[drum_type].append({'time': float(onset_time), 'velocity': velocity})

        total_hits = sum(len(v) for v in drum_hits.values())

        output = {
            'audio_path': str(audio_path),
            'duration': float(duration),
            'drum_hits': drum_hits,
            'total_hits': int(total_hits)
        }

        out_path = audio_path.with_name(audio_path.stem + "_label.json")
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

        return {"file": str(audio_path.name), "hits": total_hits, "ok": True}

    except Exception as e:
        err_path = audio_path.with_name(audio_path.stem + "_error.log")
        with open(err_path, 'w') as f:
            f.write(f"Error processing {audio_path}:\n{traceback.format_exc()}")
        return {"file": str(audio_path.name), "hits": 0, "ok": False, "error": str(e)}


def main(folder_path: str):
    folder = Path(folder_path)
    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))

    if not audio_files:
        print(f"No audio files found in {folder}")
        return

    print(f"Processing {len(audio_files)} drum files with {N_WORKERS} workers...")

    with Pool(N_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, audio_files), total=len(audio_files)))

    ok = sum(1 for r in results if r["ok"])
    fail = len(results) - ok

    print(f"\n✅ Completed: {ok} files labeled, ❌ {fail} failed.")
    if fail:
        print("Check *_error.log files for details.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/auto_label_drums.py /path/to/drum/folder")
    else:
        main(sys.argv[1])