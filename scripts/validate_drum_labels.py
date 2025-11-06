import json
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np


def validate_label(json_path: Path):
    try:
        data = json.loads(json_path.read_text())
        drum_hits = data.get("drum_hits", {})
        total_hits = data.get("total_hits", 0)

        if total_hits == 0:
            return False, "no_hits"

        if not all(isinstance(v, list) for v in drum_hits.values()):
            return False, "invalid_format"

        if not (5 <= total_hits <= 100):
            return False, "outlier"

        return True, None
    except Exception as e:
        return False, str(e)


def main(folder_path: str):
    folder = Path(folder_path)
    json_files = list(folder.glob("*_label.json"))

    if not json_files:
        print(f"No label files found in {folder}")
        return

    stats = {"kick": [], "snare": [], "hihat": []}
    fails = []

    for f in tqdm(json_files):
        ok, reason = validate_label(f)
        if not ok:
            fails.append((f.name, reason))
            continue

        data = json.loads(f.read_text())
        for k in stats.keys():
            stats[k].append(len(data["drum_hits"].get(k, [])))

    print("\n--- Validation Summary ---")
    print(f"Total JSONs checked: {len(json_files)}")
    print(f"Valid: {len(json_files) - len(fails)}, Invalid: {len(fails)}")

    if fails:
        print("\nInvalid files:")
        for name, reason in fails[:10]:
            print(f"  {name} → {reason}")

    for k, arr in stats.items():
        if arr:
            print(f"\n{k.capitalize()} avg hits: {np.mean(arr):.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_drum_labels.py /path/to/drum/folder")
    else:
        main(sys.argv[1])