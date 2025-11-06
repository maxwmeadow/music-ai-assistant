import argparse
from pathlib import Path
import json
from statistics import mean
from tqdm import tqdm
import sys
import numpy as np

EPS = 1e-6

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("preprocessed_folder", type=str)
    p.add_argument("--grid", type=float, default=0.125)
    p.add_argument("--epsilon", type=float, default=1e-3)
    return p.parse_args()

def is_on_grid(value: float, grid: float, eps: float) -> bool:
    return abs(round(value / grid) * grid - value) <= eps

def validate_file(path: Path, grid: float, eps: float):
    data = json.loads(path.read_text())
    tracks = data.get("tracks", [])
    total_notes = sum(len(t.get("notes", [])) for t in tracks)
    issues = []
    if total_notes == 0:
        issues.append("empty_sequence")

    per_instrument_counts = {}
    for t in tracks:
        inst = t.get("instrument", "unknown")
        note_list = t.get("notes", [])
        per_instrument_counts[inst] = len(note_list)
        for n in note_list:
            start = n.get("start", 0.0)
            vel = n.get("velocity", -1.0)
            dur = n.get("duration", 0.0)

            # alignment
            if not is_on_grid(start, grid, eps):
                issues.append(f"start_not_on_grid:{start:.4f}")

            # velocity range
            if not (0.0 - EPS <= vel <= 1.0 + EPS):
                issues.append(f"velocity_out_of_range:{vel}")

            # duration bounds
            if not (0.05 - EPS <= dur <= 4.0 + EPS):
                issues.append(f"duration_out_of_bounds:{dur}")

    return {"file": path.name, "total_notes": total_notes, "per_instrument": per_instrument_counts, "issues": issues}

def main():
    args = parse_args()
    folder = Path(args.preprocessed_folder)
    files = sorted(folder.glob("*_preprocessed.json"))
    if not files:
        print("No preprocessed JSONs found in", folder)
        sys.exit(1)

    results = []
    for f in tqdm(files):
        results.append(validate_file(f, args.grid, args.epsilon))

    issues = [r for r in results if r["issues"]]
    print("\n--- Validation Summary ---")
    print(f"Total preprocessed files: {len(files)}")
    print(f"Files with issues: {len(issues)}")
    if issues:
        print("\nSample Issues:")
        for i in issues[:10]:
            print(i["file"], "->", i["issues"][:5])

    # stats
    total_notes = [r["total_notes"] for r in results]
    print(f"\nAverage notes per file: {mean(total_notes):.2f}")
    # per-instrument averages
    all_insts = {}
    for r in results:
        for inst, cnt in r["per_instrument"].items():
            all_insts.setdefault(inst, []).append(cnt)
    for inst, arr in all_insts.items():
        print(f"  {inst}: avg {mean(arr):.2f}")

    # Write issues log
    issues_log = folder / "preprocess_issues.log"
    with open(issues_log, "w", encoding="utf-8") as f:
        for r in issues:
            f.write(json.dumps(r) + "\n")
    if issues:
        print(f"\nWrote issues to: {issues_log}")

if __name__ == "__main__":
    main()