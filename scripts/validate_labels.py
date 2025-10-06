import json
from pathlib import Path
import argparse
import numpy as np


def validate_single_label(label_path):
    """Validate a single label file"""
    issues = []

    try:
        with open(label_path) as f:
            data = json.load(f)

        notes = data.get('notes', [])
        start_times = data.get('start_times', [])
        durations = data.get('durations', [])

        # Check for empty notes
        if len(notes) == 0:
            issues.append("empty_notes")
            return issues

        # Check array lengths match
        if not (len(notes) == len(start_times) == len(durations)):
            issues.append("length_mismatch")

        # Check MIDI range (21-108 is standard piano range)
        for note in notes:
            if note < 21 or note > 108:
                issues.append("out_of_range")
                break

        # Check for negative durations
        if any(d < 0 for d in durations):
            issues.append("negative_duration")

        # Check for very short notes
        if any(d < 0.05 for d in durations):
            issues.append("very_short_notes")

        # Check start times are monotonically increasing
        if not all(start_times[i] <= start_times[i + 1] for i in range(len(start_times) - 1)):
            issues.append("non_monotonic_times")

        return issues

    except Exception as e:
        return [f"parse_error: {str(e)}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, required=True)
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    label_files = list(label_dir.glob('*_label.json'))

    print(f"Validating {len(label_files)} label files...")

    issue_counts = {}
    valid_count = 0

    for label_file in label_files:
        issues = validate_single_label(label_file)

        if not issues:
            valid_count += 1
        else:
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Validation Results:")
    print(f"Valid files: {valid_count}/{len(label_files)} ({100 * valid_count / len(label_files):.1f}%)")

    if issue_counts:
        print(f"\nIssues found:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count} files")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()