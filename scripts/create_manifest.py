import json
from pathlib import Path
import argparse


def create_training_manifest(label_dir, output_file, min_notes=1):
    """Create manifest of valid training samples"""
    label_dir = Path(label_dir)
    valid_samples = []

    total_duration = 0
    total_notes = 0

    for label_file in label_dir.glob('*_label.json'):
        try:
            with open(label_file) as f:
                data = json.load(f)

            # Filter out empty or bad labels
            if len(data['notes']) >= min_notes:
                audio_path = data['audio_path']

                valid_samples.append({
                    'audio_path': audio_path,
                    'label_path': str(label_file)
                })

                total_notes += len(data['notes'])
                # Estimate duration from last note
                if data['start_times'] and data['durations']:
                    duration = data['start_times'][-1] + data['durations'][-1]
                    total_duration += duration

        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue

    manifest = {
        "samples": valid_samples,
        "stats": {
            "total_samples": len(valid_samples),
            "total_duration_seconds": total_duration,
            "avg_notes_per_sample": total_notes / len(valid_samples) if valid_samples else 0,
            "total_notes": total_notes
        }
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest created: {output_path}")
    print(f"Total samples: {manifest['stats']['total_samples']}")
    print(
        f"Total duration: {manifest['stats']['total_duration_seconds']:.1f}s ({manifest['stats']['total_duration_seconds'] / 3600:.1f} hours)")
    print(f"Avg notes per sample: {manifest['stats']['avg_notes_per_sample']:.1f}")
    print(f"Total notes: {manifest['stats']['total_notes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='dataset/train_manifest.json')
    parser.add_argument('--min_notes', type=int, default=3, help='Minimum notes required')

    args = parser.parse_args()
    create_training_manifest(args.label_dir, args.output, args.min_notes)