# scripts/create_combined_manifest.py
import json
from pathlib import Path
import shutil

print("=" * 60)
print("CREATING COMBINED TRAINING MANIFEST")
print("=" * 60)

# Get project root
project_root = Path(__file__).parent.parent

# Source directories
vocal_labels_dir = project_root / "dataset" / "labels"
synthetic_labels_dir = project_root / "dataset" / "synthetic_labels"

# Output directory
combined_dir = project_root / "dataset" / "combined_labels"
combined_dir.mkdir(parents=True, exist_ok=True)

print(f"\nProject root: {project_root}")
print(f"Vocal labels: {vocal_labels_dir}")
print(f"Synthetic labels: {synthetic_labels_dir}")
print(f"Output: {combined_dir}")
print()

# Count existing files
vocal_labels = list(vocal_labels_dir.glob("*.json"))
synthetic_labels = list(synthetic_labels_dir.glob("*.json"))

print(f"Found {len(vocal_labels)} vocal labels")
print(f"Found {len(synthetic_labels)} synthetic labels")
print(f"Total: {len(vocal_labels) + len(synthetic_labels)} samples")
print()

# Copy vocal labels with prefix
print("Copying vocal labels...")
vocal_copied = 0
for label_file in vocal_labels:
    dest = combined_dir / f"vocal_{label_file.name}"
    shutil.copy2(label_file, dest)
    vocal_copied += 1
    if vocal_copied % 100 == 0:
        print(f"  Copied {vocal_copied}/{len(vocal_labels)} vocal labels")

print(f"✓ Copied {vocal_copied} vocal labels")

# Copy synthetic labels (no prefix needed, already have synth_ in name)
print("\nCopying synthetic labels...")
synthetic_copied = 0
for label_file in synthetic_labels:
    dest = combined_dir / label_file.name
    shutil.copy2(label_file, dest)
    synthetic_copied += 1
    if synthetic_copied % 100 == 0:
        print(f"  Copied {synthetic_copied}/{len(synthetic_labels)} synthetic labels")

print(f"✓ Copied {synthetic_copied} synthetic labels")

# Verify
combined_labels = list(combined_dir.glob("*.json"))
print()
print("=" * 60)
print(f"✓ COMBINED DATASET CREATED")
print("=" * 60)
print(f"Total labels in combined_labels/: {len(combined_labels)}")
print(f"  - Vocal: {len([f for f in combined_labels if f.name.startswith('vocal_')])}")
print(f"  - Synthetic: {len([f for f in combined_labels if f.name.startswith('synth_')])}")
print()
print(f"Location: {combined_dir}")
print(f"Disk usage: ~{len(combined_labels) * 1} KB")
print()

# Quick validation - check a few samples
print("Validating samples...")
sample_files = list(combined_labels)[:5]
valid_count = 0
for sample_file in sample_files:
    try:
        with open(sample_file) as f:
            data = json.load(f)
            # Check required fields
            assert 'audio_path' in data
            assert 'notes' in data
            assert 'start_times' in data
            assert 'durations' in data

            # Check audio file exists
            audio_path = Path(data['audio_path'])
            if audio_path.exists():
                valid_count += 1
            else:
                print(f"  Warning: Audio not found for {sample_file.name}")
    except Exception as e:
        print(f"  Error validating {sample_file.name}: {e}")

print(f"✓ Validated {valid_count}/{len(sample_files)} sample files")

if valid_count == len(sample_files):
    print("\n✓ All samples look good!")
    print()
    print("Ready to train with:")
    print(
        f"  python train_hum2melody.py --labels dataset/combined_labels --batch-size 24 --lr 0.001 --epochs 50 --num-workers 4 --patience 10")
else:
    print(f"\n⚠️  Some samples had issues - check paths")

print()