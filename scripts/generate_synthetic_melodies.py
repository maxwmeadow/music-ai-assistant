# scripts/generate_synthetic_melodies.py
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import random


def generate_melody(num_notes=10, min_note=48, max_note=84):
    """Generate a random melody with perfect pitch"""
    notes = [random.randint(min_note, max_note) for _ in range(num_notes)]
    durations = [random.uniform(0.3, 1.5) for _ in range(num_notes)]
    start_times = [sum(durations[:i]) + i * 0.1 for i in range(num_notes)]
    return notes, start_times, durations


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synthesize_note(freq, duration, sr=16000):
    t = np.linspace(0, duration, int(sr * duration))

    # Harmonics
    fundamental = np.sin(2 * np.pi * freq * t)
    harmonic2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    harmonic3 = 0.2 * np.sin(2 * np.pi * freq * 3 * t)
    audio = fundamental + harmonic2 + harmonic3

    # Envelope
    attack = int(0.05 * len(t))
    release = int(0.1 * len(t))
    envelope_curve = np.ones(len(t))
    envelope_curve[:attack] = np.linspace(0, 1, attack)
    envelope_curve[-release:] = np.linspace(1, 0, release)
    audio = audio * envelope_curve

    # Noise
    noise = np.random.randn(len(audio)) * 0.02
    audio = audio + noise

    return audio * 0.8


def create_melody_audio(notes, start_times, durations, sr=16000):
    total_duration = start_times[-1] + durations[-1] + 0.5
    total_samples = int(total_duration * sr)
    audio = np.zeros(total_samples)

    for note, start, duration in zip(notes, start_times, durations):
        freq = midi_to_hz(note)
        note_audio = synthesize_note(freq, duration, sr)
        start_sample = int(start * sr)
        end_sample = start_sample + len(note_audio)
        if end_sample <= len(audio):
            audio[start_sample:end_sample] += note_audio

    return audio


# ===== PATHS =====
project_root = Path(__file__).parent.parent

# Audio: Store on G: drive with your other vocals
audio_output_dir = Path("G:/music_data_medium/organized/synthetic_vocal")

# Labels: Keep on C: drive (tiny files)
labels_output_dir = project_root / "dataset" / "synthetic_labels"

# Create directories
audio_output_dir.mkdir(parents=True, exist_ok=True)
labels_output_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root: {project_root}")
print(f"Audio output (G: drive): {audio_output_dir}")
print(f"Labels output (C: drive): {labels_output_dir}")
print()

# ===== GENERATE =====
num_samples = 500

print(f"Generating {num_samples} synthetic melodies...")
for i in range(num_samples):
    num_notes = random.randint(5, 15)
    notes, start_times, durations = generate_melody(num_notes)
    audio = create_melody_audio(notes, start_times, durations)

    # Save audio on G: drive
    audio_filename = f"synth_{i:04d}.wav"
    audio_path = audio_output_dir / audio_filename
    sf.write(audio_path, audio, 16000)

    # Save label on C: drive (with absolute path to G: drive audio)
    label = {
        'audio_path': str(audio_path.absolute()),  # Points to G: drive
        'notes': notes,
        'start_times': start_times,
        'durations': durations,
        'confidence': 1.0
    }

    label_filename = f"synth_{i:04d}_label.json"
    label_path = labels_output_dir / label_filename

    with open(label_path, 'w') as f:
        json.dump(label, f)

    if (i + 1) % 50 == 0:
        print(f"  Generated {i + 1}/{num_samples}")

print(f"\n✓ Created {num_samples} synthetic samples")
print(f"  Audio files (G: drive): {audio_output_dir}")
print(f"  Label files (C: drive): {labels_output_dir}")
print(f"\nDisk usage:")
print(f"  G: drive audio: ~{num_samples * 2} MB")
print(f"  C: drive labels: ~{num_samples * 1} KB")