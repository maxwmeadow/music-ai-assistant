"""
Audio Pitch Analysis Script
Analyzes the actual pitches present in an audio file
"""

import librosa
import numpy as np
import sys
from pathlib import Path


def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"


def analyze_audio(audio_path: str):
    """Analyze pitches in an audio file."""
    print("=" * 60)
    print(f"ANALYZING: {audio_path}")
    print("=" * 60)

    # Load audio
    print("\n[1] Loading audio...")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    print(f"    Duration: {duration:.2f}s")
    print(f"    Sample rate: {sr} Hz")
    print(f"    Samples: {len(audio)}")

    # Extract pitch using pYIN (same as your training data)
    print("\n[2] Extracting pitches with pYIN...")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048,
        hop_length=512
    )

    # Filter out unvoiced frames
    voiced_f0 = f0[voiced_flag]
    voiced_times = librosa.frames_to_time(
        np.where(voiced_flag)[0],
        sr=sr,
        hop_length=512
    )

    if len(voiced_f0) == 0:
        print("\n❌ NO PITCHED AUDIO DETECTED!")
        print("   This audio has no clear humming/singing.")
        print("   Try recording yourself humming a clear melody.")
        return

    print(f"    Voiced frames: {len(voiced_f0)} / {len(f0)} ({len(voiced_f0) / len(f0) * 100:.1f}%)")

    # Convert to MIDI notes
    print("\n[3] Converting to MIDI notes...")
    midi_notes = librosa.hz_to_midi(voiced_f0)
    rounded_midi = np.round(midi_notes).astype(int)

    # Find unique notes
    unique_notes = np.unique(rounded_midi)
    print(f"    Unique MIDI notes detected: {len(unique_notes)}")
    print(f"    Range: {midi_to_note_name(unique_notes.min())} to {midi_to_note_name(unique_notes.max())}")

    # Analyze note distribution
    print("\n[4] Note Distribution:")
    from collections import Counter
    note_counts = Counter(rounded_midi)
    sorted_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"    Top 10 most common notes:")
    for midi_note, count in sorted_notes[:10]:
        percentage = (count / len(rounded_midi)) * 100
        note_name = midi_to_note_name(midi_note)
        bar = "█" * int(percentage / 2)
        print(f"    {note_name:4s} (MIDI {midi_note:3d}): {bar:25s} {percentage:5.1f}% ({count:4d} frames)")

    # Detect note transitions
    print("\n[5] Detecting Note Transitions...")
    transitions = []
    current_note = rounded_midi[0]
    current_start = voiced_times[0]

    for i in range(1, len(rounded_midi)):
        if rounded_midi[i] != current_note:
            # Note changed
            duration = voiced_times[i] - current_start
            if duration > 0.05:  # Filter very short notes
                transitions.append({
                    'note': current_note,
                    'start': current_start,
                    'duration': duration
                })
            current_note = rounded_midi[i]
            current_start = voiced_times[i]

    # Add last note
    duration = voiced_times[-1] - current_start
    if duration > 0.05:
        transitions.append({
            'note': current_note,
            'start': current_start,
            'duration': duration
        })

    print(f"    Detected {len(transitions)} note segments (>50ms)")
    print(f"\n    Melody sequence:")
    for i, trans in enumerate(transitions[:20]):  # Show first 20
        note_name = midi_to_note_name(trans['note'])
        print(f"    {i + 1:2d}. {note_name:4s} @ {trans['start']:5.2f}s for {trans['duration']:.3f}s")

    if len(transitions) > 20:
        print(f"    ... and {len(transitions) - 20} more notes")

    # Pitch stability analysis
    print("\n[6] Pitch Stability:")
    pitch_std = np.std(midi_notes)
    pitch_range = np.ptp(midi_notes)  # peak-to-peak
    print(f"    Standard deviation: {pitch_std:.2f} semitones")
    print(f"    Range: {pitch_range:.2f} semitones")

    if pitch_std < 1.0:
        print("    ✅ Very stable pitch (good for model)")
    elif pitch_std < 2.0:
        print("    ✅ Stable pitch (good for model)")
    elif pitch_std < 3.0:
        print("    ⚠️  Moderate vibrato/drift")
    else:
        print("    ⚠️  High variation (may confuse model)")

    # Energy analysis
    print("\n[7] Energy/Volume Analysis:")
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)

    print(f"    Average RMS: {avg_rms:.4f}")
    print(f"    Max RMS: {max_rms:.4f}")

    if avg_rms < 0.01:
        print("    ⚠️  Very quiet recording")
    elif avg_rms < 0.05:
        print("    ⚠️  Quiet recording")
    else:
        print("    ✅ Good recording volume")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Duration:        {duration:.2f}s")
    print(f"Voiced:          {len(voiced_f0) / len(f0) * 100:.1f}%")
    print(f"Unique notes:    {len(unique_notes)}")
    print(f"Note range:      {midi_to_note_name(unique_notes.min())} to {midi_to_note_name(unique_notes.max())}")
    print(f"Pitch stability: {pitch_std:.2f} semitones std")
    print(f"Note segments:   {len(transitions)}")

    # Expected model performance
    print("\n" + "=" * 60)
    print("EXPECTED MODEL PERFORMANCE")
    print("=" * 60)

    if len(voiced_f0) / len(f0) < 0.3:
        print("❌ Low voiced content - model will struggle")
    elif pitch_std > 3.0:
        print("⚠️  High pitch variation - may be challenging")
    elif len(unique_notes) < 3:
        print("⚠️  Very few notes - simple melody")
    else:
        print("✅ Good audio for model prediction!")
        print(f"   Expect model to detect {len(transitions) // 2} to {len(transitions)} notes")

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <audio_file.wav>")
        print("\nExample:")
        print("  python analyze_audio.py backend/audio_uploads/hum2melody_20251009_161815.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    analyze_audio(audio_path)