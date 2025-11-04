"""
Compare Ground Truth Audio vs Model Predictions
Shows what was actually hummed vs what the model predicted
"""

import librosa
import numpy as np
import sys
import sqlite3
from pathlib import Path


def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"


def get_ground_truth_notes(audio_path: str):
    """Extract actual pitches from audio."""
    print("\n[GROUND TRUTH] Analyzing actual audio...")

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Extract pitch
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048,
        hop_length=512
    )

    # Get voiced segments
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) == 0:
        print("  ❌ No pitched audio detected!")
        return []

    voiced_times = librosa.frames_to_time(
        np.where(voiced_flag)[0],
        sr=sr,
        hop_length=512
    )

    # Convert to MIDI
    midi_notes = librosa.hz_to_midi(voiced_f0)
    rounded_midi = np.round(midi_notes).astype(int)

    # Detect note transitions
    transitions = []
    current_note = rounded_midi[0]
    current_start = voiced_times[0]

    for i in range(1, len(rounded_midi)):
        if rounded_midi[i] != current_note:
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

    print(f"  ✅ Detected {len(transitions)} ground truth notes")
    return transitions


def get_model_predictions(audio_filename: str, db_path: str = "backend/training_data.db"):
    """Get model predictions from database."""
    print("\n[MODEL PREDICTIONS] Loading from database...")

    if not Path(db_path).exists():
        print(f"  ❌ Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First, check what columns exist
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"  Available columns: {columns}")

    # Try different column names that might contain the predictions
    possible_columns = ['predicted_notes', 'prediction', 'notes', 'output', 'result']
    pred_column = None

    for col in possible_columns:
        if col in columns:
            pred_column = col
            break

    if not pred_column:
        print(f"  ❌ Could not find predictions column. Available: {columns}")
        print(f"  Showing all data for file: {Path(audio_filename).name}")

        # Show all data for this file
        cursor.execute(f"SELECT * FROM predictions WHERE audio_path LIKE ? ORDER BY id DESC LIMIT 1",
                       (f"%{Path(audio_filename).name}%",))
        result = cursor.fetchone()

        if result:
            print(f"  Row data:")
            for i, col_name in enumerate(columns):
                val = result[i]
                val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"    {col_name}: {val_str}")

        conn.close()
        return []

    # Find the prediction for this audio file
    cursor.execute(f"""
        SELECT id, {pred_column}
        FROM predictions 
        WHERE audio_path LIKE ? 
        ORDER BY id DESC 
        LIMIT 1
    """, (f"%{Path(audio_filename).name}%",))

    result = cursor.fetchone()
    conn.close()

    if not result:
        print(f"  ❌ No predictions found for {Path(audio_filename).name}")
        return []

    pred_id, predicted_json = result
    print(f"  ✅ Found prediction ID: {pred_id}")

    # Parse JSON
    import json
    try:
        predicted = json.loads(predicted_json)
    except json.JSONDecodeError as e:
        print(f"  ❌ Failed to parse predictions: {e}")
        print(f"  Raw data: {predicted_json[:200]}")
        return []

    notes = []

    # Handle different JSON structures
    if isinstance(predicted, list):
        # Direct list of notes
        for note in predicted:
            if isinstance(note, dict):
                notes.append({
                    'note': note.get('pitch', note.get('note', 60)),
                    'start': note.get('start', note.get('time', 0)),
                    'duration': note.get('duration', 0.5),
                    'velocity': note.get('velocity', note.get('confidence', 0.5))
                })
    elif isinstance(predicted, dict):
        # Could be wrapped in a structure
        if 'notes' in predicted:
            for note in predicted['notes']:
                notes.append({
                    'note': note.get('pitch', note.get('note', 60)),
                    'start': note.get('start', note.get('time', 0)),
                    'duration': note.get('duration', 0.5),
                    'velocity': note.get('velocity', note.get('confidence', 0.5))
                })

    print(f"  ✅ Model predicted {len(notes)} notes")
    return notes


def compare_notes(ground_truth, predictions):
    """Compare ground truth vs predictions."""
    print("\n" + "=" * 80)
    print("COMPARISON: GROUND TRUTH vs MODEL PREDICTIONS")
    print("=" * 80)

    if not ground_truth:
        print("\n❌ No ground truth available!")
        return

    if not predictions:
        print("\n❌ No model predictions available!")
        return

    # Show side by side
    print(f"\n{'GROUND TRUTH (what you hummed)':<40} | {'MODEL PREDICTIONS':<40}")
    print("-" * 80)

    max_len = max(len(ground_truth), len(predictions))

    for i in range(max_len):
        # Ground truth
        if i < len(ground_truth):
            gt = ground_truth[i]
            gt_note = midi_to_note_name(gt['note'])
            gt_str = f"{gt_note:4s} @ {gt['start']:5.2f}s for {gt['duration']:.3f}s"
        else:
            gt_str = ""

        # Prediction
        if i < len(predictions):
            pred = predictions[i]
            pred_note = midi_to_note_name(pred['note'])
            pred_str = f"{pred_note:4s} @ {pred['start']:5.2f}s for {pred['duration']:.3f}s (conf: {pred.get('velocity', 0):.2f})"

            # Check if it matches
            if i < len(ground_truth):
                time_diff = abs(pred['start'] - ground_truth[i]['start'])
                pitch_diff = abs(pred['note'] - ground_truth[i]['note'])

                if pitch_diff == 0 and time_diff < 0.5:
                    match = "✅"
                elif pitch_diff <= 2 and time_diff < 1.0:
                    match = "~"
                else:
                    match = "❌"
            else:
                match = "❓"
        else:
            pred_str = ""
            match = ""

        print(f"{gt_str:<40} | {pred_str:<38} {match}")

    # Calculate metrics
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)

    # Pitch accuracy (within 1 semitone and 0.5s)
    matches = 0
    for pred in predictions:
        for gt in ground_truth:
            time_diff = abs(pred['start'] - gt['start'])
            pitch_diff = abs(pred['note'] - gt['note'])
            if pitch_diff <= 1 and time_diff < 0.5:
                matches += 1
                break

    if predictions:
        precision = matches / len(predictions)
        print(f"Precision: {precision:.1%} ({matches}/{len(predictions)} predicted notes matched ground truth)")

    if ground_truth:
        recall = matches / len(ground_truth)
        print(f"Recall:    {recall:.1%} ({matches}/{len(ground_truth)} ground truth notes detected)")

    if predictions and ground_truth:
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"F1 Score:  {f1:.1%}")
        else:
            print(f"F1 Score:  0.0%")

    # Timing analysis
    if predictions and ground_truth:
        time_errors = []
        for pred in predictions:
            for gt in ground_truth:
                if abs(pred['note'] - gt['note']) <= 1:  # Same pitch (within 1 semitone)
                    time_diff = abs(pred['start'] - gt['start'])
                    time_errors.append(time_diff)
                    break

        if time_errors:
            avg_time_error = np.mean(time_errors)
            print(f"\nTiming Error: {avg_time_error:.3f}s average offset")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_predictions.py <audio_file.wav>")
        print("\nExample:")
        print("  python compare_predictions.py backend/audio_uploads/hum2melody_20251009_161815.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"ANALYZING: {audio_path}")
    print("=" * 80)

    # Get ground truth
    ground_truth = get_ground_truth_notes(audio_path)

    # Get predictions
    predictions = get_model_predictions(audio_path)

    # Compare
    compare_notes(ground_truth, predictions)


if __name__ == "__main__":
    main()