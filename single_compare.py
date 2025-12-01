"""
Simple Ground Truth vs Model Comparison
No database needed - just runs the model directly on audio
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

def get_ground_truth(audio_path: str):
    """Extract actual pitches from audio."""
    print("\n" + "=" * 80)
    print("[1/2] GROUND TRUTH - What you actually hummed")
    print("=" * 80)

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
    notes = []
    current_note = rounded_midi[0]
    current_start = voiced_times[0]

    for i in range(1, len(rounded_midi)):
        if rounded_midi[i] != current_note:
            duration = voiced_times[i] - current_start
            if duration > 0.05:  # Filter very short notes
                notes.append({
                    'pitch': current_note,
                    'start': current_start,
                    'duration': duration
                })
            current_note = rounded_midi[i]
            current_start = voiced_times[i]

    # Add last note
    duration = voiced_times[-1] - current_start
    if duration > 0.05:
        notes.append({
            'pitch': current_note,
            'start': current_start,
            'duration': duration
        })

    print(f"\nDetected {len(notes)} notes:")
    for i, note in enumerate(notes[:15], 1):
        note_name = midi_to_note_name(note['pitch'])
        print(f"  {i:2d}. {note_name:4s} @ {note['start']:5.2f}s for {note['duration']:.3f}s")

    if len(notes) > 15:
        print(f"  ... and {len(notes) - 15} more")

    return notes

def get_model_predictions(audio_path: str):
    """Run the model on audio and get predictions."""
    print("\n" + "=" * 80)
    print("[2/2] MODEL PREDICTIONS - What the model predicted")
    print("=" * 80)

    # Import the predictor
    sys.path.insert(0, str(Path(__file__).parent / 'backend'))

    try:
        from backend.inference.predictor import ImprovedMelodyPredictor
        print("✅ Predictor imported")
    except ImportError as e:
        print(f"❌ Failed to import predictor: {e}")
        print("Make sure you're running this from the project root!")
        return []

    # Load the model
    checkpoint_path = Path("backend/checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return []

    print(f"Loading model from: {checkpoint_path}")

    try:
        predictor = ImprovedMelodyPredictor(
            str(checkpoint_path),
            threshold=0.3,  # Use RAW mode threshold
            min_note_duration=0.05
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return []

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Run prediction (RAW mode if available)
    print("Running prediction...")
    try:
        if hasattr(predictor, 'predict_from_audio_RAW'):
            print("Using RAW prediction mode (no post-processing)")
            track = predictor.predict_from_audio_RAW(audio)
        else:
            print("Using normal prediction mode")
            track = predictor.predict_from_audio(audio)

        print(f"✅ Model generated {len(track.notes)} notes")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Convert Track.notes to our format
    notes = []
    print(f"\nPredicted {len(track.notes)} notes:")
    for i, note in enumerate(track.notes[:15], 1):
        # Note object has pitch, start, duration, velocity
        note_name = midi_to_note_name(note.pitch)
        print(f"  {i:2d}. {note_name:4s} @ {note.start:5.2f}s for {note.duration:.3f}s (conf: {note.velocity:.2f})")

        notes.append({
            'pitch': note.pitch,
            'start': note.start,
            'duration': note.duration,
            'velocity': note.velocity
        })

    if len(track.notes) > 15:
        print(f"  ... and {len(track.notes) - 15} more")

    return notes

def compare(ground_truth, predictions):
    """Compare ground truth vs predictions."""
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if not ground_truth:
        print("\n❌ No ground truth available!")
        return

    if not predictions:
        print("\n❌ No model predictions available!")
        return

    # Side by side
    print(f"\n{'GROUND TRUTH':<45} | {'MODEL PREDICTION':<45}")
    print("-" * 80)

    max_len = max(len(ground_truth), len(predictions))

    for i in range(max_len):
        # Ground truth
        if i < len(ground_truth):
            gt = ground_truth[i]
            gt_note = midi_to_note_name(gt['pitch'])
            gt_str = f"{i+1:2d}. {gt_note:4s} @ {gt['start']:5.2f}s for {gt['duration']:.3f}s"
        else:
            gt_str = ""

        # Prediction
        if i < len(predictions):
            pred = predictions[i]
            pred_note = midi_to_note_name(pred['pitch'])
            pred_str = f"{i+1:2d}. {pred_note:4s} @ {pred['start']:5.2f}s for {pred['duration']:.3f}s (v:{pred['velocity']:.2f})"

            # Check match
            if i < len(ground_truth):
                time_diff = abs(pred['start'] - ground_truth[i]['start'])
                pitch_diff = abs(pred['pitch'] - ground_truth[i]['pitch'])

                if pitch_diff == 0 and time_diff < 0.5:
                    match = "✅ EXACT"
                elif pitch_diff <= 1 and time_diff < 0.5:
                    match = "✅ CLOSE"
                elif pitch_diff <= 2 and time_diff < 1.0:
                    match = "~ SIMILAR"
                else:
                    match = "❌ MISS"
            else:
                match = "❓ EXTRA"
        else:
            pred_str = ""
            match = "❌ MISSING"

        print(f"{gt_str:<45} | {pred_str:<40} {match}")

    # Calculate metrics
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)

    # Find matches (within 1 semitone and 0.5s)
    matches = 0
    for pred in predictions:
        for gt in ground_truth:
            time_diff = abs(pred['start'] - gt['start'])
            pitch_diff = abs(pred['pitch'] - gt['pitch'])
            if pitch_diff <= 1 and time_diff < 0.5:
                matches += 1
                break

    precision = matches / len(predictions) if predictions else 0
    recall = matches / len(ground_truth) if ground_truth else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.1%} ({matches}/{len(predictions)} predictions were correct)")
    print(f"Recall:    {recall:.1%} ({matches}/{len(ground_truth)} ground truth notes detected)")
    print(f"F1 Score:  {f1:.1%}")

    # Timing analysis
    time_errors = []
    for pred in predictions:
        for gt in ground_truth:
            if abs(pred['pitch'] - gt['pitch']) <= 1:
                time_diff = abs(pred['start'] - gt['start'])
                time_errors.append(time_diff)
                break

    if time_errors:
        avg_time_error = np.mean(time_errors)
        print(f"\nAverage timing error: {avg_time_error:.3f}s")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if f1 >= 0.8:
        print("🎉 EXCELLENT - Model is working very well!")
    elif f1 >= 0.6:
        print("✅ GOOD - Model is working, could use some tuning")
    elif f1 >= 0.4:
        print("⚠️  OKAY - Model learned something but needs improvement")
    elif f1 >= 0.2:
        print("❌ POOR - Model is struggling, needs more training")
    else:
        print("💀 VERY POOR - Model is basically guessing randomly")

    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_compare.py <audio_file.wav>")
        print("\nExample:")
        print("  python simple_compare.py backend/audio_uploads/hum2melody_20251009_163236.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    print("=" * 80)
    print(f"ANALYZING: {audio_path}")
    print("=" * 80)

    # Get ground truth
    ground_truth = get_ground_truth(audio_path)

    # Get model predictions
    predictions = get_model_predictions(audio_path)

    # Compare
    compare(ground_truth, predictions)

if __name__ == "__main__":
    main()