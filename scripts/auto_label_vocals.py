# Updated script using librosa's PYIN (faster, no TensorFlow needed)
import librosa
import numpy as np
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import argparse


def process_single_file(args):
    """Process a single vocal file and extract melody labels"""
    audio_path, output_dir = args

    try:
        # Load audio at 16kHz mono
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Use librosa's pYIN pitch detection (much faster than CREPE)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=sr,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            frame_length=2048
        )

        # Get time stamps for each frame
        times = librosa.times_like(f0, sr=sr)

        # Filter out unvoiced frames and NaN values
        voiced_mask = ~np.isnan(f0) & (voiced_probs > 0.5)
        reliable_freq = f0[voiced_mask]
        reliable_time = times[voiced_mask]
        reliable_conf = voiced_probs[voiced_mask]

        if len(reliable_freq) == 0:
            return audio_path.stem, "no_reliable_pitch"

        # Convert frequency to MIDI
        midi_notes = librosa.hz_to_midi(reliable_freq)
        midi_rounded = np.round(midi_notes)

        # Segment into discrete notes
        notes = []
        start_times = []
        durations = []
        confidences = []

        current_note = midi_rounded[0]
        current_start = reliable_time[0]
        current_conf_sum = reliable_conf[0]
        current_conf_count = 1

        for i in range(1, len(midi_rounded)):
            if abs(midi_rounded[i] - current_note) > 0.5:
                duration = reliable_time[i - 1] - current_start
                if i < len(reliable_time) - 1:
                    duration += (reliable_time[1] - reliable_time[0])

                if duration >= 0.1:
                    notes.append(int(current_note))
                    start_times.append(float(current_start))
                    durations.append(float(duration))
                    confidences.append(float(current_conf_sum / current_conf_count))

                current_note = midi_rounded[i]
                current_start = reliable_time[i]
                current_conf_sum = reliable_conf[i]
                current_conf_count = 1
            else:
                current_conf_sum += reliable_conf[i]
                current_conf_count += 1

        # Last note
        if len(reliable_time) > 1:
            duration = reliable_time[-1] - current_start + (reliable_time[1] - reliable_time[0])
            if duration >= 0.1:
                notes.append(int(current_note))
                start_times.append(float(current_start))
                durations.append(float(duration))
                confidences.append(float(current_conf_sum / current_conf_count))

        # Create output
        label_data = {
            'audio_path': str(audio_path),
            'notes': notes,
            'start_times': start_times,
            'durations': durations,
            'confidence': float(np.mean(confidences)) if confidences else 0.0
        }

        # Save to JSON
        output_path = output_dir / f"{audio_path.stem}_label.json"
        with open(output_path, 'w') as f:
            json.dump(label_data, f, indent=2)

        return audio_path.stem, "success"

    except Exception as e:
        return audio_path.stem, f"error: {str(e)}"


# [Rest of main() function stays the same]


def main():
    parser = argparse.ArgumentParser(description='Auto-label vocal stems with CREPE')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with vocal files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for label outputs')
    parser.add_argument('--num_files', type=int, default=1000, help='Number of files to process')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of vocal files
    vocal_files = sorted(list(input_dir.glob('*.wav')))[:args.num_files]

    print(f"Found {len(vocal_files)} vocal files to process")
    print(f"Using {args.workers} parallel workers")
    print(f"Output directory: {output_dir}")

    # Prepare arguments for multiprocessing
    process_args = [(f, output_dir) for f in vocal_files]

    # Process with progress bar
    failed_files = []
    successful = 0

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(process_args),
            desc="Processing files"
        ))

    # Analyze results
    for filename, status in results:
        if status == "success":
            successful += 1
        else:
            failed_files.append((filename, status))

    # Save failure log
    if failed_files:
        with open(output_dir / 'failed_files.txt', 'w') as f:
            for filename, error in failed_files:
                f.write(f"{filename}: {error}\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}/{len(vocal_files)}")
    print(f"Failed: {len(failed_files)}/{len(vocal_files)}")
    if failed_files:
        print(f"Failed files logged to: {output_dir / 'failed_files.txt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()