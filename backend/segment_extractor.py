"""
Segment Extractor - Extract visualization data from hum2melody

Provides detailed segment data for interactive tuning UI:
- Detected onset/offset times and confidences
- Note segments with pitches
- Waveform data for visualization
"""

import numpy as np
import librosa
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


def extract_segments_with_detection(
    audio_path: str,
    onset_high: float = 0.30,
    onset_low: float = 0.10,
    offset_high: float = 0.30,
    offset_low: float = 0.10,
    min_confidence: float = 0.25,
    manual_onsets: Optional[List[float]] = None,
    manual_offsets: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Extract pitch predictions for onset/offset segments.

    If manual_onsets and manual_offsets are provided:
      - Uses those exact markers (no automatic detection)
      - Parameters are ignored
      - Lower confidence threshold for accepting notes

    If manual markers NOT provided:
      - Runs automatic amplitude-based detection with parameters
      - Uses normal confidence threshold

    Args:
        audio_path: Path to audio file
        onset_high: High threshold for automatic onset detection (ignored if manual)
        onset_low: Low threshold for automatic onset detection (ignored if manual)
        offset_high: High threshold for automatic offset detection (ignored if manual)
        offset_low: Low threshold for automatic offset detection (ignored if manual)
        min_confidence: Minimum confidence to keep a note (ignored if manual)
        manual_onsets: User-specified onset times (skips automatic detection)
        manual_offsets: User-specified offset times (skips automatic detection)

    Returns:
        Dictionary with segments, onsets, offsets, and parameters
    """
    from backend.hum2melody.data.amplitude_onset_offset_detector import detect_onsets_offsets_amplitude
    from backend.hum2melody.models.combined_model_loader import load_combined_model
    import torch

    print(f"[SegmentExtractor] Extracting segments from: {audio_path}")

    # Determine mode
    if manual_onsets is not None and manual_offsets is not None:
        print(f"[SegmentExtractor] USER MODE: Using {len(manual_onsets)} manual onsets, {len(manual_offsets)} manual offsets")
    else:
        print(f"[SegmentExtractor] AUTO MODE: onset_high={onset_high}, offset_high={offset_high}, min_conf={min_confidence}")

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    audio = audio / np.max(np.abs(audio) + 1e-8)  # Normalize

    print(f"  Duration: {duration:.2f}s")

    # 1. Run onset/offset detection (or use manual markers)
    if manual_onsets is not None and manual_offsets is not None:
        print(f"  Using manual markers: {len(manual_onsets)} onsets, {len(manual_offsets)} offsets")

        # Pair onsets with offsets to create segments
        onset_times = sorted(manual_onsets)
        offset_times = sorted(manual_offsets)

        # Match onsets to offsets (each onset pairs with next unused offset)
        segments_tuples = []
        used_offsets = set()

        for onset in onset_times:
            # Find the next offset after this onset that hasn't been used
            best_offset = None
            for offset in offset_times:
                if offset > onset and offset not in used_offsets:
                    best_offset = offset
                    break

            if best_offset is not None:
                segments_tuples.append((onset, best_offset))
                used_offsets.add(best_offset)
            else:
                print(f"  WARNING: No matching offset for onset at {onset:.3f}s")

        print(f"  Created {len(segments_tuples)} segments from manual markers")
    else:
        # Run automatic detection
        segments_tuples, onset_times, offset_times = detect_onsets_offsets_amplitude(
            audio,
            sr=sr,
            hop_length=512,
            onset_high=onset_high,
            offset_high=offset_high,
            min_note_len=0.05,
            spectral_weight=0.2  # 80% amplitude, 20% spectral
        )

        print(f"  Detected {len(segments_tuples)} segments automatically")

    # 2. Load model and get pitch predictions
    checkpoint_path = Path("hum2melody/checkpoints/combined_hum2melody_full.pth")

    if not checkpoint_path.exists():
        print(f"  [WARNING] Checkpoint not found at {checkpoint_path}, returning segments without pitch")
        return {
            'segments': [{'start': s, 'end': e, 'duration': e - s, 'pitch': 60, 'confidence': 0.5, 'note_name': 'C4'}
                        for s, e in segments_tuples],
            'onsets': [{'time': t, 'confidence': 1.0} for t in onset_times],
            'offsets': [{'time': t, 'confidence': 1.0} for t in offset_times],
            'duration': float(duration),
            'sample_rate': sr,
            'parameters': {
                'onset_high': onset_high,
                'onset_low': onset_low,
                'offset_high': offset_high,
                'offset_low': offset_low,
                'min_confidence': min_confidence
            }
        }

    # Load model
    try:
        model = load_combined_model(str(checkpoint_path), device='cpu')
        model.eval()
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return {
            'segments': [{'start': s, 'end': e, 'duration': e - s, 'pitch': 60, 'confidence': 0.5, 'note_name': 'C4'}
                        for s, e in segments_tuples],
            'onsets': [{'time': t, 'confidence': 1.0} for t in onset_times],
            'offsets': [{'time': t, 'confidence': 1.0} for t in offset_times],
            'duration': float(duration),
            'sample_rate': sr,
            'parameters': {
                'onset_high': onset_high,
                'onset_low': onset_low,
                'offset_high': offset_high,
                'offset_low': offset_low,
                'min_confidence': min_confidence
            }
        }

    # Extract CQT
    cqt = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=512,
        n_bins=88,
        bins_per_octave=12,
        fmin=27.5
    )

    # Normalize CQT
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_normalized = (cqt_db + 80) / 80
    cqt_normalized = np.clip(cqt_normalized, 0, 1)

    # Pad or truncate to 500 frames
    target_frames = 500
    if cqt_normalized.shape[1] < target_frames:
        pad = target_frames - cqt_normalized.shape[1]
        cqt_normalized = np.pad(cqt_normalized, ((0, 0), (0, pad)))
    else:
        cqt_normalized = cqt_normalized[:, :target_frames]

    # Convert to tensors
    cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)
    extras_tensor = torch.zeros(1, 1, target_frames, 24)

    # Run model
    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt_tensor, extras_tensor)

    # Convert to probabilities
    frame_probs = torch.sigmoid(frame)[0].cpu().numpy()  # (125, 88)
    voicing = torch.sigmoid(f0[:, :, 1])[0].cpu().numpy()  # (125,)

    output_frame_rate = 7.8125  # fps (31.25 / 4)

    # 3. Extract pitch for each segment
    detected_notes = []

    for start_time, end_time in segments_tuples:
        # Convert to frame indices
        start_frame = int(start_time * output_frame_rate)
        end_frame = int(end_time * output_frame_rate)

        # Clip to valid range
        start_frame = max(0, min(start_frame, len(frame_probs) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(frame_probs)))

        # Get frames in segment
        segment_probs = frame_probs[start_frame:end_frame]
        segment_voicing = voicing[start_frame:end_frame]

        if len(segment_probs) == 0:
            continue

        # Weight by voicing
        weighted_probs = segment_probs * segment_voicing[:, np.newaxis]

        # Average over time
        avg_probs = weighted_probs.mean(axis=0)

        # Get pitch with highest probability
        pitch_idx = avg_probs.argmax()
        confidence = float(avg_probs[pitch_idx])

        # Convert to MIDI
        midi_note = pitch_idx + 21

        # If manual markers were provided, ALWAYS include the note (user placed it explicitly)
        # If automatic detection, filter by confidence
        if manual_onsets is not None:
            # User placed these markers - trust them completely
            detected_notes.append({
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(end_time - start_time),
                'pitch': int(midi_note),
                'confidence': confidence,
                'note_name': librosa.midi_to_note(midi_note)
            })
        elif confidence >= min_confidence:
            # Automatic detection - filter by confidence
            detected_notes.append({
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(end_time - start_time),
                'pitch': int(midi_note),
                'confidence': confidence,
                'note_name': librosa.midi_to_note(midi_note)
            })
        else:
            print(f"    Segment {start_time:.3f}-{end_time:.3f}s: confidence {confidence:.3f} below threshold {min_confidence:.3f}, skipping")

    if manual_onsets is not None:
        print(f"  Extracted {len(detected_notes)} notes (manual markers - no confidence filter)")
    else:
        print(f"  Extracted {len(detected_notes)} notes (filtered by confidence >= {min_confidence})")

    # 4. Build result
    result = {
        'segments': detected_notes,
        'onsets': [{'time': float(t), 'confidence': 1.0} for t in onset_times],
        'offsets': [{'time': float(t), 'confidence': 1.0} for t in offset_times],
        'duration': float(duration),
        'sample_rate': sr,
        'parameters': {
            'onset_high': onset_high,
            'onset_low': onset_low,
            'offset_high': offset_high,
            'offset_low': offset_low,
            'min_confidence': min_confidence
        }
    }

    return result
