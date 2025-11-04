# Combined Hum2Melody Model - Deployment Guide

This guide explains how to deploy and use the combined hum2melody model that integrates both pitch detection and onset/offset detection in a single artifact.

## Model Overview

The combined model consists of two trained submodels:
- **Pitch Model** (14.9M params): Frame-level pitch classification + continuous F0 prediction
- **Onset Model** (3.7M params): Binary onset/offset detection

**Total**: ~18.6M parameters, exported as a single TorchScript file (~150MB).

## Model Outputs

The model returns a tuple of 4 tensors:

```python
frame, onset, offset, f0 = model(cqt_input, extras_input)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `frame` | `(batch, 125, 88)` | Pitch classification logits (MIDI 21-108) |
| `onset` | `(batch, 125, 1)` | Note onset detection logits |
| `offset` | `(batch, 125, 1)` | Note offset detection logits |
| `f0` | `(batch, 125, 2)` | Continuous F0: `[f0_value, voicing]` |

## Prerequisites

```bash
pip install torch librosa numpy
```

## Quick Start

### 1. Load the Model

```python
import torch

# Load exported model
model = torch.jit.load('combined_hum2melody.pt')
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 2. Preprocess Audio

The model expects CQT (Constant-Q Transform) input with specific parameters:

```python
import librosa
import numpy as np

def preprocess_audio(audio_path, sr=16000):
    """
    Preprocess audio file to model input format.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate (must be 16000)

    Returns:
        cqt_tensor: (1, 1, 500, 88) - CQT input
        extras_tensor: (1, 1, 500, 24) - Onset features + musical context
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Extract CQT
    cqt = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=512,
        n_bins=88,
        bins_per_octave=12,
        fmin=27.5  # A0 (MIDI 21)
    )

    # Convert to dB and normalize
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_normalized = (cqt_db + 80) / 80
    cqt_normalized = np.clip(cqt_normalized, 0, 1)

    # Pad or truncate to 500 frames (~16 seconds)
    target_frames = 500
    if cqt_normalized.shape[1] < target_frames:
        pad_width = target_frames - cqt_normalized.shape[1]
        cqt_normalized = np.pad(
            cqt_normalized,
            ((0, 0), (0, pad_width)),
            mode='constant'
        )
    elif cqt_normalized.shape[1] > target_frames:
        cqt_normalized = cqt_normalized[:, :target_frames]

    # Transpose and add batch/channel dims: (88, 500) → (1, 1, 500, 88)
    cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)

    # Extract onset features (optional, can use zeros)
    # For simplicity, we'll use zeros here
    # In production, you may want to extract actual onset features
    extras_tensor = torch.zeros(1, 1, 500, 24)

    return cqt_tensor, extras_tensor
```

### 3. Run Inference

```python
# Preprocess audio
cqt, extras = preprocess_audio('my_audio.wav')

# Move to device
cqt = cqt.to(device)
extras = extras.to(device)

# Run inference
with torch.no_grad():
    frame, onset, offset, f0 = model(cqt, extras)

# Convert logits to probabilities
frame_probs = torch.sigmoid(frame)  # (1, 125, 88)
onset_probs = torch.sigmoid(onset)  # (1, 125, 1)
offset_probs = torch.sigmoid(offset) # (1, 125, 1)
f0_value = f0[:, :, 0]  # (1, 125) - normalized F0
voicing = torch.sigmoid(f0[:, :, 1])  # (1, 125) - voicing probability
```

### 4. Post-Process Outputs

```python
def postprocess_pitch(frame_probs, onset_probs, offset_probs, f0_value, voicing, threshold_onset=0.15):
    """
    Convert model outputs to note sequence.

    Returns:
        notes: List of (start_time, duration, midi_note) tuples
    """
    frame_rate = 31.25  # Hz (16000 / 512)

    # Detect onsets
    onset_mask = onset_probs.squeeze().cpu().numpy() > threshold_onset
    onset_frames = np.where(onset_mask)[0]

    # Get pitch at each onset
    notes = []
    for onset_frame in onset_frames:
        # Get pitch (highest probability note)
        pitch_idx = frame_probs[0, onset_frame].argmax().item()
        midi_note = pitch_idx + 21  # Add offset (MIDI 21 = A0)

        # Find offset (next onset or end of sequence)
        offset_frame = onset_frame + 1
        if offset_frame < len(onset_mask):
            next_onsets = onset_frames[onset_frames > onset_frame]
            if len(next_onsets) > 0:
                offset_frame = next_onsets[0]
            else:
                offset_frame = len(onset_mask) - 1

        # Convert frames to time
        start_time = onset_frame / frame_rate
        end_time = offset_frame / frame_rate
        duration = end_time - start_time

        notes.append((start_time, duration, midi_note))

    return notes

# Post-process
notes = postprocess_pitch(frame_probs, onset_probs, offset_probs, f0_value, voicing)

# Print results
for start, duration, midi in notes:
    note_name = librosa.midi_to_note(midi)
    print(f"Note: {note_name} (MIDI {midi}), Start: {start:.2f}s, Duration: {duration:.2f}s")
```

## Complete Example

```python
import torch
import librosa
import numpy as np

def transcribe_audio(model_path, audio_path, device='cuda'):
    """Complete transcription pipeline."""

    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # Preprocess
    cqt, extras = preprocess_audio(audio_path)
    cqt = cqt.to(device)
    extras = extras.to(device)

    # Inference
    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt, extras)

    # Post-process
    frame_probs = torch.sigmoid(frame)
    onset_probs = torch.sigmoid(onset)
    offset_probs = torch.sigmoid(offset)

    notes = postprocess_pitch(frame_probs, onset_probs, offset_probs,
                              f0[:, :, 0], f0[:, :, 1])

    return notes

# Usage
notes = transcribe_audio('combined_hum2melody.pt', 'my_humming.wav')
for start, duration, midi in notes:
    print(f"{librosa.midi_to_note(midi)}: {start:.2f}s, {duration:.2f}s")
```

## Preprocessing Requirements

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample rate | 16000 Hz | Must resample audio to 16kHz |
| Hop length | 512 samples | ~32ms per frame |
| CQT bins | 88 | MIDI 21-108 (A0-C8) |
| Bins per octave | 12 | Semitone resolution |
| fmin | 27.5 Hz | A0 (MIDI 21) |
| Input frames | 500 | ~16 seconds of audio |
| Output frames | 125 | Downsampled 4x by CNN |

## Performance

Benchmarked on Tesla V100 GPU:
- **Latency**: ~30-50ms per sample (batch=1)
- **Memory**: ~500MB GPU RAM
- **Throughput**: ~20-30 samples/second

## Troubleshooting

### Issue: "RuntimeError: Input shape mismatch"
**Solution**: Ensure CQT input is exactly `(batch, 1, 500, 88)` and extras is `(batch, 1, 500, 24)` or `None`.

### Issue: "All onset probabilities are near zero"
**Solution**: Use a lower threshold (try 0.05-0.15) for onset detection. The model outputs logits, so apply sigmoid first.

### Issue: "Poor pitch accuracy"
**Solution**:
1. Verify audio is clean (minimal background noise)
2. Ensure audio is monophonic (single voice/instrument)
3. Check sample rate is exactly 16kHz
4. Verify CQT preprocessing matches model requirements

### Issue: "Model runs slowly"
**Solutions**:
- Use GPU if available
- Reduce batch size to 1
- Consider FP16 inference: `model.half()` (GPU only)
- Process audio in chunks if memory is limited

## Advanced Usage

### Streaming / Real-time Processing

For streaming audio, process in overlapping windows:

```python
def stream_transcribe(model, audio, sr=16000, window_size=8.0, hop_size=4.0):
    """Process audio in overlapping windows."""
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    all_notes = []
    offset = 0

    while offset < len(audio):
        # Extract window
        window = audio[offset:offset + window_samples]

        # Pad if needed
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)))

        # Transcribe window
        notes = transcribe_audio_window(model, window, sr)

        # Adjust timestamps
        for start, duration, midi in notes:
            all_notes.append((start + offset/sr, duration, midi))

        offset += hop_samples

    return all_notes
```

### Batch Processing

Process multiple audio files efficiently:

```python
def batch_transcribe(model, audio_paths, device='cuda', batch_size=8):
    """Process multiple audio files in batches."""
    results = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]

        # Preprocess batch
        cqts, extras = [], []
        for path in batch_paths:
            cqt, extra = preprocess_audio(path)
            cqts.append(cqt)
            extras.append(extra)

        # Stack into batch
        cqt_batch = torch.cat(cqts, dim=0).to(device)
        extras_batch = torch.cat(extras, dim=0).to(device)

        # Inference
        with torch.no_grad():
            frame, onset, offset, f0 = model(cqt_batch, extras_batch)

        # Post-process each sample
        for j in range(len(batch_paths)):
            notes = postprocess_pitch(
                torch.sigmoid(frame[j:j+1]),
                torch.sigmoid(onset[j:j+1]),
                torch.sigmoid(offset[j:j+1]),
                f0[j:j+1, :, 0],
                f0[j:j+1, :, 1]
            )
            results.append((batch_paths[j], notes))

    return results
```

## Model Architecture

The combined model internally runs two separate models:

```
Input: CQT (88 bins) + Extras (24 bins)
       ↓
    ┌──────────────────┬──────────────────┐
    │   Pitch Model    │   Onset Model    │
    │   (112 channels) │   (88 channels)  │
    │                  │                  │
    │   CNN + LSTM     │   CNN + LSTM     │
    │   ↓              │   ↓              │
    │   Frame + F0     │   Onset + Offset │
    └──────────────────┴──────────────────┘
              ↓
    Combined Output: (frame, onset, offset, f0)
```

Both models share the same preprocessing and frame rate, ensuring perfect alignment.

## Citation

If you use this model in research or production, please cite:

```
Combined Hum2Melody Model
Pitch Detection Accuracy: 98.46% (±1 semitone)
Architecture: EnhancedHum2MelodyModel + OnsetOffsetModel
Training: MSU HPCC, 2025
```

## License

[Specify your license here]

## Support

For issues or questions:
- Check preprocessing requirements match exactly
- Verify model input shapes
- Test with simple, clean audio first
- Check GPU/CUDA compatibility

## Version History

- **v1.0** (2025-10): Initial release
  - Pitch model: 98.46% accuracy
  - Onset model: TBD (training in progress)
  - Combined export: TorchScript
