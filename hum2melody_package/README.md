# Hum2Melody Model Package

A complete deep learning package for audio-to-melody transcription, combining pitch detection and onset/offset detection in a single model.

## Overview

This package contains everything needed to use, train, and evaluate the Hum2Melody model, which performs:
- **Pitch Detection**: Frame-level pitch classification (MIDI 21-108) + continuous F0 prediction
- **Onset/Offset Detection**: Binary detection of note boundaries
- **Combined Output**: Synchronized predictions for complete melody transcription

### Model Architecture

- **Pitch Model** (EnhancedHum2MelodyModel): 14.9M parameters
- **Onset Model** (EnhancedOnsetOffsetModel): 3.7M parameters
- **Total**: ~18.6M parameters
- **Checkpoint Size**: 135 MB

### Performance Metrics

| Metric | Value |
|--------|-------|
| Pitch Accuracy (±1 semitone) | 98.46% |
| Frame F1 | 0.837 |
| Onset F1 | 0.321 |
| Offset F1 | 0.310 |

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import librosa; print('✓ Dependencies installed')"
```

### 2. Simple Inference

```python
import sys
sys.path.insert(0, 'hum2melody_package')

import torch
from models import load_combined_model

# Load model
model = load_combined_model('checkpoints/combined_hum2melody_full.pth', device='cuda')
model.eval()

# Preprocess audio (see examples/simple_inference.py for full implementation)
# cqt, extras = preprocess_audio('my_audio.wav')

# Run inference
with torch.no_grad():
    frame, onset, offset, f0 = model(cqt, extras)

# Convert to probabilities
frame_probs = torch.sigmoid(frame)
onset_probs = torch.sigmoid(onset)
```

### 3. Command-Line Inference

```bash
cd examples
python simple_inference.py --audio path/to/audio.wav --checkpoint ../checkpoints/combined_hum2melody_full.pth
```

## Package Structure

```
hum2melody_package/
├── models/                 # Model architectures
│   ├── combined_model_loader.py   # Load single combined checkpoint (recommended)
│   ├── combined_model.py          # Load from separate checkpoints
│   ├── hum2melody_model.py        # Pitch detection model
│   ├── onset_model.py             # Basic onset/offset model
│   ├── enhanced_onset_model.py    # Enhanced onset/offset model
│   ├── pretrained_features.py     # Feature extraction utilities
│   ├── musical_components.py      # Shared model components
│   └── onset_informed_decoder.py  # Decoder utilities
├── data/                   # Data loading
│   ├── melody_dataset.py          # PyTorch Dataset classes
│   └── synthetic_data_generator.py # Synthetic data generation
├── scripts/                # Training and utilities
│   ├── train_hum2melody.py        # Train pitch model
│   ├── train_onset_model.py       # Train onset model (basic)
│   ├── train_enhanced_onset_model.py  # Train onset model (enhanced)
│   ├── create_combined_checkpoint.py  # Merge checkpoints into single file
│   ├── export_combined_model.py   # Export utilities
│   ├── verify_combined_model.py   # Verification script
│   └── benchmark_combined_model.py # Performance benchmarking
├── evaluation/             # Model evaluation
│   ├── evaluate_combined_model.py     # Standard evaluation
│   └── evaluate_combined_detailed.py  # Detailed analysis
├── examples/               # Usage examples
│   └── simple_inference.py        # Basic inference example
├── checkpoints/            # Trained models
│   ├── combined_hum2melody_full.pth   # Combined model checkpoint (135 MB)
│   └── combined_hum2melody_full.json  # Model metadata
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Usage Examples

### Loading the Model

**Option 1: Combined Checkpoint (Recommended)**
```python
from models import load_combined_model

model = load_combined_model('checkpoints/combined_hum2melody_full.pth', device='cuda')
```

**Option 2: Separate Checkpoints**
```python
from models import CombinedHum2MelodyModel

model = CombinedHum2MelodyModel(
    pitch_ckpt_path='path/to/pitch_model.pth',
    onset_ckpt_path='path/to/onset_model.pth',
    device='cuda'
)
```

### Preprocessing Audio

```python
import librosa
import numpy as np
import torch

def preprocess_audio(audio_path, sr=16000, target_frames=500):
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

    # Normalize to [0, 1]
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    cqt_normalized = (cqt_db + 80) / 80
    cqt_normalized = np.clip(cqt_normalized, 0, 1)

    # Pad/truncate to target_frames
    if cqt_normalized.shape[1] < target_frames:
        pad = target_frames - cqt_normalized.shape[1]
        cqt_normalized = np.pad(cqt_normalized, ((0, 0), (0, pad)))
    else:
        cqt_normalized = cqt_normalized[:, :target_frames]

    # Convert to tensor: (88, 500) → (1, 1, 500, 88)
    cqt_tensor = torch.FloatTensor(cqt_normalized.T).unsqueeze(0).unsqueeze(0)
    extras_tensor = torch.zeros(1, 1, target_frames, 24)

    return cqt_tensor, extras_tensor
```

### Running Inference

```python
# Preprocess
cqt, extras = preprocess_audio('audio.wav')

# Move to device
cqt = cqt.to('cuda')
extras = extras.to('cuda')

# Inference
with torch.no_grad():
    frame, onset, offset, f0 = model(cqt, extras)

# Convert to probabilities
frame_probs = torch.sigmoid(frame)    # (1, 125, 88) - pitch
onset_probs = torch.sigmoid(onset)    # (1, 125, 1) - onsets
offset_probs = torch.sigmoid(offset)  # (1, 125, 1) - offsets
voicing = torch.sigmoid(f0[:, :, 1])  # (1, 125) - voicing
```

### Extracting Notes

```python
def extract_notes(frame_probs, onset_probs, onset_threshold=0.15):
    """Extract note sequence from model outputs."""
    frame_rate = 31.25  # Hz (output frames)

    # Detect onsets
    onset_mask = onset_probs.squeeze().cpu().numpy() > onset_threshold
    onset_frames = np.where(onset_mask)[0]

    notes = []
    for i, onset_frame in enumerate(onset_frames):
        # Get pitch
        pitch_idx = frame_probs[0, onset_frame].argmax().item()
        midi_note = pitch_idx + 21  # Add MIDI offset

        # Get duration (to next onset)
        if i + 1 < len(onset_frames):
            offset_frame = onset_frames[i + 1]
        else:
            offset_frame = len(onset_mask)

        start_time = onset_frame / frame_rate
        duration = (offset_frame - onset_frame) / frame_rate

        notes.append((start_time, duration, midi_note))

    return notes

# Use it
notes = extract_notes(frame_probs, onset_probs)
for start, duration, midi in notes:
    note_name = librosa.midi_to_note(midi)
    print(f"{note_name}: {start:.2f}s, {duration:.2f}s")
```

## Training

### Train Pitch Model

```bash
cd scripts
python train_hum2melody.py \
    --manifest path/to/manifest.json \
    --output-dir ../checkpoints/pitch_model \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

### Train Onset Model

```bash
python train_enhanced_onset_model.py \
    --manifest path/to/manifest.json \
    --output-dir ../checkpoints/onset_model \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

### Create Combined Checkpoint

```bash
python create_combined_checkpoint.py \
    --pitch-ckpt ../checkpoints/pitch_model/best_model.pth \
    --onset-ckpt ../checkpoints/onset_model/best_model.pth \
    --output ../checkpoints/combined_model.pth
```

## Evaluation

### Basic Evaluation

```bash
cd evaluation
python evaluate_combined_model.py \
    --checkpoint ../checkpoints/combined_hum2melody_full.pth \
    --manifest path/to/test_manifest.json \
    --output evaluation_results.json
```

### Detailed Evaluation

```bash
python evaluate_combined_detailed.py \
    --checkpoint ../checkpoints/combined_hum2melody_full.pth \
    --manifest path/to/test_manifest.json \
    --output detailed_results.json
```

## Model Specifications

### Input Requirements

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16000 Hz | Audio must be resampled to 16kHz |
| Hop Length | 512 samples | ~32ms per frame |
| Input Frames | 500 | ~16 seconds of audio |
| CQT Bins | 88 | MIDI range 21-108 (A0-C8) |
| Bins per Octave | 12 | Semitone resolution |
| fmin | 27.5 Hz | A0 (MIDI 21) |

### Output Specifications

| Output | Shape | Description |
|--------|-------|-------------|
| `frame` | (batch, 125, 88) | Pitch classification logits |
| `onset` | (batch, 125, 1) | Onset detection logits |
| `offset` | (batch, 125, 1) | Offset detection logits |
| `f0` | (batch, 125, 2) | Continuous F0: [value, voicing] |

Note: 500 input frames → 125 output frames (4× downsampling by CNN)

### Preprocessing Pipeline

```
Audio File
    ↓
Load at 16kHz (mono)
    ↓
CQT (88 bins, hop=512)
    ↓
Convert to dB
    ↓
Normalize to [0, 1]
    ↓
Pad/Truncate to 500 frames
    ↓
Tensor: (1, 1, 500, 88)
```

## Advanced Topics

### Custom Feature Extraction

The model accepts optional "extras" features (24 channels):
- Onset strength features (from pretrained models)
- Musical context features

For simple inference, you can use zeros:
```python
extras = torch.zeros(1, 1, 500, 24)
```

For better results, extract onset features:
```python
from models.pretrained_features import extract_onset_strength_features

onset_features = extract_onset_strength_features(audio, sr=16000)
# Shape: (frames, 20)
```

### Batch Processing

```python
# Stack multiple audio files
cqt_batch = torch.cat([cqt1, cqt2, cqt3], dim=0)  # (3, 1, 500, 88)
extras_batch = torch.cat([extras1, extras2, extras3], dim=0)

# Single forward pass
with torch.no_grad():
    frame, onset, offset, f0 = model(cqt_batch, extras_batch)
```

### GPU Optimization

```python
# Use mixed precision for faster inference
model = model.half()  # FP16
cqt = cqt.half()
extras = extras.half()

# Enable TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
```

## Troubleshooting

### Issue: "RuntimeError: Input shape mismatch"
**Solution**: Ensure input tensors have correct shapes:
- CQT: `(batch, 1, 500, 88)`
- Extras: `(batch, 1, 500, 24)` or `None`

### Issue: "No notes detected"
**Solution**:
- Lower the onset threshold (try 0.05-0.15)
- Check that audio is clean and monophonic
- Verify audio sample rate is 16kHz
- Ensure CQT preprocessing is correct

### Issue: "Poor pitch accuracy"
**Solution**:
- Verify audio quality (minimal background noise)
- Ensure audio is monophonic (single melody line)
- Check that CQT parameters match model training
- Try adjusting the voicing threshold

### Issue: "CUDA out of memory"
**Solution**:
- Reduce batch size to 1
- Process audio in shorter chunks
- Use CPU inference: `device='cpu'`
- Enable gradient checkpointing (for training)

## Citation

If you use this model in your research or application, please cite:

```
@software{hum2melody2025,
  title={Hum2Melody: Deep Learning for Audio-to-Melody Transcription},
  author={[Your Name]},
  year={2025},
  note={Trained on MSU HPCC}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Acknowledgments

- Trained on Michigan State University High Performance Computing Center (MSU HPCC)
- Model architecture inspired by state-of-the-art music transcription systems
- CQT preprocessing based on librosa

## Support

For issues, questions, or contributions:
1. Check the examples in `examples/`
2. Review the DEPLOYMENT.md guide
3. Open an issue with:
   - Your environment details (Python version, PyTorch version, GPU)
   - Minimal code to reproduce the issue
   - Error messages and stack traces

## Version History

### v1.0 (October 2025)
- Initial release
- Combined model (pitch + onset/offset)
- 98.46% pitch accuracy (±1 semitone)
- Single checkpoint deployment
- Complete training and evaluation pipelines
