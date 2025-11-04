# API Reference - Hybrid Hum2Melody v2.0

Complete API documentation for the Hybrid Hum2Melody package.

---

## Main Interface

### ChunkedHybridHum2Melody

**Location**: `inference.hybrid_inference_chunked`

Main class for performing inference on humming audio.

#### Constructor

```python
ChunkedHybridHum2Melody(
    checkpoint_path: str,
    device: str = 'cpu',
    onset_high: float = 0.30,
    onset_low: float = 0.10,
    min_confidence: float = 0.10,
    chunk_duration: float = 15.0,
    overlap: float = 1.0
)
```

**Parameters**:
- `checkpoint_path` (str): Path to combined model checkpoint (.pth file)
- `device` (str, optional): Device to run inference on. Options: 'cpu', 'cuda'. Default: 'cpu'
- `onset_high` (float, optional): High threshold for onset detection. Range: [0.1, 0.5]. Default: 0.30
- `onset_low` (float, optional): Low threshold for onset continuation. Range: [0.05, 0.3]. Default: 0.10
- `min_confidence` (float, optional): Minimum confidence to keep a note. Range: [0.0, 1.0]. Default: 0.10. **Recommended: 0.25**
- `chunk_duration` (float, optional): Duration of each processing chunk in seconds. Default: 15.0
- `overlap` (float, optional): Overlap between chunks in seconds. Default: 1.0

**Example**:
```python
from hybrid_hum2melody import ChunkedHybridHum2Melody

model = ChunkedHybridHum2Melody(
    checkpoint_path='checkpoints/combined_hum2melody_full.pth',
    device='cpu',
    min_confidence=0.25,  # Recommended for production
)
```

#### Methods

##### predict_chunked()

Process audio and return detected notes.

```python
predict_chunked(audio_path: str) -> List[Dict[str, Any]]
```

**Parameters**:
- `audio_path` (str): Path to audio file (WAV, MP3, FLAC, etc.)

**Returns**:
- `List[Dict]`: List of detected notes, each containing:
  - `start` (float): Start time in seconds
  - `end` (float): End time in seconds
  - `duration` (float): Note duration in seconds
  - `midi` (int): MIDI note number (21-108, A0 to C8)
  - `note` (str): Note name (e.g., "C4", "F♯3")
  - `confidence` (float): Model confidence [0.0, 1.0]

**Raises**:
- `FileNotFoundError`: If audio file doesn't exist
- `RuntimeError`: If processing fails

**Example**:
```python
notes = model.predict_chunked('my_humming.wav')

for note in notes:
    print(f"{note['note']} at {note['start']:.2f}s "
          f"(conf: {note['confidence']:.3f})")
```

##### preprocess_audio()

Preprocess audio for model input (used internally).

```python
preprocess_audio(audio_path: str, target_frames: int = 500)
    -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters**:
- `audio_path` (str): Path to audio file
- `target_frames` (int, optional): Number of frames to pad/truncate to. Default: 500

**Returns**:
- `Tuple[torch.Tensor, torch.Tensor]`: (cqt_tensor, extras_tensor)

##### get_pitch_predictions()

Get pitch predictions from model (used internally).

```python
get_pitch_predictions(cqt: torch.Tensor, extras: torch.Tensor)
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Parameters**:
- `cqt` (torch.Tensor): CQT features
- `extras` (torch.Tensor): Extra features

**Returns**:
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: (frame_probs, onset_probs, voicing)

---

## Onset Detection

### detect_onsets_offsets()

**Location**: `data.onset_offset_detector`

Multi-band spectral flux onset detector.

```python
detect_onsets_offsets(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512,
    onset_high: float = 0.30,
    onset_low: float = 0.10,
    min_duration: float = 0.05
) -> List[Tuple[float, float]]
```

**Parameters**:
- `audio` (np.ndarray): Audio samples
- `sr` (int, optional): Sample rate. Default: 16000
- `hop_length` (int, optional): Hop length for STFT. Default: 512
- `onset_high` (float, optional): High threshold. Default: 0.30
- `onset_low` (float, optional): Low threshold. Default: 0.10
- `min_duration` (float, optional): Minimum segment duration in seconds. Default: 0.05

**Returns**:
- `List[Tuple[float, float]]`: List of (start_time, end_time) tuples

**Example**:
```python
from data.onset_offset_detector import detect_onsets_offsets
import librosa

audio, sr = librosa.load('audio.wav', sr=16000, mono=True)
segments = detect_onsets_offsets(
    audio, sr=sr,
    onset_high=0.30,
    onset_low=0.10
)

for start, end in segments:
    print(f"Note segment: {start:.2f}s - {end:.2f}s")
```

---

## Model Architecture

### CombinedHum2MelodyModel

**Location**: `models.combined_model`

Combined pitch and onset model.

```python
CombinedHum2MelodyModel(
    n_bins: int = 88,
    hidden_size: int = 256,
    num_notes: int = 88,
    dropout: float = 0.3,
    use_attention: bool = True,
    use_multi_scale: bool = False,
    use_transition_model: bool = False
)
```

**Parameters**:
- `n_bins` (int): Number of CQT bins. Default: 88
- `hidden_size` (int): Hidden layer size. Default: 256
- `num_notes` (int): Number of note classes. Default: 88
- `dropout` (float): Dropout rate. Default: 0.3
- `use_attention` (bool): Use attention mechanism. Default: True
- `use_multi_scale` (bool): Use multi-scale processing. Default: False
- `use_transition_model` (bool): Use transition model. Default: False

**Methods**:
- `forward(cqt, extras)`: Forward pass
- `get_pitch_predictions(cqt, extras)`: Get pitch predictions
- `get_onset_predictions(cqt, extras)`: Get onset predictions (not used in hybrid)

---

## Model Loading

### load_combined_model()

**Location**: `models.combined_model_loader`

Load combined model from checkpoint.

```python
load_combined_model(
    checkpoint_path: str,
    device: str = 'cpu',
    strict: bool = False
) -> CombinedHum2MelodyModel
```

**Parameters**:
- `checkpoint_path` (str): Path to .pth checkpoint
- `device` (str, optional): Device to load on. Default: 'cpu'
- `strict` (bool, optional): Strict state dict loading. Default: False

**Returns**:
- `CombinedHum2MelodyModel`: Loaded model in eval mode

**Example**:
```python
from models.combined_model_loader import load_combined_model

model = load_combined_model(
    'checkpoints/combined_hum2melody_full.pth',
    device='cpu'
)
```

---

## Evaluation

### compare_to_audio()

**Location**: `scripts.analyze_results`

Compare predictions to actual audio content using CQT.

```python
compare_to_audio(notes: List[Dict], audio_path: str) -> Dict[str, Any]
```

**Parameters**:
- `notes` (List[Dict]): Predicted notes
- `audio_path` (str): Path to audio file

**Returns**:
- `Dict[str, Any]`: Accuracy metrics
  - `exact_matches` (int): Number of exact matches
  - `within_1` (int): Number within 1 semitone
  - `within_2` (int): Number within 2 semitones
  - `exact_accuracy` (float): Exact match percentage
  - `within_1_accuracy` (float): Within 1 semitone percentage
  - `within_2_accuracy` (float): Within 2 semitone percentage
  - `mismatches` (List[Dict]): Details of errors

**Example**:
```python
from scripts.analyze_results import compare_to_audio

notes = model.predict_chunked('audio.wav')
metrics = compare_to_audio(notes, 'audio.wav')

print(f"Exact accuracy: {metrics['exact_accuracy']:.1f}%")
print(f"Within 1 ST: {metrics['within_1_accuracy']:.1f}%")
```

---

## Utilities

### midi_to_note_name()

Convert MIDI number to note name.

```python
def midi_to_note_name(midi: int) -> str:
    """
    Convert MIDI number to note name with octave.

    Args:
        midi: MIDI note number (21-108)

    Returns:
        Note name (e.g., "C4", "F♯3", "A0")
    """
    note_names = ['C', 'C♯', 'D', 'D♯', 'E', 'F',
                  'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"
```

### hz_to_midi()

Convert frequency to MIDI number.

```python
def hz_to_midi(freq: float) -> int:
    """
    Convert frequency in Hz to MIDI note number.

    Args:
        freq: Frequency in Hz

    Returns:
        MIDI note number (rounded)
    """
    import numpy as np
    return int(round(12 * np.log2(freq / 440.0) + 69))
```

### midi_to_hz()

Convert MIDI number to frequency.

```python
def midi_to_hz(midi: int) -> float:
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        midi: MIDI note number

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))
```

---

## Command Line Interface

### test_my_humming.py

**Location**: `scripts/test_my_humming.py`

Command-line interface for inference.

```bash
python scripts/test_my_humming.py [options] audio_file [audio_file ...]
```

**Options**:
- `--checkpoint PATH`: Path to checkpoint (default: hum2melody_package/checkpoints/combined_hum2melody_full.pth)
- `--device {cpu,cuda}`: Device to use (default: cpu)
- `--onset-high FLOAT`: Onset high threshold (default: 0.30)
- `--onset-low FLOAT`: Onset low threshold (default: 0.10)
- `--min-confidence FLOAT`: Minimum confidence (default: 0.10)
- `--visualize`: Create visualization
- `--save-json`: Save results to JSON

**Example**:
```bash
python scripts/test_my_humming.py my_hum.wav \
    --min-confidence 0.25 \
    --visualize \
    --save-json
```

### analyze_results.py

**Location**: `scripts/analyze_results.py`

Analyze prediction quality and accuracy.

```bash
python scripts/analyze_results.py
```

Automatically finds `*_notes.json` files and analyzes them against audio.

---

## Data Structures

### Note Dictionary

Output format for predicted notes:

```python
{
    'start': float,        # Start time in seconds
    'end': float,          # End time in seconds
    'duration': float,     # Duration in seconds
    'midi': int,           # MIDI note number (21-108)
    'note': str,           # Note name (e.g., "C4")
    'confidence': float    # Model confidence [0.0, 1.0]
}
```

### Segment Tuple

Format for onset detection output:

```python
(start_time: float, end_time: float)
```

### Accuracy Metrics

Format returned by evaluation functions:

```python
{
    'total_valid': int,
    'exact_matches': int,
    'within_1': int,
    'within_2': int,
    'exact_accuracy': float,
    'within_1_accuracy': float,
    'within_2_accuracy': float,
    'mismatches': List[Dict]
}
```

---

## Configuration Presets

### Production (Recommended)

```python
model = ChunkedHybridHum2Melody(
    checkpoint_path='checkpoints/combined_hum2melody_full.pth',
    min_confidence=0.25,  # Balance accuracy and coverage
    onset_high=0.30,
    onset_low=0.10,
    device='cpu'
)
```

**Expected**: 85% accuracy, 80% coverage

### High Recall

```python
model = ChunkedHybridHum2Melody(
    checkpoint_path='checkpoints/combined_hum2melody_full.pth',
    min_confidence=0.15,  # Keep more notes
    onset_high=0.25,      # More sensitive
    onset_low=0.08,
    device='cpu'
)
```

**Expected**: 78% accuracy, 90% coverage

### High Precision

```python
model = ChunkedHybridHum2Melody(
    checkpoint_path='checkpoints/combined_hum2melody_full.pth',
    min_confidence=0.40,  # Only confident predictions
    onset_high=0.40,      # More conservative
    onset_low=0.15,
    device='cpu'
)
```

**Expected**: 90% accuracy, 60% coverage

---

## Error Handling

### Common Exceptions

**FileNotFoundError**: Audio file or checkpoint not found
```python
try:
    notes = model.predict_chunked('nonexistent.wav')
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

**RuntimeError**: Model inference failed
```python
try:
    notes = model.predict_chunked('corrupted.wav')
except RuntimeError as e:
    print(f"Inference failed: {e}")
```

**ValueError**: Invalid parameters
```python
try:
    model = ChunkedHybridHum2Melody(
        checkpoint_path='model.pth',
        min_confidence=2.0  # Invalid: must be [0, 1]
    )
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

---

## Performance Considerations

### Memory Usage

| Audio Length | Peak Memory (CPU) |
|--------------|-------------------|
| 10s | ~500 MB |
| 30s | ~620 MB |
| 60s | ~750 MB |

**Tip**: For very long audio (>2 minutes), consider processing in separate batches.

### Speed

| Audio Length | Processing Time (CPU) | Realtime Factor |
|--------------|----------------------|------------------|
| 10s | ~5s | 0.5x (2x realtime) |
| 30s | ~15s | 0.5x |
| 60s | ~30s | 0.5x |

**Tip**: Use GPU (`device='cuda'`) for ~5-10x speedup if available.

### Batching

The system processes audio in chunks automatically. For batch processing multiple files:

```python
import glob

model = ChunkedHybridHum2Melody('checkpoints/combined_hum2melody_full.pth')

for audio_file in glob.glob('humming_samples/*.wav'):
    notes = model.predict_chunked(audio_file)
    # Process notes...
```

---

## Version Compatibility

### Model Versions

- **v1.0**: Original combined model (onset F1=32%)
- **v2.0**: Hybrid system (onset precision=88%) ✅ Current

### Checkpoint Compatibility

The package is compatible with checkpoints from:
- Combined model v1.0 (135MB, 35M params)
- Stage1 pitch model (15M params) - partial functionality

---

## Advanced Usage

### Custom Onset Detection

You can use your own onset detector:

```python
from data.onset_offset_detector import detect_onsets_offsets
import librosa

# Load audio
audio, sr = librosa.load('audio.wav', sr=16000, mono=True)

# Custom onset detection
segments = detect_onsets_offsets(
    audio, sr=sr,
    onset_high=0.35,  # Custom threshold
    onset_low=0.12
)

# Then use segments for pitch prediction...
```

### Post-Processing

Filter results after inference:

```python
notes = model.predict_chunked('audio.wav')

# Filter by confidence
confident_notes = [n for n in notes if n['confidence'] >= 0.5]

# Filter by duration
long_notes = [n for n in notes if n['duration'] >= 0.2]

# Merge adjacent same-pitch notes
def merge_notes(notes, pitch_tolerance=0):
    merged = []
    current = None
    for note in sorted(notes, key=lambda x: x['start']):
        if current is None:
            current = note
        elif (abs(note['midi'] - current['midi']) <= pitch_tolerance and
              note['start'] - current['end'] < 0.1):
            # Merge
            current['end'] = note['end']
            current['duration'] = current['end'] - current['start']
        else:
            merged.append(current)
            current = note
    if current:
        merged.append(current)
    return merged

smoothed_notes = merge_notes(notes)
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

**API Version**: 2.0
**Last Updated**: November 3, 2025
**Compatible with**: Python 3.8+, PyTorch 1.10+
