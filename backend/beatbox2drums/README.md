# 🥁 Beatbox2Drums Package

**A production-ready beatbox-to-drums classification system for identifying drum types from onset events.**

**Version**: 1.0
**Status**: ✅ Production Ready
**Test Accuracy**: 93.76% (kick: 94.87%, snare: 91.10%, hihat: 95.32%)
**License**: MIT

---

## 🎯 Overview

This package implements a CNN-based classifier for identifying drum types (kick, snare, hi-hat) from beatbox and vocal percussion recordings. Given onset times in an audio file, the model classifies each onset as one of three drum types with high accuracy.

### Key Features

- ✅ **High Accuracy**: 93.76% test accuracy on real beatbox recordings
- ✅ **Single-Label Classification**: One drum type per onset (beatboxers can't make two sounds simultaneously)
- ✅ **Frame-Level Processing**: ±50ms windows around onsets capture transient attack information
- ✅ **Fast Inference**: ~110k parameters, runs efficiently on CPU or GPU
- ✅ **Comprehensive Evaluation**: Visualization tools showing input, ground truth, and predictions aligned

---

## 📦 Package Contents

```
backend/beatbox2drums/
├── checkpoints/                      # Trained model weights
│   └── beatbox2drums_best.pth       (1.3MB)
├── models/                           # Model architectures
│   ├── drum_classifier.py           # CNN architecture
│   ├── model_loader.py              # Model loading utilities
│   └── __init__.py
├── inference/                        # Inference code
│   ├── drum_classifier.py           # Main inference class
│   ├── preprocessing.py             # Audio preprocessing
│   └── __init__.py
├── evaluation/                       # Evaluation tools
│   ├── metrics.py                   # Accuracy metrics
│   ├── visualize.py                 # Visualization plots
│   └── __init__.py
├── examples/                         # Usage examples
│   ├── basic_inference.py           # Simple classification
│   ├── evaluate_with_visualization.py  # Full evaluation with plots
│   └── __init__.py
├── README.md                         # This file
├── requirements.txt                  # Dependencies
└── setup.py                          # Installation script
```

---

## 🚀 Quick Start

### Installation

```bash
# Navigate to the package
cd backend/beatbox2drums

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Basic Usage

```python
from inference import Beatbox2DrumsClassifier

# Initialize classifier
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='checkpoints/beatbox2drums_best.pth',
    device='cpu',  # or 'cuda'
    min_confidence=0.5
)

# Classify onsets
results = classifier.classify_audio_file(
    audio_path='my_beatbox.wav',
    onset_times=[0.5, 1.0, 1.5, 2.0]  # seconds
)

# Print results
for r in results:
    print(f"{r['onset_time']:.2f}s: {r['drum_type']} ({r['confidence']:.0%})")
```

### Command Line

```bash
# Basic inference
python examples/basic_inference.py audio.wav 0.5 1.0 1.5

# Evaluation with visualization
python examples/evaluate_with_visualization.py audio.wav labels.json --output results.png
```

---

## 📊 Performance Metrics

### Test Results

| Metric | Value |
|--------|-------|
| **Overall Test Accuracy** | 93.76% |
| **Kick Accuracy** | 94.87% |
| **Snare Accuracy** | 91.10% |
| **Hi-hat Accuracy** | 95.32% |
| **Best Validation Accuracy** | 94.38% |

### Training Details

- **Dataset**: BaDumTss (145k onsets) + AVP (9.8k onsets) = 154,950 total onsets
- **Model**: CNN with 110,019 parameters
- **Input**: (1, 128, 16) mel spectrograms from ±50ms windows
- **Training Time**: ~34 minutes (99 epochs with early stopping)
- **Hardware**: 1x V100 GPU

### Why These Results Matter

1. **Real Snare Sounds**: BaDumTss dataset used claps as snare substitutes. AVP integration provided 1,650 real snare sounds, critical for 91% snare accuracy.
2. **Balanced Performance**: All three drum types achieve >90% accuracy
3. **Frame-Level Approach**: ±50ms windows capture the transient attack, which contains most diagnostic information for drums

---

## 🏗️ Architecture

### System Pipeline

```
Audio Input
    ↓
[Onset Detection] (external, e.g., librosa.onset.onset_detect)
    ↓
[Extract ±50ms windows around each onset]
    ↓
[Compute Mel Spectrogram]
    ├─ 128 mel bins
    ├─ 16 time frames
    ├─ 22050 Hz sample rate
    ├─ hop_length=64
    └─ n_fft=1024
    ↓
[CNN Classifier] (110k params)
    ├─ Conv Block 1: 1→32 channels + MaxPool
    ├─ Conv Block 2: 32→64 channels + MaxPool
    ├─ Conv Block 3: 64→128 channels
    ├─ Global Average Pooling
    └─ FC Layers: 128→128→3
    ↓
[Softmax] → Probabilities for [kick, snare, hihat]
    ↓
[Output: Drum type + confidence]
```

### Model Details

**Input Shape**: (batch, 1, 128, 16)
- 1 channel (grayscale mel spectrogram)
- 128 mel frequency bins (20-8000 Hz)
- 16 time frames (~100ms window)

**Output Shape**: (batch, 3)
- 3 classes: kick (0), snare (1), hihat (2)
- Softmax probabilities (sum to 1.0)

**Architecture**:
- 3 convolutional blocks with BatchNorm, ReLU, MaxPool, Dropout
- Global Average Pooling (reduces spatial dimensions)
- 2 fully connected layers with Dropout
- Total parameters: 110,019

---

## 🎛️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_confidence` | 0.5 | Minimum confidence to keep a prediction |
| `device` | auto | Device for inference ('cpu', 'cuda', or None for auto) |
| `batch_size` | 32 | Batch size for processing multiple onsets |
| `tolerance` | 0.05 | Time tolerance for matching ground truth (50ms) |

### Recommended Settings

**Production (High Quality)**:
```python
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='checkpoints/beatbox2drums_best.pth',
    min_confidence=0.6,  # High confidence threshold
    device='cuda'
)
```

**Balanced**:
```python
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='checkpoints/beatbox2drums_best.pth',
    min_confidence=0.5,  # Default
    device='cpu'
)
```

**Sensitive (Catch More)**:
```python
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='checkpoints/beatbox2drums_best.pth',
    min_confidence=0.3,  # Lower threshold
    device='cpu'
)
```

---

## 📈 Visualization

The package includes comprehensive visualization tools that show:

1. **Audio Waveform**: Time-domain representation
2. **Mel Spectrogram**: Frequency content over time
3. **Ground Truth Onsets**: Labeled drum types at correct times
4. **Predicted Onsets**: Model predictions with confidence scores

Example usage:

```python
from evaluation import plot_drum_classification_results

# Create visualization
fig = plot_drum_classification_results(
    audio_path='audio.wav',
    ground_truth=[
        {'onset_time': 0.5, 'drum_type': 'kick'},
        {'onset_time': 1.0, 'drum_type': 'snare'},
        ...
    ],
    predictions=[
        {'onset_time': 0.51, 'drum_type': 'kick', 'confidence': 0.95},
        {'onset_time': 1.02, 'drum_type': 'snare', 'confidence': 0.88},
        ...
    ],
    output_path='results.png'
)
```

---

## 📖 API Reference

### Beatbox2DrumsClassifier

Main inference class for drum classification.

```python
classifier = Beatbox2DrumsClassifier(
    checkpoint_path,      # Path to .pth checkpoint
    device=None,          # 'cpu', 'cuda', or None for auto
    min_confidence=0.5    # Minimum confidence threshold
)

# Classify single onset
result = classifier.classify_onset(audio, sr, onset_time)

# Classify multiple onsets (batch)
results = classifier.classify_onsets(audio, sr, onset_times)

# Classify from audio file
results = classifier.classify_audio_file(audio_path, onset_times)

# Get drum counts
counts = classifier.get_drum_counts(results)

# Print results
classifier.print_results(results)
```

### Preprocessing

```python
from inference.preprocessing import (
    extract_onset_window,      # Extract ±50ms around onset
    compute_mel_spectrogram,    # Compute (128, 16) mel spec
    preprocess_onset,           # Full preprocessing pipeline
    batch_preprocess_onsets     # Batch preprocessing
)
```

### Evaluation

```python
from evaluation import (
    match_onsets,              # Match predictions to ground truth
    compute_accuracy,          # Calculate accuracy metrics
    print_evaluation_results,  # Print metrics
    plot_drum_classification_results,  # Visualization
    plot_confusion_matrix      # Confusion matrix plot
)
```

---

## 🔧 Development History

### Dataset Integration

**BaDumTss Dataset** (145,173 onsets):
- ⚠️ No real snare sounds (uses MIDI 39 = clap as snare substitute)
- 6,250 files from beatbox recordings
- Used as primary dataset

**AVP Dataset** (9,778 onsets):
- ✅ 1,650 real snare sounds
- 28 participants × 2 modes (Fixed/Personal)
- Critical for achieving 91% snare accuracy

**Combined**: 154,950 onsets (80/10/10 train/val/test split)

### Key Technical Decisions

1. **Single-Label Classification**: Beatboxers can only make one sound at a time (amateur performers)
2. **Frame-Level Preprocessing**: ±50ms windows capture transient attack
3. **Mel Spectrogram Shape**: (128, 16) chosen based on analysis:
   - hop_length=64 → 16 frames in 100ms window
   - 128 mel bins → good frequency resolution
4. **Class Weights**: Used during training to handle slight class imbalance

---

## 🎯 Known Limitations

1. **Requires External Onset Detection**: Package classifies given onsets, doesn't detect them
2. **Amateur Beatboxing Only**: Trained on amateur recordings, may not generalize to professional beatboxers
3. **Three Classes Only**: kick, snare, hihat (no tom, crash, ride, etc.)
4. **No Polyphony**: Assumes single drum per onset

---

## 🤝 Contributing

This is a production package. For modifications:

1. Update code in appropriate module
2. Run tests
3. Update documentation
4. Update version in setup.py

---

## 📄 License

MIT License - See LICENSE file

---

## 📧 Contact

For questions or issues:
- Check examples/ for usage patterns
- Review API reference above
- See evaluation/ for metrics and visualization

---

## 🏆 Citation

If you use this package, please cite:

```
Beatbox2Drums: CNN-based Drum Classification for Beatbox Recordings
Version 1.0, 2025
Test Accuracy: 93.76% (kick: 94.87%, snare: 91.10%, hihat: 95.32%)
Datasets: BaDumTss-PAKDD22 + AVP (Amateur Vocal Percussion)
```

---

**Status**: ✅ Production Ready
**Last Updated**: November 4, 2025
**Version**: 1.0
