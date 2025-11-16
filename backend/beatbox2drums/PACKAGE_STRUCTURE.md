# Beatbox2Drums Package Structure (CNN-Based)

This package contains only the CNN-based components for the Beatbox2Drums pipeline.

## Directory Structure

```
backend/beatbox2drums/
├── inference/                      # Inference components
│   ├── cnn_onset_detector.py      # CNN-based onset detection
│   ├── beatbox2drums_pipeline.py  # Complete end-to-end pipeline
│   └── __init__.py                # Module exports
│
├── models/                         # Model architectures
│   ├── drum_classifier.py         # CNN drum classifier architecture
│   ├── onset_model.py             # CNN onset detection model
│   └── __init__.py
│
├── data/                           # Dataset classes
│   ├── drum_dataset_cnn.py        # Dataset for CNN-based training
│   └── __init__.py
│
├── scripts/                        # Training & visualization scripts
│   ├── prepare_cnn_onset_data.py          # Prepare onset detection data
│   ├── train_cnn_onset_detector.py        # Train onset detector
│   ├── visualize_cnn_predictions.py       # Visualize onset detection
│   ├── large_scale_onset_analysis.py      # Large-scale onset analysis
│   ├── prepare_classifier_data_cnn.py     # Prepare classifier data
│   ├── train_classifier_cnn.py            # Train drum classifier
│   ├── visualize_pipeline_predictions.py  # Visualize pipeline predictions
│   ├── evaluate_pipeline_performance.py   # Evaluate pipeline performance
│   ├── analyze_postprocessing_options.py  # (Optional) Analyze NMS/threshold options
│   ├── visualize_specific_files.py        # (Optional) Visualize specific files
│   └── __init__.py
│
├── examples/                       # Usage examples
│   ├── pipeline_example.py        # Basic pipeline usage example
│   └── __init__.py
│
├── tests/                          # Test directory (placeholder)
│   └── __init__.py
│
├── PIPELINE_README.md              # Comprehensive documentation
├── CNN_ONSET_TRAINING_PROGRESS.md  # Training history
└── PACKAGE_STRUCTURE.md            # This file
```

## Key Components

### 1. Inference Pipeline

**`inference/beatbox2drums_pipeline.py`**
- Complete end-to-end pipeline
- Combines CNN onset detection + CNN classification
- **Usage:**
  ```python
  from backend.beatbox2drums.inference import Beatbox2DrumsPipeline

  pipeline = Beatbox2DrumsPipeline(
      onset_checkpoint_path='path/to/onset_model.h5',
      classifier_checkpoint_path='path/to/classifier.pth'
  )
  drum_hits = pipeline.predict('audio.wav')
  ```

**`inference/cnn_onset_detector.py`**
- CNN-based onset detection with NMS
- Uses TensorFlow/Keras model (.h5)

### 2. Model Architectures

**`models/drum_classifier.py`**
- DrumClassifierCNN: PyTorch CNN for drum type classification
- Input: 128-band mel spectrogram
- Output: Softmax probabilities for kick/snare/hihat

**`models/onset_model.py`**
- DrumOnsetDetectionCNN: Temporal CNN for onset detection
- Input: 80-band mel spectrogram
- Output: Frame-wise onset probabilities

### 3. Training Scripts

**Onset Detection:**
1. `scripts/prepare_cnn_onset_data.py` - Generate training data
2. `scripts/train_cnn_onset_detector.py` - Train model

**Drum Classification:**
1. `scripts/prepare_classifier_data_cnn.py` - Generate training data using CNN onsets
2. `scripts/train_classifier_cnn.py` - Train classifier

### 4. Visualization Scripts

**Onset Detection:**
- `scripts/visualize_cnn_predictions.py` - Individual sample visualizations
- `scripts/large_scale_onset_analysis.py` - Large-scale onset analysis

**Pipeline:**
- `scripts/visualize_pipeline_predictions.py` - Individual pipeline predictions
- `scripts/evaluate_pipeline_performance.py` - Comprehensive performance evaluation

## Removed Legacy Components

The following legacy components have been removed:
- ❌ `inference/onset_detector_amplitude.py` - Amplitude-based onset detection
- ❌ `inference/onset_detector_aubio.py` - Aubio-based onset detection
- ❌ `inference/onset_detector.py` - Old onset detector base class
- ❌ `inference/drum_classifier.py` - Old direct classifier
- ❌ `inference/preprocessing.py` - Legacy preprocessing
- ❌ `evaluation/` - Legacy evaluation module
- ❌ `data/drum_dataset.py` - Legacy dataset class
- ❌ `models/model_loader.py` - Legacy model loader
- ❌ `examples/basic_inference.py` - Legacy example
- ❌ `examples/evaluate_with_visualization.py` - Legacy example
- ❌ `scripts/evaluate_by_dataset.py` - Legacy evaluation
- ❌ `scripts/evaluate_onset_detection.py` - Legacy onset evaluation
- ❌ `scripts/evaluate_onset_detection_aubio.py` - Aubio evaluation
- ❌ `scripts/evaluate_onset_detection_amplitude.py` - Amplitude evaluation
- ❌ `scripts/extract_onset_windows_with_detection.py` - Legacy extraction
- ❌ `scripts/generate_example_visualizations.py` - Legacy viz
- ❌ `scripts/generate_high_accuracy_visualizations.py` - Legacy viz
- ❌ `scripts/generate_test_visualizations.py` - Legacy viz
- ❌ `scripts/generate_visualizations_standalone.py` - Legacy viz
- ❌ `scripts/train.py` - Legacy training
- ❌ `scripts/verify_model.py` - Legacy verification

## Performance

- **Onset Detection**: 91.3% F1-score with 50ms NMS
- **Drum Classification**: 99.39% validation accuracy
  - Kick: 100.00%
  - Snare: 98.97%
  - Hihat: 99.37%

## Quick Start

See [PIPELINE_README.md](PIPELINE_README.md) for detailed usage instructions and examples.
