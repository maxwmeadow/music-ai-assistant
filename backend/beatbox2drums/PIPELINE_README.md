# Beatbox2Drums CNN Pipeline

Complete end-to-end pipeline for converting beatbox audio to drum hit events using CNN-based onset detection and drum classification.

## Overview

This pipeline implements a two-stage approach:
1. **CNN Onset Detection**: Identifies potential drum hits in the audio using a temporal convolutional network
2. **CNN Classification**: Classifies each detected onset as kick, snare, or hihat using a mel-spectrogram classifier

### Key Features

- **99.39% Classification Accuracy** on validation data
- **Train/Test Consistency**: Both models trained and evaluated using CNN-detected onsets
- **NMS Post-Processing**: 50ms non-maximum suppression eliminates timing jitter
- **Confidence Filtering**: Optional threshold to reject uncertain predictions
- **GPU Accelerated**: 10-50x faster than CPU-only processing

## Quick Start

### 1. Basic Usage

```python
from backend.beatbox2drums.inference.beatbox2drums_pipeline import Beatbox2DrumsPipeline

# Initialize pipeline
pipeline = Beatbox2DrumsPipeline(
    onset_checkpoint_path='beatbox2drums/cnn_onset_checkpoints/best_model.pth',
    classifier_checkpoint_path='beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth',
    onset_threshold=0.5,                    # Onset detection threshold
    onset_peak_delta=0.05,                  # 50ms NMS window
    classifier_confidence_threshold=0.3      # Min confidence for classification
)

# Predict drum hits from audio file
drum_hits = pipeline.predict('path/to/beatbox.wav')

# Print results
for hit in drum_hits:
    print(f"{hit.drum_type}: {hit.time:.3f}s (confidence: {hit.confidence:.3f})")
```

### 2. Get Detailed Results

```python
# Get additional information
results = pipeline.predict('path/to/beatbox.wav', return_details=True)

print(f"Total onsets detected: {results['total_onsets']}")
print(f"Drum hits after filtering: {len(results['drum_hits'])}")
print(f"Rejected (low confidence): {results['rejected_count']}")

# Get statistics
stats = pipeline.get_statistics(results['drum_hits'])
print(f"\nDetected:")
print(f"  Kicks: {stats['by_type']['kick']}")
print(f"  Snares: {stats['by_type']['snare']}")
print(f"  Hihats: {stats['by_type']['hihat']}")
```

## Visualization Tools

### Individual Sample Visualization

```bash
python beatbox2drums_package_onset_aware/scripts/visualize_pipeline_predictions.py \
    --onset-checkpoint beatbox2drums/cnn_onset_checkpoints/best_model.pth \
    --classifier-checkpoint beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth \
    --manifest beatbox2drums/dataset/combined/manifest.json \
    --split val \
    --num-samples 5 \
    --output-dir beatbox2drums/visualizations/pipeline
```

**Output**: Generates detailed visualizations showing:
- Waveform with onset detections
- Mel spectrogram with classified drum hits
- Predicted drum hits timeline with confidence scores
- Ground truth comparison

### Large-Scale Performance Evaluation

```bash
python beatbox2drums_package_onset_aware/scripts/evaluate_pipeline_performance.py \
    --onset-checkpoint beatbox2drums/cnn_onset_checkpoints/best_model.pth \
    --classifier-checkpoint beatbox2drums/classifier_checkpoints_cnn/best_model_cnn.pth \
    --manifest beatbox2drums/dataset/combined/manifest.json \
    --split val \
    --output-dir beatbox2drums/evaluation_results
```

**Output**: Comprehensive evaluation including:
- Per-class Precision, Recall, F1-Score
- True Positive / False Positive / False Negative counts
- Timing error distribution
- Confidence distribution
- Performance visualization plots

## Onset Detection Visualizations

### Individual Onset Detection Examples

```bash
python beatbox2drums_package_onset_aware/scripts/visualize_cnn_predictions.py \
    --checkpoint beatbox2drums/cnn_onset_checkpoints/best_model.pth \
    --manifest beatbox2drums/dataset/combined/manifest.json \
    --num-samples 5 \
    --output-dir beatbox2drums/visualizations/onset_detection
```

### Large-Scale Onset Analysis

```bash
python beatbox2drums_package_onset_aware/scripts/large_scale_onset_analysis.py \
    --checkpoint beatbox2drums/cnn_onset_checkpoints/best_model.pth \
    --manifest beatbox2drums/dataset/combined/manifest.json \
    --split val \
    --output-dir beatbox2drums/onset_analysis
```

## Performance Metrics

### Classifier Performance (99.39% Validation Accuracy)

| Drum Type | Accuracy |
|-----------|----------|
| Kick      | 100.00%  |
| Snare     | 98.97%   |
| Hihat     | 99.37%   |

### Training Data Statistics

- **Training Windows**: 118,765 (from 5,220 audio files)
- **Validation Windows**: 14,530 (from 645 audio files)
- **CNN Onset Match Rate**: 95.1%
- **Class Distribution**:
  - Kick: 26.3%
  - Snare: 32.7%
  - Hihat: 40.9%

## Technical Details

### Model Architecture

**Onset Detector (DrumOnsetDetectionCNN)**:
- Input: 80-band mel spectrogram
- Architecture: Temporal CNN with 8 convolutional layers
- Output: Frame-wise onset probabilities
- Training: Trained on beatbox audio with ground truth onset times
- Post-processing: NMS with 50ms window

**Drum Classifier (DrumClassifierCNN)**:
- Input: 128-band mel spectrogram (12 frames, ~330ms window)
- Architecture: CNN with adaptive pooling (handles variable window sizes)
- Output: Softmax probabilities for 3 classes (kick, snare, hihat)
- Training: Trained on CNN-detected onsets matched to ground truth types
- Parameters: ~200K trainable parameters

### Audio Processing Parameters

```python
{
    'sample_rate': 16000,          # Hz
    'n_mels_classifier': 128,      # Mel bands for classifier
    'n_mels_onset': 80,            # Mel bands for onset detector
    'hop_length': 441,             # ~27.55ms frames
    'window_frames': 12,           # Classifier window (330ms)
    'nms_window': 0.05,            # 50ms NMS window
}
```

## Directory Structure

```
beatbox2drums_package_onset_aware/
├── inference/
│   ├── cnn_onset_detector.py         # CNN onset detector
│   ├── beatbox2drums_pipeline.py     # Complete pipeline
│   └── drum_classifier.py            # (Legacy) Direct classifier
├── models/
│   ├── drum_classifier.py            # CNN classifier architecture
│   └── onset_model.py                # CNN onset model architecture
├── scripts/
│   ├── visualize_pipeline_predictions.py      # Individual sample viz
│   ├── evaluate_pipeline_performance.py       # Large-scale evaluation
│   ├── visualize_cnn_predictions.py           # Onset detection viz
│   ├── large_scale_onset_analysis.py          # Onset detection analysis
│   ├── prepare_classifier_data_cnn.py         # Data preparation
│   └── train_classifier_cnn.py                # Classifier training
├── data/
│   └── drum_dataset_cnn.py           # Dataset class for CNN-based data
└── examples/
    └── basic_inference.py            # Example usage scripts
```

## Workflow

### Training Pipeline

1. **Prepare CNN Onset Detector**:
   ```bash
   # Train onset detector on ground truth onsets
   python scripts/train_cnn_onset_detector.py
   ```

2. **Generate Classifier Training Data**:
   ```bash
   # Use CNN onset detector to generate training data
   sbatch generate_classifier_train_data.sb
   ```

3. **Train Classifier**:
   ```bash
   # Train classifier on CNN-detected onsets
   sbatch train_classifier_cnn.sb
   ```

### Inference Pipeline

1. Load audio → 2. CNN onset detection → 3. For each onset: extract window → 4. CNN classification → 5. Confidence filtering → 6. Output drum hits

## Configuration

### Onset Detection Thresholds

- **`onset_threshold`** (default: 0.5): Minimum probability for onset detection
  - Higher = fewer false positives, more false negatives
  - Lower = more false negatives, fewer false positives

- **`onset_peak_delta`** (default: 0.05s): NMS window size
  - Merges multiple detections within 50ms
  - Keeps only the highest probability peak

### Classification Thresholds

- **`classifier_confidence_threshold`** (default: 0.3): Minimum confidence for drum type
  - Higher = more precise but may miss some hits
  - Lower = more recall but may include uncertain predictions

## Troubleshooting

### Low Onset Detection

- Try lowering `onset_threshold` (e.g., 0.4 or 0.3)
- Check audio quality and sample rate

### Many False Positives

- Increase `classifier_confidence_threshold` (e.g., 0.4 or 0.5)
- Increase `onset_peak_delta` for more aggressive NMS

### Wrong Drum Type Classifications

- This indicates the classifier needs more training data for those types
- Check ground truth labels for consistency

## Citation

If you use this work, please cite:

```bibtex
@software{beatbox2drums_cnn_pipeline,
  title={Beatbox2Drums: CNN-Based Beatbox to Drum Transcription},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/beatbox2drums}
}
```

## License

MIT License - See LICENSE file for details
