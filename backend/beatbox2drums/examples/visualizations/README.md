# Example Visualizations

This directory contains example visualization images showing the beatbox2drums model's predictions on test samples.

## What the Visualizations Show

Each visualization image contains 4 aligned plots:

1. **Audio Waveform** (top)
   - Time-domain representation of the input audio
   - Shows amplitude over time

2. **Mel Spectrogram** (middle-top)
   - Frequency content of the audio over time
   - Colormap: viridis (yellow = higher energy, purple = lower energy)
   - Frequency range: 0-8000 Hz

3. **Ground Truth Onsets** (middle-bottom)
   - Correct drum classifications from labeled data
   - Color-coded by drum type
   - Shows what the model should predict

4. **Predicted Onsets** (bottom)
   - Model's predictions for each onset
   - Color-coded by predicted drum type
   - Marker size and transparency indicate confidence
     - Larger, more opaque = higher confidence
     - Smaller, more transparent = lower confidence

## Color Scheme

- **Red** (triangle down ▼): Kick drum
- **Teal** (square ■): Snare drum
- **Yellow** (triangle up ▲): Hi-hat

## Example Results

The visualization images in this directory show real test samples with varying accuracy:

- `sample_1_5149.png`: 70% accuracy (7/10 onsets correct)
- `sample_2_63.png`: 50% accuracy (13/26 onsets correct)
- `sample_3_2035.png`: 76.2% accuracy (16/21 onsets correct)

## Generating More Visualizations

To generate additional visualization images:

```bash
# Using the standalone script (no installation required)
python3 scripts/generate_visualizations_standalone.py --num-samples 5

# Or after installing the package
python3 scripts/generate_example_visualizations.py --num-samples 5
```

## Interpreting the Results

**Good predictions**: When ground truth and predictions align vertically with matching colors

**Classification errors**: When markers are vertically aligned but colors differ (wrong drum type predicted)

**Timing errors**: When markers are misaligned horizontally (onset time mismatch, though within 50ms tolerance)

## Overall Model Performance

- **Test Accuracy**: 93.76%
- **Kick**: 94.87%
- **Snare**: 91.10%
- **Hi-hat**: 95.32%

Note: Individual samples may show lower accuracy than the overall test set average, which is normal variation.
