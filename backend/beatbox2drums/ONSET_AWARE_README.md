# Beatbox2Drums (Onset-Aware Training)

This is an **onset-aware** version of the beatbox2drums package that solves the train/test distribution mismatch problem.

## The Problem

The original beatbox2drums model was trained using:
- ✅ **Perfect ground truth onset times** from human-labeled data
- ✅ Exactly centered ±50ms windows around each drum hit

But at inference, it uses:
- ❌ **Algorithmically detected onsets** with timing errors
- ❌ Windows that may not be perfectly centered on drum hits

This creates a **distribution mismatch**: the model never saw imperfectly-timed onsets during training, so it struggles with them at inference.

### Why Only Kicks Were Detected

- **Kicks**: Strong energy, low frequency, longer duration → forgiving to timing errors
- **Snares/Hihats**: Sharp transients, high frequency, short duration → require precise timing

## The Solution

Train the model using **detected onsets** (not ground truth), so it learns to handle the same timing imperfections it will see at inference.

### Workflow

```
1. Run onset detection on training audio
2. Match detected onsets to ground truth labels (within tolerance)
3. Extract ±50ms windows around DETECTED onsets (not GT onsets)
4. Train model on these "imperfect" windows
5. At inference, use the same onset detection → consistent distribution
```

## Quick Start

### 1. Evaluate Onset Detection

First, tune your onset detector to match ground truth as closely as possible:

```bash
# Evaluate onset detection on validation set
python scripts/evaluate_onset_detection.py \
    --manifest /path/to/combined/manifest.json \
    --split val \
    --onset-threshold 0.3 \
    --tolerance 0.05 \
    --output onset_eval_results.json

# Try different parameters
python scripts/evaluate_onset_detection.py \
    --onset-threshold 0.2 \
    --onset-wait 3 \
    --multiband
```

**Goal**: Maximize F1 score and minimize timing error.

### 2. Preprocess Data with Onset Detection

Once you have good onset detection parameters, preprocess the dataset:

```bash
# Extract onset windows using DETECTED onsets
python scripts/extract_onset_windows_with_detection.py \
    --manifest /path/to/combined/manifest.json \
    --output-dir /path/to/preprocessed_onset_aware \
    --onset-threshold 0.3 \
    --onset-wait 5 \
    --match-tolerance 0.05 \
    --num-workers 8
```

This creates:
```
preprocessed_onset_aware/
├── train/
│   ├── kick/
│   ├── snare/
│   └── hihat/
├── val/
└── test/
```

**Key difference**: Windows are centered on **detected** onset times, not ground truth.

### 3. Train Model

Train the model on onset-aware data:

```bash
python scripts/train.py \
    --preprocessed-dir /path/to/preprocessed_onset_aware \
    --checkpoint-dir checkpoints_onset_aware \
    --epochs 100 \
    --batch-size 64 \
    --use-class-weights
```

### 4. Inference (Same Onset Detection!)

At inference, use the **exact same onset detection parameters**:

```python
from inference import OnsetDetector, Beatbox2DrumsClassifier

# Create onset detector (MUST match training parameters)
detector = OnsetDetector(
    sample_rate=22050,
    onset_threshold=0.3,
    onset_wait=5,
    # ... same as preprocessing
)

# Detect onsets
onsets = detector.detect_from_file('beatbox.wav')

# Classify detected onsets
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='checkpoints_onset_aware/best_model.pth'
)

results = classifier.classify_audio_file(
    audio_path='beatbox.wav',
    onset_times=onsets
)
```

## Onset Detection Parameters

### Key Parameters to Tune

| Parameter | Description | Default | Effect |
|-----------|-------------|---------|--------|
| `onset_threshold` | Detection sensitivity | 0.3 | Lower = more onsets, higher = fewer |
| `onset_wait` | Min frames between onsets | 5 | Higher = fewer rapid onsets |
| `match_tolerance` | GT matching tolerance | 0.05s | Time window to match detected → GT |
| `use_multiband` | Separate band detection | False | Better for mixed drum types |
| `backtrack` | Refine to true peak | True | Improves timing accuracy |

### Recommended Tuning Process

1. **Start with defaults**:
   ```bash
   python scripts/evaluate_onset_detection.py --split val
   ```

2. **Adjust threshold** to balance precision/recall:
   - Low F1 + High FP → Increase `onset_threshold`
   - Low F1 + High FN → Decrease `onset_threshold`

3. **If rapid onsets are problematic**:
   - Increase `onset_wait` (e.g., 7-10 frames)

4. **For better drum separation**:
   - Try `--multiband` (separate kick/snare/hihat bands)

5. **Iterate until**:
   - F1 score > 0.85
   - Mean timing error < 10ms
   - Balanced precision/recall

## Expected Performance

### Onset Detection (on validation set)
- **Target F1**: > 0.85
- **Target Precision**: > 0.80
- **Target Recall**: > 0.80
- **Target Mean Error**: < 10ms

### Classification (after onset-aware training)
- Should detect all drum types (not just kicks!)
- More robust to timing variations
- Better generalization to real-world beatboxing

## File Structure

```
beatbox2drums_package_onset_aware/
├── inference/
│   ├── onset_detector.py          # Configurable onset detection
│   ├── drum_classifier.py          # CNN classifier
│   └── preprocessing.py            # Feature extraction
├── scripts/
│   ├── evaluate_onset_detection.py          # Tune onset detector
│   ├── extract_onset_windows_with_detection.py  # Preprocess with detection
│   └── train.py                    # Train model
├── data/
│   └── drum_dataset.py             # Dataset loader
└── models/
    └── drum_classifier.py          # Model architecture
```

## Comparison: GT vs Onset-Aware

| Aspect | Ground Truth Training | Onset-Aware Training |
|--------|----------------------|---------------------|
| Onset timing | Perfect (human-labeled) | Imperfect (detected) |
| Window alignment | Perfectly centered | May be off-center |
| Train/test match | ❌ Mismatch | ✅ Consistent |
| Robustness | Poor to timing errors | Good to timing errors |
| Real-world performance | Only detects kicks | All drum types |

## Debugging Tips

### If onset detection F1 is low:

1. **Check audio quality**: Low-quality audio → poor onset detection
2. **Visualize onsets**: Plot detected vs ground truth
3. **Try multiband**: Separates drums by frequency
4. **Adjust tolerance**: May need > 50ms for noisy data

### If model still only predicts kicks:

1. **Check class balance** in preprocessed data
2. **Verify onset detection** matches preprocessing parameters
3. **Use class weights** during training
4. **Increase match tolerance** to get more snare/hihat samples

### If timing errors are high:

1. **Enable backtracking**: `--backtrack` (should be on by default)
2. **Increase hop length** resolution (try 256 instead of 512)
3. **Use energy method**: Usually best for drums

## Advanced: Custom Onset Detection

You can implement custom onset detection strategies:

```python
from inference.onset_detector import OnsetDetector

class CustomOnsetDetector(OnsetDetector):
    def _detect_single_band(self, audio):
        # Your custom onset detection logic
        # For example: ML-based onset detection
        pass
```

## Next Steps

1. Run `evaluate_onset_detection.py` to find optimal parameters
2. Run `extract_onset_windows_with_detection.py` to create training data
3. Train model with `train.py`
4. Test on real beatboxing and compare to GT-trained model
5. Iterate on onset detection if needed

---

**Key Insight**: The best onset detector isn't the one with perfect metrics on evaluation data—it's the one whose errors match what the model will see at inference time. Sometimes a slightly "worse" detector creates better training data!
