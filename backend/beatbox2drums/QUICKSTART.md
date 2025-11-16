# Quick Start: Onset-Aware Training

## Step-by-Step Commands

### Step 1: Test Onset Detector

First, verify the onset detector works:

```bash
cd /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums_package_onset_aware

# Test the onset detector module
python -c "from inference.onset_detector import OnsetDetector; print('✓ OnsetDetector imported successfully')"

# Run basic test
python inference/onset_detector.py
```

Expected output: Detection statistics on synthetic test audio.

---

### Step 2: Evaluate Onset Detection on Real Data

Find the best onset detection parameters:

```bash
# Evaluate on validation set (start with defaults)
python scripts/evaluate_onset_detection.py \
    --manifest /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json \
    --split val \
    --max-files 50 \
    --onset-threshold 0.3 \
    --tolerance 0.05
```

**Look for**:
- F1 Score > 0.80
- Mean timing error < 15ms
- Balanced precision/recall

**If F1 is too low**:

```bash
# Try lower threshold (more sensitive)
python scripts/evaluate_onset_detection.py \
    --split val --max-files 50 \
    --onset-threshold 0.2

# Or try multiband
python scripts/evaluate_onset_detection.py \
    --split val --max-files 50 \
    --onset-threshold 0.3 \
    --multiband
```

**Save best parameters** to a file for later reference.

---

### Step 3: Extract Training Data with Onset Detection

Once you have good parameters (let's say `threshold=0.3` works well):

```bash
# Create output directory
mkdir -p /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware

# Extract onset windows using detected onsets
python scripts/extract_onset_windows_with_detection.py \
    --manifest /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json \
    --output-dir /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware \
    --onset-threshold 0.3 \
    --onset-wait 5 \
    --match-tolerance 0.05 \
    --num-workers 8 \
    --sample-rate 22050 \
    --window-ms 100 \
    --n-mels 128 \
    --target-frames 16
```

This will take a while (processing all audio files with onset detection).

**Monitor output** for:
- Match rate: Should be > 70%
- Mean timing error: Should be < 20ms
- Balanced drum counts (kick/snare/hihat)

---

### Step 4: Verify Preprocessed Data

```bash
# Check output structure
ls -lh /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware/train/

# Count samples per class
echo "Kick:"; find /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware/train/kick/ -name "*.npy" | wc -l
echo "Snare:"; find /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware/train/snare/ -name "*.npy" | wc -l
echo "Hihat:"; find /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware/train/hihat/ -name "*.npy" | wc -l
```

---

### Step 5: Train Model on Onset-Aware Data

```bash
# Create directories
mkdir -p /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/checkpoints_onset_aware
mkdir -p /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/logs_onset_aware

# Train model
python scripts/train.py \
    --preprocessed-dir /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware \
    --checkpoint-dir /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/checkpoints_onset_aware \
    --log-dir /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/logs_onset_aware \
    --epochs 100 \
    --batch-size 64 \
    --use-class-weights \
    --weighted-sampling \
    --early-stopping-patience 15
```

**Monitor training**:
- Validation accuracy should reach > 85% across all classes
- Check per-class accuracy (kick, snare, hihat should all be > 80%)

---

### Step 6: Test Inference

Create a test script:

```python
# test_onset_aware_inference.py
import sys
sys.path.insert(0, '/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums_package_onset_aware')

from inference import OnsetDetector, Beatbox2DrumsClassifier

# Create onset detector (SAME parameters as preprocessing!)
detector = OnsetDetector(
    sample_rate=22050,
    onset_threshold=0.3,
    onset_wait=5,
    backtrack=True
)

# Load model
classifier = Beatbox2DrumsClassifier(
    checkpoint_path='/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/checkpoints_onset_aware/best_model.pth'
)

# Test on audio
audio_path = 'path/to/test/beatbox.wav'

# Detect onsets
onsets = detector.detect_from_file(audio_path)
print(f"Detected {len(onsets)} onsets")

# Classify
results = classifier.classify_audio_file(audio_path, onsets)

# Print results
for result in results:
    print(f"{result['onset_time']:.3f}s: {result['drum_type']} (confidence: {result['confidence']:.2f})")
```

---

## SLURM Job (If Running on HPC)

```bash
#!/bin/bash
#SBATCH --job-name=onset_aware_prep
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --output=onset_aware_preprocessing_%j.log

module purge
module load Conda
conda activate music-ai

cd /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums_package_onset_aware

# Run preprocessing
python scripts/extract_onset_windows_with_detection.py \
    --manifest /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/dataset/combined/manifest.json \
    --output-dir /mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums/preprocessed_onset_aware \
    --onset-threshold 0.3 \
    --num-workers 16
```

---

## Expected Timeline

1. **Step 1-2**: 10 minutes (testing + eval on 50 files)
2. **Step 3**: 2-4 hours (full preprocessing with onset detection)
3. **Step 4**: 5 minutes (verification)
4. **Step 5**: 4-8 hours (training to convergence)
5. **Step 6**: 5 minutes (testing)

**Total**: ~1 day of compute time

---

## Comparison Script

After training, compare to original GT-trained model:

```python
# compare_models.py
import sys
sys.path.insert(0, '/mnt/gs21/scratch/meadowm1/music-ai-training/beatbox2drums_package_onset_aware')

from inference import OnsetDetector
# Load both models and compare on same test set

# GT model
gt_classifier = ...  # original model

# Onset-aware model
oa_classifier = ...  # new model

# Test on multiple files
# Compare:
# - Overall accuracy
# - Per-class accuracy (especially snare/hihat!)
# - Robustness to timing variations
```

---

## Troubleshooting

### "No detected onsets matched to ground truth"

- **Cause**: Onset detector is too conservative or audio is corrupted
- **Fix**: Lower `onset_threshold` or check audio file

### "Match rate < 50%"

- **Cause**: Onset detector parameters don't match dataset characteristics
- **Fix**: Re-run Step 2 with different parameters

### "Model only predicts kick"

- **Cause**: Class imbalance in preprocessed data
- **Fix**:
  1. Check class counts in Step 4
  2. Use `--use-class-weights` in training
  3. Adjust `match_tolerance` to get more snare/hihat samples

### "Training accuracy is low"

- **Cause**: Model seeing very different data than GT-trained version
- **Fix**: This is expected initially! The model is learning from imperfect onsets. Give it more epochs.

---

## Success Criteria

**Before onset-aware training**:
- Model accuracy: ~90% overall
- But in practice: Only kicks detected

**After onset-aware training**:
- Model accuracy: ~85-90% overall (slight drop is OK)
- In practice: All drum types detected consistently!
- Robust to timing variations

The goal is **real-world performance**, not just test set accuracy.
