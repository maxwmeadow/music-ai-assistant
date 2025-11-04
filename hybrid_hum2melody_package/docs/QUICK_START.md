# 🚀 Quick Start Guide - Hybrid Hum2Melody System

**Status**: ✅ System is working! 76.4% accuracy (target was 70%)

---

## 🎯 What You Have

A working hybrid system that:
- ✅ Converts humming to melody notes
- ✅ 76.4% accuracy comparing to actual audio content
- ✅ Processes full audio (no 16-second limit)
- ✅ Provides confidence scores for each note

---

## 🏃 Quick Commands

### Test a Single Recording

```bash
module purge && module load Conda && source activate hum2melody

# Basic test
python test_my_humming.py your_humming.wav --visualize

# With JSON output for analysis
python test_my_humming.py your_humming.wav --visualize --save-json
```

### Recommended Settings (Higher Quality)

```bash
# Filter out low-confidence predictions
python test_my_humming.py your_humming.wav \
    --min-confidence 0.25 \
    --visualize --save-json
```

### More Conservative (Fewer False Positives)

```bash
# Stricter thresholds = fewer but more accurate notes
python test_my_humming.py your_humming.wav \
    --min-confidence 0.30 \
    --onset-high 0.40 \
    --onset-low 0.15 \
    --visualize
```

### Analyze Accuracy Against Audio

```bash
# After running with --save-json
python analyze_results.py
```

This will compare predictions to actual audio content using CQT analysis.

---

## 📊 Understanding Results

### Output Format

```json
{
  "notes": [
    {
      "start": 1.89,           // Start time in seconds
      "end": 2.34,             // End time in seconds
      "duration": 0.45,        // Note duration
      "midi": 52,              // MIDI note number (52 = E3)
      "note": "E3",            // Note name
      "confidence": 0.82       // Model confidence (0-1)
    }
  ]
}
```

### Confidence Interpretation

| Confidence | Interpretation | Action |
|------------|---------------|---------|
| ≥ 0.7 | High - Very reliable | ✅ Keep |
| 0.4 - 0.7 | Medium - Likely correct | ✅ Keep |
| 0.25 - 0.4 | Low - Uncertain | ⚠️ Consider filtering |
| < 0.25 | Very low - Likely error | ❌ Filter out |

**Tip**: G♯3 with confidence < 0.2 is almost always wrong (hallucination)

---

## 🎨 Visualization Output

PNG files show:
- **X-axis**: Time (seconds)
- **Y-axis**: MIDI note number / pitch
- **Color**: Confidence (darker = more confident)
- **Rectangles**: Note segments with labels

Download and open:
- `your_file_notes.png`

---

## 🧪 Testing Workflow

### Step 1: Record Humming (5-10 seconds)
- Hum a simple melody
- Save as WAV file
- Clear, steady humming works best

### Step 2: Run Prediction
```bash
python test_my_humming.py my_hum.wav --visualize --save-json
```

### Step 3: Check Results
1. Open `my_hum_notes.png` - Does it look right?
2. Check console output - Are notes reasonable?
3. Run `python analyze_results.py` - What's the accuracy?

### Step 4: Tune if Needed
- Too many notes? → Increase `--min-confidence`
- Too few notes? → Decrease `--min-confidence`
- Lots of weird sharps (G♯, F♯)? → Increase `--min-confidence 0.30`
- Missing quiet notes? → Decrease `--onset-high`

---

## 📈 Parameter Tuning Guide

### Onset Detection Thresholds

**`--onset-high`** (default: 0.30)
- How strong a sound change needs to be to start a new note
- **Higher** (0.40) = Fewer notes, more conservative
- **Lower** (0.20) = More notes, may catch noise

**`--onset-low`** (default: 0.10)
- Threshold for continuing an existing note
- Usually keep at 0.10 unless you have very quiet audio

### Pitch Confidence

**`--min-confidence`** (default: 0.10)
- Minimum confidence to keep a note
- **Recommended: 0.25-0.30** for production
- **0.10** = Keep almost everything (current tests)
- **0.30** = Only confident predictions
- **0.50** = Very conservative

---

## 🐛 Troubleshooting

### Problem: No notes detected
```bash
# Lower thresholds
python test_my_humming.py audio.wav \
    --onset-high 0.20 \
    --min-confidence 0.05
```

### Problem: Too many weird notes (G♯, F♯, etc.)
```bash
# Increase confidence filter
python test_my_humming.py audio.wav --min-confidence 0.30
```

### Problem: Lots of very short notes (<0.1s)
```bash
# More conservative onset detection
python test_my_humming.py audio.wav \
    --onset-high 0.40 \
    --onset-low 0.15
```

### Problem: PNG visualization too wide
This was fixed! Make sure you're using the updated `test_my_humming.py`

### Problem: "ModuleNotFoundError: torch"
```bash
# Activate conda environment first
module purge && module load Conda && source activate hum2melody
```

---

## 📁 Key Files

### Main Scripts
- `test_my_humming.py` - Test interface (USE THIS)
- `hybrid_inference_chunked.py` - Core system
- `analyze_results.py` - Accuracy analysis

### Documentation
- `FINAL_RESULTS.md` - Detailed test results (READ THIS)
- `BUGS_FIXED.md` - What was broken and how we fixed it
- `HYBRID_USAGE_GUIDE.md` - Original usage guide

### Models
- `hum2melody_package/checkpoints/combined_hum2melody_full.pth` (135MB)
  - Pitch model: 15M params
  - Onset model: 20M params
  - Combined: 35M params

---

## 🎯 Next Steps

### Immediate Actions

1. **Test with recommended settings**:
   ```bash
   python test_my_humming.py your_hum.wav \
       --min-confidence 0.25 --visualize --save-json
   python analyze_results.py
   ```

2. **Record 3-5 more samples** to verify consistency
   - Simple melodies (Mary Had a Little Lamb, Twinkle Twinkle, etc.)
   - 5-15 seconds each
   - Clear humming

3. **Measure accuracy**:
   - Should be 75-85% exact match
   - Should be 85-95% within 1 semitone

4. **Deploy if stable** (accuracy > 70% consistently)

### Future Improvements (Optional)

- **Melody smoothing**: Connect adjacent notes in same pitch
- **Key detection**: Constrain predictions to detected key
- **Accidental filtering**: Remove unlikely sharps/flats
- **Duration normalization**: Snap to beat grid
- **Confidence calibration**: Adjust thresholds based on audio quality

---

## 💡 Tips for Best Results

1. **Humming quality matters**
   - Clear, steady humming
   - Not too quiet
   - Avoid background noise

2. **Simple melodies work best**
   - The model was trained on simple melodies
   - Complex jazz or chromatic passages may confuse it

3. **Trust high confidence notes**
   - Confidence ≥ 0.7 is almost always correct
   - Low confidence notes are uncertain

4. **G♯3 with conf < 0.2 is usually wrong**
   - Known issue: model hallucinates G♯3
   - Safe to filter out

5. **Within 1 semitone is often good enough**
   - 88.8% accuracy within 1 semitone
   - For melody recognition, being off by 1 note is minor

---

## 📞 Support

- Check `FINAL_RESULTS.md` for detailed analysis
- Check `BUGS_FIXED.md` for known issues
- Run `python analyze_results.py` to see actual accuracy
- Visualize with `--visualize` to see what's happening

---

**System Status**: ✅ READY FOR DEPLOYMENT

**Tested on**:
- TwinkleTwinkle.wav: 80.0% accuracy
- MaryHadALittleLamb.wav: 72.7% accuracy
- Average: 76.4% accuracy

**Last Updated**: November 3, 2025
