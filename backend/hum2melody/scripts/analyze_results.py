#!/usr/bin/env python3
"""
Analyze prediction results quality by comparing to actual audio content
"""
import json
import sys
from pathlib import Path
import numpy as np
from collections import Counter
import librosa

def load_predictions(json_path):
    """Load predictions from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def hz_to_midi(freq_hz):
    """Convert frequency in Hz to MIDI note number"""
    if freq_hz <= 0:
        return -1
    return int(round(12 * np.log2(freq_hz / 440.0) + 69))

def midi_to_hz(midi):
    """Convert MIDI note to Hz"""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def get_dominant_pitch_in_segment(audio, sr, start_time, end_time, hop_length=512):
    """
    Analyze audio segment and return dominant pitch using CQT.
    Returns (midi_note, confidence, energy)
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    segment = audio[start_sample:end_sample]

    if len(segment) < hop_length:
        return -1, 0.0, 0.0

    # Compute CQT (same as model uses)
    try:
        cqt = librosa.cqt(
            y=segment,
            sr=sr,
            hop_length=hop_length,
            n_bins=88,
            bins_per_octave=12,
            fmin=27.5  # A0
        )
        cqt_mag = np.abs(cqt)

        # Average over time to get dominant pitch
        pitch_energy = cqt_mag.mean(axis=1)

        # Find dominant pitch
        dominant_bin = pitch_energy.argmax()
        total_energy = pitch_energy.sum()

        if total_energy > 0:
            confidence = pitch_energy[dominant_bin] / total_energy
        else:
            confidence = 0.0

        # Convert bin to MIDI (bin 0 = MIDI 21 = A0)
        midi_note = dominant_bin + 21

        return midi_note, confidence, total_energy

    except Exception as e:
        print(f"   ⚠️  Error analyzing segment {start_time:.2f}-{end_time:.2f}s: {e}")
        return -1, 0.0, 0.0

def compare_to_audio(notes, audio_path):
    """
    Compare predicted notes to actual audio content.
    Returns accuracy metrics.
    """
    print(f"\n🎵 AUDIO CONTENT VERIFICATION")
    print(f"   Loading audio: {audio_path}")

    if not Path(audio_path).exists():
        print(f"   ❌ Audio file not found: {audio_path}")
        return None

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"   Loaded: {len(audio)} samples ({len(audio)/sr:.2f}s) @ {sr}Hz")

    # Analyze each predicted note
    exact_matches = 0
    within_1_semitone = 0
    within_2_semitones = 0
    total_valid = 0

    mismatches = []

    print(f"\n   Analyzing {len(notes)} predictions against audio...")

    for i, note in enumerate(notes):
        start_time = note['start']
        end_time = note['end']
        pred_midi = note['midi']

        # Get actual dominant pitch in this segment
        actual_midi, conf, energy = get_dominant_pitch_in_segment(
            audio, sr, start_time, end_time
        )

        if actual_midi < 0 or energy < 1e-6:
            # Segment too quiet or error
            continue

        total_valid += 1

        # Compare
        semitone_diff = abs(pred_midi - actual_midi)

        if semitone_diff == 0:
            exact_matches += 1
        if semitone_diff <= 1:
            within_1_semitone += 1
        if semitone_diff <= 2:
            within_2_semitones += 1

        # Track mismatches
        if semitone_diff > 2:
            note_names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
            pred_note_name = f"{note_names[pred_midi % 12]}{(pred_midi // 12) - 1}"
            actual_note_name = f"{note_names[actual_midi % 12]}{(actual_midi // 12) - 1}"

            mismatches.append({
                'time': start_time,
                'predicted': pred_note_name,
                'actual': actual_note_name,
                'diff_semitones': semitone_diff,
                'pred_confidence': note['confidence']
            })

    if total_valid == 0:
        print(f"   ❌ No valid segments to analyze")
        return None

    # Calculate metrics
    exact_acc = 100 * exact_matches / total_valid
    within_1_acc = 100 * within_1_semitone / total_valid
    within_2_acc = 100 * within_2_semitones / total_valid

    print(f"\n   📊 Accuracy vs Actual Audio:")
    print(f"      Exact match:        {exact_matches:3d}/{total_valid} ({exact_acc:5.1f}%)")
    print(f"      Within 1 semitone:  {within_1_semitone:3d}/{total_valid} ({within_1_acc:5.1f}%)")
    print(f"      Within 2 semitones: {within_2_semitones:3d}/{total_valid} ({within_2_acc:5.1f}%)")

    # Show worst mismatches
    if mismatches:
        print(f"\n   ⚠️  Mismatches > 2 semitones: {len(mismatches)}")
        print(f"      Top 5 worst:")
        sorted_mismatches = sorted(mismatches, key=lambda x: x['diff_semitones'], reverse=True)
        for m in sorted_mismatches[:5]:
            print(f"         {m['time']:6.2f}s: Predicted {m['predicted']:5s}, "
                  f"Actually {m['actual']:5s} (off by {m['diff_semitones']} semitones, "
                  f"conf={m['pred_confidence']:.3f})")

    return {
        'total_valid': total_valid,
        'exact_matches': exact_matches,
        'within_1': within_1_semitone,
        'within_2': within_2_semitones,
        'exact_accuracy': exact_acc,
        'within_1_accuracy': within_1_acc,
        'within_2_accuracy': within_2_acc,
        'mismatches': mismatches
    }

def analyze_predictions(notes, filename, audio_path=None):
    """Analyze prediction quality"""

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {filename}")
    print(f"{'='*70}\n")

    if not notes:
        print("❌ No notes detected!")
        return

    # Basic stats
    total_notes = len(notes)
    confidences = [n['confidence'] for n in notes]
    durations = [n['end'] - n['start'] for n in notes]

    # Handle both 'midi' and 'midi_note' keys
    midi_key = 'midi' if 'midi' in notes[0] else 'midi_note'

    print(f"📊 BASIC STATISTICS")
    print(f"   Total notes: {total_notes}")
    print(f"   Time span: {notes[0]['start']:.2f}s - {notes[-1]['end']:.2f}s ({notes[-1]['end'] - notes[0]['start']:.2f}s)")
    print()

    # Confidence analysis
    print(f"🎯 CONFIDENCE ANALYSIS")
    print(f"   Mean confidence: {np.mean(confidences):.3f}")
    print(f"   Median confidence: {np.median(confidences):.3f}")
    print(f"   Std dev: {np.std(confidences):.3f}")
    print(f"   Min: {np.min(confidences):.3f}, Max: {np.max(confidences):.3f}")

    # Confidence distribution
    high_conf = sum(1 for c in confidences if c >= 0.7)
    med_conf = sum(1 for c in confidences if 0.4 <= c < 0.7)
    low_conf = sum(1 for c in confidences if c < 0.4)

    print(f"\n   Distribution:")
    print(f"   ✅ High (≥0.7):  {high_conf:3d} notes ({100*high_conf/total_notes:5.1f}%)")
    print(f"   ⚠️  Medium (0.4-0.7): {med_conf:3d} notes ({100*med_conf/total_notes:5.1f}%)")
    print(f"   ❌ Low (<0.4):   {low_conf:3d} notes ({100*low_conf/total_notes:5.1f}%)")
    print()

    # Duration analysis
    print(f"⏱️  DURATION ANALYSIS")
    print(f"   Mean duration: {np.mean(durations):.3f}s")
    print(f"   Median duration: {np.median(durations):.3f}s")
    print(f"   Min: {np.min(durations):.3f}s, Max: {np.max(durations):.3f}s")

    # Very short notes (possible artifacts)
    very_short = sum(1 for d in durations if d <= 0.1)
    short = sum(1 for d in durations if 0.1 < d <= 0.2)
    normal = sum(1 for d in durations if d > 0.2)

    print(f"\n   Distribution:")
    print(f"   Very short (≤0.1s): {very_short:3d} notes ({100*very_short/total_notes:5.1f}%) ⚠️ Possible artifacts")
    print(f"   Short (0.1-0.2s):   {short:3d} notes ({100*short/total_notes:5.1f}%)")
    print(f"   Normal (>0.2s):     {normal:3d} notes ({100*normal/total_notes:5.1f}%)")
    print()

    # Pitch analysis
    print(f"🎵 PITCH ANALYSIS")
    pitch_names = []
    pitches = [n[midi_key] for n in notes]

    for midi in pitches:
        note_names = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
        octave = (midi // 12) - 1
        note = note_names[midi % 12]
        pitch_names.append(f"{note}{octave}")

    pitch_counter = Counter(pitch_names)
    print(f"   Unique pitches: {len(pitch_counter)}")
    print(f"   Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
    print(f"\n   Most common pitches:")
    for pitch, count in pitch_counter.most_common(8):
        print(f"      {pitch:5s}: {count:3d} times ({100*count/total_notes:5.1f}%)")
    print()

    # Check for suspicious pitches (accidentals)
    accidentals = ['C♯', 'D♯', 'F♯', 'G♯', 'A♯']
    accidental_notes = [p for p in pitch_names if any(acc in p for acc in accidentals)]

    if accidental_notes:
        print(f"   ⚠️  Accidentals detected: {len(accidental_notes)} notes ({100*len(accidental_notes)/total_notes:.1f}%)")
        acc_counter = Counter(accidental_notes)
        for pitch, count in acc_counter.most_common():
            print(f"      {pitch}: {count} times")
        print(f"   Note: Simple melodies typically don't have many accidentals")
    else:
        print(f"   ✅ No accidentals (all natural notes)")
    print()

    # Quality assessment
    print(f"🎯 QUALITY ASSESSMENT")

    quality_score = 0
    max_score = 0
    issues = []

    # Confidence score (30 points)
    max_score += 30
    if np.mean(confidences) >= 0.6:
        quality_score += 30
        print(f"   ✅ High average confidence ({np.mean(confidences):.3f})")
    elif np.mean(confidences) >= 0.4:
        quality_score += 20
        print(f"   ⚠️  Medium average confidence ({np.mean(confidences):.3f})")
        issues.append("Many notes have medium confidence")
    else:
        quality_score += 10
        print(f"   ❌ Low average confidence ({np.mean(confidences):.3f})")
        issues.append("Low confidence suggests uncertain predictions")

    # Artifact detection (20 points)
    max_score += 20
    artifact_rate = very_short / total_notes
    if artifact_rate < 0.1:
        quality_score += 20
        print(f"   ✅ Few artifacts ({100*artifact_rate:.1f}% very short notes)")
    elif artifact_rate < 0.3:
        quality_score += 10
        print(f"   ⚠️  Some artifacts ({100*artifact_rate:.1f}% very short notes)")
        issues.append(f"{very_short} very short notes might be artifacts")
    else:
        quality_score += 0
        print(f"   ❌ Many artifacts ({100*artifact_rate:.1f}% very short notes)")
        issues.append(f"{very_short} very short notes likely artifacts")

    # Pitch consistency (30 points)
    max_score += 30
    accidental_rate = len(accidental_notes) / total_notes
    if accidental_rate < 0.1:
        quality_score += 30
        print(f"   ✅ Clean pitch set ({100*accidental_rate:.1f}% accidentals)")
    elif accidental_rate < 0.3:
        quality_score += 20
        print(f"   ⚠️  Some unexpected pitches ({100*accidental_rate:.1f}% accidentals)")
        issues.append("Accidentals detected in simple melody")
    else:
        quality_score += 10
        print(f"   ❌ Many unexpected pitches ({100*accidental_rate:.1f}% accidentals)")
        issues.append("Too many accidentals for simple melody")

    # Coverage (20 points)
    max_score += 20
    if total_notes >= 20:
        quality_score += 20
        print(f"   ✅ Good coverage ({total_notes} notes)")
    elif total_notes >= 10:
        quality_score += 15
        print(f"   ⚠️  Moderate coverage ({total_notes} notes)")
    else:
        quality_score += 5
        print(f"   ❌ Sparse coverage ({total_notes} notes)")
        issues.append("Few notes detected")

    print()
    final_score = 100 * quality_score / max_score
    print(f"   {'='*50}")
    print(f"   OVERALL QUALITY: {quality_score}/{max_score} ({final_score:.1f}%)")

    if final_score >= 80:
        print(f"   ✅ EXCELLENT - System appears to be working well")
    elif final_score >= 60:
        print(f"   ⚠️  GOOD - System working but has some issues")
    elif final_score >= 40:
        print(f"   ⚠️  FAIR - Significant issues detected")
    else:
        print(f"   ❌ POOR - Major problems with predictions")
    print(f"   {'='*50}")

    if issues:
        print(f"\n   Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
    print()

    return {
        'total_notes': total_notes,
        'mean_confidence': float(np.mean(confidences)),
        'artifact_rate': float(artifact_rate),
        'accidental_rate': float(accidental_rate),
        'quality_score': final_score,
        'issues': issues
    }

def main():
    """Main function"""

    # Find JSON prediction files
    results_dir = Path('.')
    json_files = list(results_dir.glob('*_notes.json')) + list(results_dir.glob('*_predictions.json'))

    if not json_files:
        print("❌ No prediction JSON files found")
        print("   Run: python test_my_humming.py <audio> --save-json")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"ANALYZING PREDICTION QUALITY")
    print(f"{'='*70}")
    print(f"Found {len(json_files)} prediction file(s)\n")

    results = {}
    for json_file in sorted(json_files):
        predictions = load_predictions(json_file)
        notes = predictions.get('notes', [])
        audio_path = predictions.get('audio_file', None)

        # Try to find audio file if not in JSON
        if not audio_path:
            # Try common patterns
            base_name = json_file.stem.replace('_notes', '').replace('_predictions', '')
            audio_path = base_name + '.wav'

        filename = Path(audio_path).name if audio_path else json_file.stem + '.wav'

        # Compare to actual audio content
        audio_results = compare_to_audio(notes, audio_path) if audio_path else None

        # Analyze prediction quality
        quality_results = analyze_predictions(notes, filename, audio_path)

        # Combine results
        results[filename] = {
            'quality': quality_results,
            'audio_accuracy': audio_results
        }

    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}\n")

        quality_scores = [r['quality']['quality_score'] for r in results.values() if r['quality']]
        audio_accs = [r['audio_accuracy']['exact_accuracy'] for r in results.values()
                     if r['audio_accuracy']]

        if quality_scores:
            avg_quality = np.mean(quality_scores)
            print(f"Average quality score: {avg_quality:.1f}%")

        if audio_accs:
            avg_audio_acc = np.mean(audio_accs)
            print(f"Average audio accuracy (exact): {avg_audio_acc:.1f}%")

        print()

        # Decision based on audio accuracy (most important metric)
        if audio_accs:
            avg_acc = np.mean(audio_accs)
            if avg_acc >= 70:
                print(f"✅ SYSTEM IS WORKING WELL!")
                print(f"   {avg_acc:.1f}% accuracy - Ready for deployment")
            elif avg_acc >= 50:
                print(f"⚠️  SYSTEM IS FUNCTIONAL BUT NEEDS TUNING")
                print(f"   {avg_acc:.1f}% accuracy - Try adjusting parameters")
            else:
                print(f"❌ SYSTEM HAS ACCURACY ISSUES")
                print(f"   {avg_acc:.1f}% accuracy - Investigate further")
        else:
            print(f"⚠️  Could not verify against audio")
            if quality_scores and np.mean(quality_scores) >= 70:
                print(f"   Quality metrics look good, but audio verification failed")
            elif quality_scores:
                print(f"   Quality metrics suggest issues")

    print(f"\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}\n")
    print(f"1. 📊 View visualizations:")
    print(f"      Open the PNG files to see note patterns")
    print(f"")
    print(f"2. 🎵 Listen and compare:")
    print(f"      Play your recordings and check if detected notes match")
    print(f"")
    print(f"3. 📝 Create ground truth (optional):")
    print(f"      For precise metrics, create ground truth files")
    print(f"      python evaluate_my_humming.py <audio> <ground_truth>")
    print()

if __name__ == '__main__':
    main()
