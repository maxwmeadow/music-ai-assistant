#!/usr/bin/env python3
"""
Test Music Theory Integration with Hum2Melody

This script tests that the music theory post-processing works correctly
with notes from the hum2melody package.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.music_theory import (
    MusicTheoryProcessor,
    convert_hum2melody_to_internal,
    convert_internal_to_hum2melody
)


def test_note_conversion():
    """Test note format conversion functions."""
    print("\n" + "="*70)
    print("TEST 1: Note Format Conversion")
    print("="*70)

    # Hum2melody format note
    hum2melody_notes = [
        {
            'start': 0.0,
            'end': 0.5,
            'duration': 0.5,
            'midi': 60,  # C4
            'note': 'C4',
            'confidence': 0.85
        },
        {
            'start': 0.5,
            'end': 1.0,
            'duration': 0.5,
            'midi': 64,  # E4
            'note': 'E4',
            'confidence': 0.90
        }
    ]

    # Convert to internal
    internal_notes = convert_hum2melody_to_internal(hum2melody_notes)
    print(f"\nConverted {len(hum2melody_notes)} notes to internal format:")
    for i, note in enumerate(internal_notes):
        print(f"  {i+1}. pitch={note['pitch']}, start={note['start']}, "
              f"duration={note['duration']}, confidence={note['confidence']}")

    # Convert back
    back_to_hum2melody = convert_internal_to_hum2melody(internal_notes)
    print(f"\nConverted back to hum2melody format:")
    for i, note in enumerate(back_to_hum2melody):
        print(f"  {i+1}. {note['note']} (MIDI {note['midi']}), "
              f"start={note['start']}, duration={note['duration']}")

    assert len(internal_notes) == len(hum2melody_notes)
    assert internal_notes[0]['pitch'] == 60
    assert internal_notes[1]['pitch'] == 64
    print("\n[OK] Note conversion working correctly!")


def test_music_theory_processing():
    """Test full music theory processing pipeline."""
    print("\n" + "="*70)
    print("TEST 2: Music Theory Processing Pipeline")
    print("="*70)

    # Create processor (without chord analysis)
    processor = MusicTheoryProcessor(enable_chord_analysis=False)

    # Simulate hum2melody output - C major scale
    hum2melody_notes = [
        {'start': 0.0, 'end': 0.5, 'duration': 0.5, 'midi': 60, 'note': 'C4', 'confidence': 0.9},
        {'start': 0.5, 'end': 1.0, 'duration': 0.5, 'midi': 62, 'note': 'D4', 'confidence': 0.85},
        {'start': 1.0, 'end': 1.5, 'duration': 0.5, 'midi': 64, 'note': 'E4', 'confidence': 0.88},
        {'start': 1.5, 'end': 2.0, 'duration': 0.5, 'midi': 65, 'note': 'F4', 'confidence': 0.87},
        {'start': 2.0, 'end': 2.5, 'duration': 0.5, 'midi': 67, 'note': 'G4', 'confidence': 0.92},
        {'start': 2.5, 'end': 3.0, 'duration': 0.5, 'midi': 69, 'note': 'A4', 'confidence': 0.86},
        {'start': 3.0, 'end': 3.5, 'duration': 0.5, 'midi': 71, 'note': 'B4', 'confidence': 0.89},
        {'start': 3.5, 'end': 4.5, 'duration': 1.0, 'midi': 72, 'note': 'C5', 'confidence': 0.93},
    ]

    # Process
    result = processor.process(hum2melody_notes, input_format="hum2melody")

    # Check result structure
    assert 'notes' in result
    assert 'metadata' in result
    assert 'harmony' in result

    print(f"\n[Results] Processing Results:")
    print(f"  Detected key: {result['metadata']['key']}")
    print(f"  Detected tempo: {result['metadata']['tempo']:.1f} BPM")
    print(f"  Grid resolution: {result['metadata']['grid_resolution']}")
    print(f"  Output notes: {len(result['notes'])}")

    # Should detect C major
    assert 'C' in result['metadata']['key'] or 'A' in result['metadata']['key'], \
        f"Expected C major or A minor, got {result['metadata']['key']}"

    # Tempo should be reasonable (from 0.5s intervals = 120 BPM)
    assert 80 <= result['metadata']['tempo'] <= 160, \
        f"Tempo {result['metadata']['tempo']} seems unreasonable"

    # Should have chord analysis disabled
    assert result['harmony'] is None, "Chord analysis should be disabled"

    print("\n[OK] Music theory processing working correctly!")


def test_with_chord_analysis():
    """Test with chord analysis enabled."""
    print("\n" + "="*70)
    print("TEST 3: Music Theory with Chord Analysis")
    print("="*70)

    # Create processor WITH chord analysis
    processor = MusicTheoryProcessor(enable_chord_analysis=True)

    # C major arpeggio
    hum2melody_notes = [
        {'start': 0.0, 'end': 1.0, 'duration': 1.0, 'midi': 60, 'note': 'C4', 'confidence': 0.9},
        {'start': 1.0, 'end': 2.0, 'duration': 1.0, 'midi': 64, 'note': 'E4', 'confidence': 0.9},
        {'start': 2.0, 'end': 3.0, 'duration': 1.0, 'midi': 67, 'note': 'G4', 'confidence': 0.9},
        {'start': 3.0, 'end': 4.0, 'duration': 1.0, 'midi': 72, 'note': 'C5', 'confidence': 0.9},
    ]

    # Process
    result = processor.process(hum2melody_notes, input_format="hum2melody")

    # Should have chord analysis
    assert result['harmony'] is not None, "Chord analysis should be enabled"
    print(f"\n  Detected {len(result['harmony'])} chords")

    if result['harmony']:
        print("\n  Chord progression:")
        for chord in result['harmony'][:3]:  # Show first 3
            print(f"    {chord.get('roman', '?')}: {chord.get('root', '?')} "
                  f"{chord.get('quality', '?')}")

    print("\n[OK] Chord analysis working!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MUSIC THEORY + HUM2MELODY INTEGRATION TEST")
    print("="*70)

    try:
        # Test 1: Format conversion
        test_note_conversion()

        # Test 2: Main processing
        test_music_theory_processing()

        # Test 3: With chords
        test_with_chord_analysis()

        print("\n" + "="*70)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("="*70)
        print("\nThe music_theory module is ready to use with hum2melody!")
        print("="*70 + "\n")

        return 0

    except AssertionError as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
