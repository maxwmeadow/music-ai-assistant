"""
Usage examples for AudioProcessor and ModelServer classes.
Demonstrates how to use the components directly in Python.
"""

import asyncio
from pathlib import Path
from backend.audio_processor import AudioProcessor
from backend.model_server import ModelServer
from backend.database import TrainingDataDB


async def example_melody_processing():
    """Example: Process humming audio and generate melody"""
    print("=== Example: Melody Generation ===\n")
    
    # Initialize components
    processor = AudioProcessor(target_sr=16000)
    model = ModelServer()
    
    # Load audio file (replace with your audio file)
    audio_file = Path("your_humming.wav")
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        print("Please record some humming and save as 'your_humming.wav'")
        return
    
    # Read audio bytes
    audio_bytes = audio_file.read_bytes()
    
    # Process audio
    print("Processing audio...")
    features = processor.preprocess_for_hum2melody(audio_bytes)
    
    print(f"Extracted features:")
    print(f"  Duration: {features['duration']:.2f} seconds")
    print(f"  Sample rate: {features['sample_rate']} Hz")
    print(f"  Mel spectrogram shape: {features['mel_spectrogram'].shape}")
    print(f"  Detected {len(features['onset_times'])} note onsets")
    print(f"  Onset times: {features['onset_times'][:5]}...")
    
    # Generate melody prediction
    print("\nGenerating melody...")
    melody_track = await model.predict_melody(features)
    
    print(f"Generated melody track:")
    print(f"  Track ID: {melody_track.id}")
    print(f"  Instrument: {melody_track.instrument}")
    print(f"  Number of notes: {len(melody_track.notes) if melody_track.notes else 0}")
    
    if melody_track.notes:
        print(f"\nFirst 3 notes:")
        for i, note in enumerate(melody_track.notes[:3]):
            print(f"    Note {i+1}: Pitch={note.pitch}, Duration={note.duration:.2f}s, Velocity={note.velocity:.2f}")


async def example_drum_processing():
    """Example: Process beatbox audio and generate drum pattern"""
    print("\n=== Example: Drum Generation ===\n")
    
    # Initialize components
    processor = AudioProcessor(target_sr=16000)
    model = ModelServer()
    
    # Load audio file (replace with your audio file)
    audio_file = Path("your_beatbox.wav")
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        print("Please record some beatboxing and save as 'your_beatbox.wav'")
        return
    
    # Read audio bytes
    audio_bytes = audio_file.read_bytes()
    
    # Process audio
    print("Processing audio...")
    features = processor.preprocess_for_beatbox(audio_bytes)
    
    print(f"Extracted features:")
    print(f"  Duration: {features['duration']:.2f} seconds")
    print(f"  Sample rate: {features['sample_rate']} Hz")
    print(f"  Estimated tempo: {features['tempo']:.1f} BPM")
    print(f"  Detected {len(features['onset_times'])} onsets")
    print(f"  MFCC shape: {len(features['mfcc'])} coefficients")
    
    # Generate drum prediction
    print("\nGenerating drum pattern...")
    drums_track = await model.predict_drums(features)
    
    print(f"Generated drum track:")
    print(f"  Track ID: {drums_track.id}")
    print(f"  Number of samples: {len(drums_track.samples) if drums_track.samples else 0}")
    
    if drums_track.samples:
        print(f"\nFirst 5 drum hits:")
        for i, sample in enumerate(drums_track.samples[:5]):
            print(f"    {i+1}. {sample.sample} at {sample.start:.2f}s")


async def example_arrangement():
    """Example: Arrange a melody with accompaniment"""
    print("\n=== Example: Arrangement ===\n")
    
    from backend.schemas import IR, Track, Note
    
    # Create a simple melody IR
    melody_notes = [
        Note(pitch=60, duration=1.0, velocity=0.8),
        Note(pitch=62, duration=1.0, velocity=0.7),
        Note(pitch=64, duration=1.0, velocity=0.8),
        Note(pitch=65, duration=1.0, velocity=0.7)
    ]
    
    melody_track = Track(
        id="melody",
        instrument="lead_synth",
        notes=melody_notes
    )
    
    original_ir = IR(
        metadata={"tempo": 120, "key": "C"},
        tracks=[melody_track]
    )
    
    print(f"Original IR:")
    print(f"  Tracks: {len(original_ir.tracks)}")
    print(f"  Track IDs: {[t.id for t in original_ir.tracks]}")
    
    # Arrange it
    model = ModelServer()
    enhanced_ir = await model.arrange_track(original_ir, style="pop")
    
    print(f"\nEnhanced IR:")
    print(f"  Tracks: {len(enhanced_ir.tracks)}")
    print(f"  Track IDs: {[t.id for t in enhanced_ir.tracks]}")
    
    # Show what was added
    for track in enhanced_ir.tracks:
        print(f"\n  Track: {track.id}")
        if track.instrument:
            print(f"    Instrument: {track.instrument}")
        if track.notes:
            print(f"    Notes: {len(track.notes)}")
        if track.samples:
            print(f"    Samples: {len(track.samples)}")


def example_database_usage():
    """Example: Database operations"""
    print("\n=== Example: Database Operations ===\n")
    
    # Initialize database
    db = TrainingDataDB()
    
    # Save an audio sample
    audio_id = db.save_audio_sample(
        file_path="example_audio/hum_001.wav",
        model_type="hum2melody",
        file_format="wav",
        sample_rate=16000,
        duration=2.5,
        metadata={
            "user_id": "test_user",
            "recording_quality": "good"
        }
    )
    
    print(f"Saved audio sample with ID: {audio_id}")
    
    # Save a prediction
    prediction_data = {
        "notes": [
            {"pitch": 60, "duration": 0.5},
            {"pitch": 62, "duration": 0.5}
        ]
    }
    
    prediction_id = db.save_prediction(
        audio_sample_id=audio_id,
        model_type="hum2melody",
        prediction=prediction_data
    )
    
    print(f"Saved prediction with ID: {prediction_id}")
    
    # Save feedback
    feedback_id = db.save_feedback(
        audio_sample_id=audio_id,
        rating=5,
        prediction_id=prediction_id,
        feedback_text="Great melody generation!"
    )
    
    print(f"Saved feedback with ID: {feedback_id}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(f"  Samples by type: {stats['samples_by_type']}")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Average rating: {stats['average_rating']}")
    print(f"  Total feedback: {stats['total_feedback']}")
    
    # Get samples for a specific model
    samples = db.get_samples_by_model_type("hum2melody", limit=5)
    print(f"\nRecent hum2melody samples: {len(samples)}")
    for sample in samples:
        print(f"  ID: {sample['id']}, Duration: {sample['duration']}s, Created: {sample['created_at']}")


def example_audio_generation():
    """Example: Generate test audio programmatically"""
    print("\n=== Example: Generate Test Audio ===\n")
    
    import numpy as np
    import soundfile as sf
    
    # Generate a simple melody (C major scale)
    sample_rate = 16000
    duration = 2.0  # seconds
    
    # Notes in C major scale (MIDI numbers)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    
    # Convert MIDI to frequency
    def midi_to_freq(midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))
    
    # Generate audio
    audio = np.zeros(int(sample_rate * duration))
    note_duration = duration / len(notes)
    
    for i, midi in enumerate(notes):
        freq = midi_to_freq(midi)
        start_sample = int(i * note_duration * sample_rate)
        end_sample = int((i + 1) * note_duration * sample_rate)
        
        # Generate sine wave for this note
        t = np.arange(end_sample - start_sample) / sample_rate
        note_audio = np.sin(2 * np.pi * freq * t)
        
        # Apply envelope (fade in/out)
        envelope = np.ones_like(note_audio)
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        note_audio *= envelope
        
        audio[start_sample:end_sample] = note_audio
    
    # Save audio
    output_file = "generated_melody.wav"
    sf.write(output_file, audio, sample_rate)
    print(f"Generated test audio: {output_file}")
    print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")


async def run_all_examples():
    """Run all examples"""
    print("=" * 60)
    print("Audio Processing Examples")
    print("=" * 60)
    
    # Generate test audio first
    example_audio_generation()
    
    # Database examples (don't require audio files)
    example_database_usage()
    
    # Arrangement example (doesn't require audio files)
    await example_arrangement()
    
    # Audio processing examples (require audio files)
    print("\n" + "=" * 60)
    print("Note: The following examples require audio files.")
    print("Please record 'your_humming.wav' and 'your_beatbox.wav'")
    print("or modify the file paths in the code.")
    print("=" * 60)
    
    await example_melody_processing()
    await example_drum_processing()


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())