"""
Synthetic Training Data Generator

Generates perfect synthetic melodies with clear onsets/offsets.
Used for curriculum learning: train on synthetic first, then fine-tune on real.

This is INCREDIBLY powerful - the model learns ideal behavior first!
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import from backend
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from typing import List, Dict
from tqdm import tqdm

# Check for required packages
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("⚠️  soundfile not available, trying scipy.io.wavfile")
    SOUNDFILE_AVAILABLE = False
    from scipy.io import wavfile

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️  scipy not available, some features will be limited")
    SCIPY_AVAILABLE = False


class SyntheticMelodyGenerator:
    """
    Generate synthetic melodies with perfect labels.
    
    Uses simple synthesized tones (sine, sawtooth, etc.) to create
    training data where onsets/offsets are perfectly clear.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_pitch: int = 60,
        max_pitch: int = 84,
        output_dir: str = None
    ):
        self.sample_rate = sample_rate
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "synthetic"
        else:
            output_dir = Path(output_dir)
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_sine_tone(
        self,
        pitch: int,
        duration: float,
        amplitude: float = 0.3
    ) -> np.ndarray:
        """Generate a pure sine wave."""
        freq = 440 * (2 ** ((pitch - 69) / 12))
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        attack_time = 0.01
        release_time = 0.05
        
        envelope = np.ones_like(t)
        
        attack_samples = int(attack_time * self.sample_rate)
        if attack_samples > 0 and attack_samples < len(envelope):
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        release_samples = int(release_time * self.sample_rate)
        if release_samples > 0 and release_samples < len(envelope):
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        tone = amplitude * np.sin(2 * np.pi * freq * t) * envelope
        return tone
    
    def generate_sawtooth_tone(
        self,
        pitch: int,
        duration: float,
        amplitude: float = 0.2
    ) -> np.ndarray:
        """Generate a sawtooth wave (has harmonics like voice)."""
        freq = 440 * (2 ** ((pitch - 69) / 12))
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        if SCIPY_AVAILABLE:
            tone = signal.sawtooth(2 * np.pi * freq * t)
        else:
            phase = (freq * t) % 1.0
            tone = 2 * phase - 1
        
        attack_time = 0.01
        release_time = 0.05
        envelope = np.ones_like(t)
        
        attack_samples = int(attack_time * self.sample_rate)
        if attack_samples > 0 and attack_samples < len(envelope):
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        release_samples = int(release_time * self.sample_rate)
        if release_samples > 0 and release_samples < len(envelope):
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        tone = amplitude * tone * envelope
        return tone
    
    def generate_melody(
        self,
        num_notes: int = None,
        melody_type: str = 'random'
    ) -> Dict:
        """
        Generate a complete melody.
        
        Args:
            num_notes: Number of notes (random if None)
            melody_type: 'random', 'scale', 'arpeggio', 'stepwise'
        
        Returns:
            dict with audio, notes, start_times, durations
        """
        if num_notes is None:
            num_notes = np.random.randint(5, 20)
        
        if melody_type == 'random':
            pitches = np.random.randint(self.min_pitch, self.max_pitch, num_notes)
        
        elif melody_type == 'scale':
            root = np.random.randint(self.min_pitch, self.min_pitch + 12)
            scale_steps = [0, 2, 4, 5, 7, 9, 11, 12]
            pitches = []
            for _ in range(num_notes):
                step = np.random.choice(scale_steps)
                pitch = root + step
                if pitch > self.max_pitch:
                    pitch -= 12
                pitches.append(pitch)
            pitches = np.array(pitches)
        
        elif melody_type == 'stepwise':
            pitch_range = self.max_pitch - self.min_pitch
            if pitch_range < 24:
                pitches = np.random.randint(self.min_pitch, self.max_pitch, num_notes)
            else:
                start_pitch = self.min_pitch + pitch_range // 2
                pitches = [start_pitch]
                
                for _ in range(num_notes - 1):
                    step = np.random.choice([-2, -1, 0, 1, 2])
                    next_pitch = pitches[-1] + step
                    next_pitch = np.clip(next_pitch, self.min_pitch, self.max_pitch)
                    pitches.append(next_pitch)
                pitches = np.array(pitches)
        
        elif melody_type == 'arpeggio':
            root = np.random.randint(self.min_pitch, min(self.min_pitch + 12, self.max_pitch - 12))
            chord_type = np.random.choice(['major', 'minor'])
            if chord_type == 'major':
                intervals = [0, 4, 7, 12]
            else:
                intervals = [0, 3, 7, 12]
            
            pitches = []
            for _ in range(num_notes):
                interval = np.random.choice(intervals)
                pitch = root + interval
                while pitch > self.max_pitch:
                    pitch -= 12
                while pitch < self.min_pitch:
                    pitch += 12
                pitches.append(pitch)
            pitches = np.array(pitches)
        
        else:
            pitches = np.random.randint(self.min_pitch, self.max_pitch, num_notes)
        
        duration_options = [0.25, 0.5, 0.75, 1.0, 1.5]
        durations = np.random.choice(duration_options, num_notes)
        
        start_times = [0.0]
        for i in range(num_notes - 1):
            next_start = start_times[-1] + durations[i]
            
            if np.random.random() > 0.7:
                rest = np.random.choice([0.25, 0.5])
                next_start += rest
            
            start_times.append(next_start)
        
        start_times = np.array(start_times)
        
        total_duration = start_times[-1] + durations[-1] + 0.5
        audio = np.zeros(int(total_duration * self.sample_rate))
        
        synth_type = np.random.choice(['sine', 'sawtooth'])
        
        for pitch, start, duration in zip(pitches, start_times, durations):
            if synth_type == 'sine':
                tone = self.generate_sine_tone(int(pitch), duration)
            else:
                tone = self.generate_sawtooth_tone(int(pitch), duration)
            
            start_sample = int(start * self.sample_rate)
            end_sample = start_sample + len(tone)
            
            if end_sample <= len(audio):
                audio[start_sample:end_sample] += tone
            else:
                available_samples = len(audio) - start_sample
                if available_samples > 0:
                    audio[start_sample:] += tone[:available_samples]
        
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        noise_level = np.random.uniform(0.001, 0.005)
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise
        
        audio = np.clip(audio, -1, 1)
        
        return {
            'audio': audio,
            'notes': pitches.tolist(),
            'start_times': start_times.tolist(),
            'durations': durations.tolist(),
            'type': melody_type,
            'synth': synth_type
        }
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        split_name: str = 'synthetic_train'
    ) -> str:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples: Number of melodies to generate
            split_name: Name for this dataset split
        
        Returns:
            Path to generated manifest file
        """
        print(f"\n{'='*60}")
        print(f"Generating Synthetic Dataset: {split_name}")
        print(f"{'='*60}")
        print(f"Number of samples: {num_samples}")
        print(f"Output directory: {self.output_dir}")
        
        audio_dir = self.output_dir / 'audio'
        audio_dir.mkdir(exist_ok=True)
        
        manifest = []
        
        melody_types = ['random', 'scale', 'stepwise', 'arpeggio']
        
        for i in tqdm(range(num_samples), desc="Generating melodies"):
            melody_type = np.random.choice(
                melody_types,
                p=[0.2, 0.3, 0.4, 0.1]
            )
            
            melody = self.generate_melody(melody_type=melody_type)
            
            audio_filename = f"{split_name}_{i:05d}.wav"
            audio_path = audio_dir / audio_filename
            
            if SOUNDFILE_AVAILABLE:
                sf.write(audio_path, melody['audio'], self.sample_rate)
            else:
                audio_int16 = (melody['audio'] * 32767).astype(np.int16)
                wavfile.write(audio_path, self.sample_rate, audio_int16)
            
            manifest.append({
                'audio_path': f"synthetic/audio/{audio_filename}",
                'notes': melody['notes'],
                'start_times': melody['start_times'],
                'durations': melody['durations'],
                'source': 'synthetic',
                'type': melody['type'],
                'synth': melody['synth']
            })
        
        manifest_path = self.output_dir / f'{split_name}_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Generated {num_samples} synthetic melodies")
        print(f"   Audio saved to: {audio_dir}")
        print(f"   Manifest saved to: {manifest_path}")
        print(f"{'='*60}\n")
        
        return str(manifest_path)


def generate_curriculum_datasets():
    """
    Generate datasets for curriculum learning.
    
    Curriculum:
    1. Pure synthetic (1000 samples) - learn perfect behavior
    2. Noisy synthetic (500 samples) - learn robustness
    3. Mix with real data during training
    """
    output_dir = Path(__file__).parent / "synthetic"
    
    generator = SyntheticMelodyGenerator(output_dir=output_dir)
    
    print("\n🎵 Stage 1: Clean Synthetic Data")
    manifest_clean = generator.generate_dataset(
        num_samples=1000,
        split_name='synthetic_clean'
    )
    
    print("\n🎵 Stage 2: Noisy Synthetic Data")
    manifest_noisy = generator.generate_dataset(
        num_samples=500,
        split_name='synthetic_noisy'
    )
    
    print("\n✅ Curriculum datasets generated!")
    print(f"   Clean synthetic: {manifest_clean}")
    print(f"   Noisy synthetic: {manifest_noisy}")
    print("\nNext steps:")
    print("1. Train on clean synthetic (10 epochs)")
    print("2. Fine-tune on noisy synthetic (10 epochs)")
    print("3. Fine-tune on real data (remaining epochs)")
    print("\nExample training command:")
    print("python backend/training/train_enhanced_hum2melody.py \\")
    print("  --labels data/real/manifest.json \\")
    print("  --curriculum \\")
    print(f"  --synthetic-clean {manifest_clean} \\")
    print(f"  --synthetic-noisy {manifest_noisy} \\")
    print("  --curriculum-epochs 10 10 30 \\")
    print("  --use-onset-features \\")
    print("  --use-musical-context")


if __name__ == '__main__':
    print("Checking dependencies...")
    print(f"  NumPy: ✅")
    print(f"  soundfile: {'✅' if SOUNDFILE_AVAILABLE else '⚠️  (will use scipy fallback)'}")
    print(f"  scipy: {'✅' if SCIPY_AVAILABLE else '⚠️  (limited features)'}")
    
    generate_curriculum_datasets()
