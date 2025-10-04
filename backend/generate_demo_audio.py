"""Generate demo audio files that persist for testing."""
import wave
import struct
import math

def generate_test_audio(filename: str, duration: float = 2.0, frequency: float = 440.0):
    """Generate a simple sine wave audio file."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    
    samples = []
    for i in range(num_samples):
        value = math.sin(2 * math.pi * frequency * i / sample_rate)
        samples.append(int(value * 32767))
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for sample in samples:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Created: {filename}")

if __name__ == "__main__":
    # Generate demo files
    generate_test_audio("demo_hum.wav", duration=2.0, frequency=440.0)
    generate_test_audio("demo_beatbox.wav", duration=2.0, frequency=200.0)
    print("\nDemo files created! Use these for testing.")