"""
Test script for audio processing endpoints.
Run this after starting the backend server.

Usage:
    python test_audio_endpoints.py
"""

import requests
import json
from pathlib import Path
import wave
import struct
import math

BASE_URL = "http://localhost:8000"


def generate_test_audio(filename: str, duration: float = 2.0, frequency: float = 440.0):
    """
    Generate a simple sine wave audio file for testing.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        frequency: Frequency in Hz (A4 = 440Hz)
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    
    # Generate sine wave
    samples = []
    for i in range(num_samples):
        value = math.sin(2 * math.pi * frequency * i / sample_rate)
        samples.append(int(value * 32767))  # 16-bit audio
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for sample in samples:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated test audio: {filename}")


def test_health():
    """Test health endpoint"""
    print("\n=== Testing /health ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_stats():
    """Test stats endpoint"""
    print("\n=== Testing /stats ===")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_hum2melody():
    """Test hum2melody endpoint"""
    print("\n=== Testing /hum2melody ===")
    
    # Generate test audio
    test_file = "test_hum.wav"
    generate_test_audio(test_file, duration=2.0, frequency=440.0)
    
    try:
        # Upload audio
        with open(test_file, 'rb') as f:
            files = {'audio': ('test_hum.wav', f, 'audio/wav')}
            data = {'save_training_data': 'true'}
            
            response = requests.post(
                f"{BASE_URL}/hum2melody",
                files=files,
                data=data
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response:")
            print(f"  Status: {result['status']}")
            print(f"  Audio ID: {result['audio_id']}")
            print(f"  Duration: {result['metadata']['duration']:.2f}s")
            print(f"  Num notes: {result['metadata']['num_notes']}")
            
            # Print first few notes
            if result['ir']['tracks']:
                notes = result['ir']['tracks'][0].get('notes', [])
                if notes:
                    print(f"  First 3 notes: {notes[:3]}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


def test_beatbox2drums():
    """Test beatbox2drums endpoint"""
    print("\n=== Testing /beatbox2drums ===")
    
    # Generate test audio with different frequencies (simulating different drum sounds)
    test_file = "test_beatbox.wav"
    generate_test_audio(test_file, duration=2.0, frequency=200.0)
    
    try:
        # Upload audio
        with open(test_file, 'rb') as f:
            files = {'audio': ('test_beatbox.wav', f, 'audio/wav')}
            data = {'save_training_data': 'true'}
            
            response = requests.post(
                f"{BASE_URL}/beatbox2drums",
                files=files,
                data=data
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response:")
            print(f"  Status: {result['status']}")
            print(f"  Audio ID: {result['audio_id']}")
            print(f"  Duration: {result['metadata']['duration']:.2f}s")
            print(f"  Tempo: {result['metadata']['tempo']:.1f} BPM")
            print(f"  Num samples: {result['metadata']['num_samples']}")
            
            # Print first few drum hits
            if result['ir']['tracks']:
                samples = result['ir']['tracks'][0].get('samples', [])
                if samples:
                    print(f"  First 3 samples: {samples[:3]}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    finally:
        # Cleanup
        Path(test_file).unlink(missing_ok=True)


def test_arrange():
    """Test arrange endpoint"""
    print("\n=== Testing /arrange ===")
    
    # Create simple IR with just a melody
    test_ir = {
        "metadata": {
            "tempo": 120,
            "key": "C",
            "time_signature": "4/4"
        },
        "tracks": [
            {
                "id": "melody",
                "instrument": "lead_synth",
                "notes": [
                    {"pitch": 60, "duration": 1.0, "velocity": 0.8},
                    {"pitch": 62, "duration": 1.0, "velocity": 0.7},
                    {"pitch": 64, "duration": 1.0, "velocity": 0.8},
                    {"pitch": 65, "duration": 1.0, "velocity": 0.7}
                ]
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/arrange",
        json={"ir": test_ir, "style": "pop"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response:")
        print(f"  Status: {result['status']}")
        print(f"  Original tracks: {result['metadata']['original_tracks']}")
        print(f"  Total tracks: {result['metadata']['total_tracks']}")
        print(f"  Added tracks: {result['metadata']['added_tracks']}")
        print(f"  Style: {result['metadata']['style']}")
        
        # Print track names
        track_ids = [track['id'] for track in result['ir']['tracks']]
        print(f"  Track IDs: {track_ids}")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_feedback():
    """Test feedback endpoint"""
    print("\n=== Testing /feedback ===")
    
    # First, we need an audio_id from a previous test
    # For simplicity, let's try with audio_id=1
    response = requests.post(
        f"{BASE_URL}/feedback",
        data={
            "audio_id": 1,
            "rating": 5,
            "feedback_text": "Great melody generation!"
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
        return True
    else:
        print(f"Error: {response.text}")
        print("Note: This might fail if no audio samples exist yet")
        return False


def run_all_tests():
    """Run all endpoint tests"""
    print("=" * 60)
    print("Testing Audio Processing Endpoints")
    print("=" * 60)
    
    results = {
        "health": test_health(),
        "stats": test_stats(),
        "hum2melody": test_hum2melody(),
        "beatbox2drums": test_beatbox2drums(),
        "arrange": test_arrange(),
        "feedback": test_feedback()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} : {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)