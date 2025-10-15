"""
Integration test for BeatBox2Drums model and dataset.

Tests that the model and dataset work together correctly:
- Dataset loads and returns correct shapes
- Model accepts dataset output
- Forward pass produces expected output shape
- End-to-end pipeline works
"""

import torch
import numpy as np
import json
import tempfile
from pathlib import Path
import soundfile as sf

# Import your implementations
from models.beatbox2drums_model import BeatBox2DrumsCNN
from data.drum_dataset import DrumDataset


def create_test_data():
    """Create temporary test audio and labels."""
    tmpdir = tempfile.mkdtemp()
    tmpdir = Path(tmpdir)
    
    # Create dummy audio file (white noise simulating drums)
    audio_path = tmpdir / 'test_drums.wav'
    sr = 16000
    duration = 2.0
    audio = np.random.randn(int(sr * duration)) * 0.1
    
    sf.write(audio_path, audio, sr)
    
    # Create dummy label in Ayaan's format
    label = {
        'audio_path': str(audio_path),
        'duration': duration,
        'drum_hits': {
            'kick': [
                {'time': 0.0, 'velocity': 0.8},
                {'time': 0.5, 'velocity': 0.7},
                {'time': 1.0, 'velocity': 0.9},
                {'time': 1.5, 'velocity': 0.85}
            ],
            'snare': [
                {'time': 0.25, 'velocity': 0.9},
                {'time': 0.75, 'velocity': 0.85},
                {'time': 1.25, 'velocity': 0.88}
            ],
            'hihat': [
                {'time': 0.125, 'velocity': 0.6},
                {'time': 0.375, 'velocity': 0.65},
                {'time': 0.625, 'velocity': 0.7},
                {'time': 0.875, 'velocity': 0.6},
                {'time': 1.125, 'velocity': 0.62},
                {'time': 1.375, 'velocity': 0.68}
            ]
        },
        'total_hits': 13
    }
    
    label_path = tmpdir / 'test_label.json'
    with open(label_path, 'w') as f:
        json.dump([label], f)
    
    return tmpdir, label_path


def test_model_architecture():
    """Test model architecture and forward pass."""
    print("\n=== Testing Model Architecture ===")
    
    # Create model
    model = BeatBox2DrumsCNN()
    print(f"✓ Model created")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 500, 128)
    print(f"✓ Created dummy input: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 125, 4)")
    
    # Verify shape
    assert output.shape == (batch_size, 125, 4), f"Shape mismatch! Got {output.shape}"
    print(f"✓ Output shape correct")
    
    # Check output range (should be logits, not probabilities)
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return model


def test_dataset_loading():
    """Test dataset loads correctly."""
    print("\n=== Testing Dataset Loading ===")
    
    # Create test data
    tmpdir, label_path = create_test_data()
    
    try:
        # Create dataset
        dataset = DrumDataset(
            labels_path=str(label_path),
            validate=True
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Load one sample
        mel, target = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Mel shape: {mel.shape}")
        print(f"  Target shape: {target.shape}")
        
        # Verify shapes
        assert mel.shape == (1, 500, 128), f"Mel shape mismatch! Got {mel.shape}"
        assert target.shape == (500, 4), f"Target shape mismatch! Got {target.shape}"
        print(f"✓ Shapes correct")
        
        # Analyze targets
        kick_frames = (target[:, 0] == 1.0).sum()
        snare_frames = (target[:, 1] == 1.0).sum()
        hihat_frames = (target[:, 2] == 1.0).sum()
        silence_frames = (target[:, 3] == 1.0).sum()
        
        print(f"  Kick frames: {kick_frames}")
        print(f"  Snare frames: {snare_frames}")
        print(f"  Hihat frames: {hihat_frames}")
        print(f"  Silence frames: {silence_frames}")
        
        # Verify at least some hits detected
        assert kick_frames >= 3, f"Expected >= 3 kick frames, got {kick_frames}"
        assert snare_frames >= 2, f"Expected >= 2 snare frames, got {snare_frames}"
        assert hihat_frames >= 4, f"Expected >= 4 hihat frames, got {hihat_frames}"
        print(f"✓ Target encoding correct")
        
        return dataset, mel, target
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)


def test_model_dataset_integration():
    """Test that model and dataset work together."""
    print("\n=== Testing Model + Dataset Integration ===")
    
    # Create test data
    tmpdir, label_path = create_test_data()
    
    try:
        # Create dataset
        dataset = DrumDataset(labels_path=str(label_path))
        print(f"✓ Dataset created")
        
        # Create model
        model = BeatBox2DrumsCNN()
        model.eval()
        print(f"✓ Model created")
        
        # Get sample from dataset
        mel, target = dataset[0]
        print(f"✓ Loaded sample from dataset")
        print(f"  Mel shape: {mel.shape}")
        print(f"  Target shape: {target.shape}")
        
        # Prepare batch (model expects batch dimension)
        mel_batch = mel.unsqueeze(0)  # (1, 1, 500, 128)
        print(f"  Batched mel shape: {mel_batch.shape}")
        
        # Forward pass through model
        with torch.no_grad():
            output = model(mel_batch)
        
        print(f"✓ Forward pass successful")
        print(f"  Model output shape: {output.shape}")
        
        # Verify output shape
        assert output.shape == (1, 125, 4), f"Output shape mismatch! Got {output.shape}"
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Check that we can threshold predictions
        threshold = 0.5
        predictions = (probs > threshold).float()
        
        # Count predicted hits per drum type
        pred_kicks = predictions[0, :, 0].sum()
        pred_snares = predictions[0, :, 1].sum()
        pred_hihats = predictions[0, :, 2].sum()
        
        print(f"✓ Predictions (threshold={threshold}):")
        print(f"  Predicted kicks: {pred_kicks:.0f}")
        print(f"  Predicted snares: {pred_snares:.0f}")
        print(f"  Predicted hihats: {pred_hihats:.0f}")
        
        # Note: Model outputs 125 frames, but targets are 500 frames
        # In training, we'll need to downsample targets to match
        print(f"\n   Note: Model outputs 125 frames, targets are 500 frames")
        print(f"     Training script must downsample targets: target[:, ::4, :]")
        
        # Test downsampling
        target_downsampled = target[::4, :]  # Take every 4th frame
        print(f"  Downsampled target shape: {target_downsampled.shape}")
        assert target_downsampled.shape[0] == 125, "Downsampled target should have 125 frames"
        print(f"✓ Target downsampling works correctly")
        
        return True
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)


def test_batch_processing():
    """Test processing multiple samples in a batch."""
    print("\n=== Testing Batch Processing ===")
    
    # Create model
    model = BeatBox2DrumsCNN()
    model.eval()
    
    # Create batch of dummy data
    batch_size = 4
    mel_batch = torch.randn(batch_size, 1, 500, 128)
    
    print(f"✓ Created batch: {mel_batch.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(mel_batch)
    
    print(f"✓ Batch forward pass successful")
    print(f"  Output shape: {output.shape}")
    
    # Verify shape
    assert output.shape == (batch_size, 125, 4), f"Batch output shape mismatch! Got {output.shape}"
    print(f"✓ Batch processing works correctly")
    
    return True


def main():
    """Run all integration tests."""
    print("="*60)
    print("BeatBox2Drums Model + Dataset Integration Tests")
    print("="*60)
    
    try:
        # Test model
        model = test_model_architecture()
        
        # Test dataset
        dataset, mel, target = test_dataset_loading()
        
        # Test integration
        test_model_dataset_integration()
        
        # Test batch processing
        test_batch_processing()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)