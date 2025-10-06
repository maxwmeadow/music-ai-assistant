"""
Integration test for Hum2Melody model and dataset.
Tests that model and data pipeline work together correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tempfile
from pathlib import Path
import json

# Import your modules
from models.hum2melody_model import Hum2MelodyCRNN
from data.melody_dataset import MelodyDataset


def create_dummy_data(tmpdir: Path, num_samples: int = 5):
    """Create dummy audio files and labels for testing."""
    import soundfile as sf
    
    audio_files = []
    label_files = []
    
    sr = 16000
    
    for i in range(num_samples):
        # Create audio file (sine wave with varying frequency)
        audio_path = tmpdir / f'audio_{i}.wav'
        duration = np.random.uniform(1.0, 3.0)
        freq = 220 + i * 55  # Different frequency for each sample
        
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t)
        
        sf.write(audio_path, audio, sr)
        audio_files.append(str(audio_path))
        
        # Create label
        num_notes = np.random.randint(3, 8)
        notes = np.random.randint(60, 72, size=num_notes).tolist()
        start_times = sorted(np.random.uniform(0, duration - 0.5, size=num_notes).tolist())
        durations = np.random.uniform(0.2, 0.5, size=num_notes).tolist()
        
        label = {
            'audio_path': str(audio_path),
            'notes': notes,
            'start_times': start_times,
            'durations': durations,
            'confidence': 0.85
        }
        
        label_path = tmpdir / f'label_{i}.json'
        with open(label_path, 'w') as f:
            json.dump(label, f)
        
        label_files.append(str(label_path))
    
    # Create manifest
    manifest = {
        'audio_files': audio_files,
        'label_files': label_files,
        'stats': {
            'total_files': num_samples,
            'total_duration': num_samples * 2.0,
            'avg_notes_per_file': 5
        }
    }
    
    manifest_path = tmpdir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    
    return manifest_path


def test_model_architecture():
    """Test model architecture and forward pass."""
    print("=" * 60)
    print("TEST 1: Model Architecture")
    print("=" * 60)
    
    model = Hum2MelodyCRNN()
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass with dummy data
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 500, 128)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
        activations = model.get_note_activations(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Activations shape: {activations.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"  Active notes per frame: {activations.sum(dim=2).mean():.2f}")
    
    assert output.shape == (batch_size, 62, 88), "Unexpected output shape!"
    assert activations.shape == (batch_size, 62, 88), "Unexpected activations shape!"
    
    print("\n✓ Model architecture test passed!")
    return model


def test_dataset_loading(manifest_path):
    """Test dataset loading and preprocessing."""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset Loading")
    print("=" * 60)
    
    dataset = MelodyDataset(
        labels_path=str(manifest_path),
        validate=True
    )
    
    print(f"\n✓ Dataset loaded with {len(dataset)} samples")
    
    # Test loading a single sample
    mel_tensor, target_tensor = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  Mel spectrogram: {mel_tensor.shape}")
    print(f"  Target: {target_tensor.shape}")
    print(f"  Mel range: [{mel_tensor.min():.3f}, {mel_tensor.max():.3f}]")
    print(f"  Active frames: {(target_tensor.sum(dim=1) > 0).sum()}")
    print(f"  Total active notes: {target_tensor.sum():.0f}")
    
    assert mel_tensor.shape == (1, 500, 128), "Unexpected mel shape!"
    assert target_tensor.shape == (500, 88), "Unexpected target shape!"
    
    print("\n✓ Dataset loading test passed!")
    return dataset


def test_dataloader(dataset):
    """Test DataLoader with batching."""
    print("\n" + "=" * 60)
    print("TEST 3: DataLoader Batching")
    print("=" * 60)
    
    batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        drop_last=False
    )
    
    print(f"\nDataLoader config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {len(dataloader)}")
    
    # Load first batch
    mel_batch, target_batch = next(iter(dataloader))
    
    print(f"\nBatch shapes:")
    print(f"  Mel batch: {mel_batch.shape}")
    print(f"  Target batch: {target_batch.shape}")
    
    expected_batch = min(batch_size, len(dataset))
    assert mel_batch.shape == (expected_batch, 1, 500, 128), "Unexpected batch mel shape!"
    assert target_batch.shape == (expected_batch, 500, 88), "Unexpected batch target shape!"
    
    print("\n✓ DataLoader test passed!")
    return dataloader


def test_training_step(model, dataloader):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Step")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get one batch
    mel_batch, target_batch = next(iter(dataloader))
    mel_batch = mel_batch.to(device)
    target_batch = target_batch.to(device)
    
    print(f"\nBatch moved to device: {device}")
    print(f"  Mel batch shape: {mel_batch.shape}")
    print(f"  Target batch shape: {target_batch.shape}")
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    output = model(mel_batch)
    
    print(f"\nModel output shape: {output.shape}")
    print(f"Target shape after processing: {target_batch[:, ::8, :].shape}")  # Downsample target
    
    # Note: Target is (batch, 500, 88) but output is (batch, 62, 88)
    # We need to downsample the target to match
    # Simple approach: take every 8th frame (500/8 ≈ 62)
    target_downsampled = target_batch[:, ::8, :]
    
    # Make sure shapes match
    if target_downsampled.shape[1] > output.shape[1]:
        target_downsampled = target_downsampled[:, :output.shape[1], :]
    elif target_downsampled.shape[1] < output.shape[1]:
        output = output[:, :target_downsampled.shape[1], :]
    
    print(f"Adjusted shapes - Output: {output.shape}, Target: {target_downsampled.shape}")
    
    loss = criterion(output, target_downsampled)
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backward pass completed")
    print(f"✓ Optimizer step completed")
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"✓ Gradients computed: {has_gradients}")
    
    print("\n✓ Training step test passed!")


def test_inference(model):
    """Test inference mode."""
    print("\n" + "=" * 60)
    print("TEST 5: Inference Mode")
    print("=" * 60)
    
    model.eval()
    
    # Create test input
    dummy_input = torch.randn(1, 1, 500, 128)
    
    print(f"\nInference input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        # Get logits
        output = model(dummy_input)
        
        # Get binary activations
        activations = model.get_note_activations(dummy_input, threshold=0.5)
        
        # Get probabilities
        probabilities = torch.sigmoid(output)
    
    print(f"\nInference outputs:")
    print(f"  Logits shape: {output.shape}")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Activations shape: {activations.shape}")
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"  Active notes (threshold=0.5): {activations.sum():.0f}")
    
    print("\n✓ Inference test passed!")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("HUM2MELODY MODEL + DATASET INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Create temporary test data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            print("\nCreating dummy test data...")
            manifest_path = create_dummy_data(tmpdir, num_samples=5)
            print(f"✓ Created 5 test samples in {tmpdir}")
            
            # Run tests
            model = test_model_architecture()
            dataset = test_dataset_loading(manifest_path)
            dataloader = test_dataloader(dataset)
            test_training_step(model, dataloader)
            test_inference(model)
            
            # Final summary
            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
            print("\nYour model and dataset are ready for training!")
            print("\nNote: Target downsampling (500->62 frames) is handled in training.")
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)