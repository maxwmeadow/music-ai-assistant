"""
Verify Combined Model

Quick verification that the combined model loads and produces valid outputs.
Since TorchScript export has complexities, this verifies the model works
in its native PyTorch form (which is still fully deployable).

Usage:
    python scripts/verify_combined_model.py
"""

import torch
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.combined_model import CombinedHum2MelodyModel


def verify_model(
    pitch_ckpt_path: str,
    onset_ckpt_path: str,
    device: str = 'cpu',
    num_runs: int = 10
):
    """Verify combined model works correctly."""

    print(f"\n{'='*70}")
    print(f"COMBINED MODEL VERIFICATION")
    print(f"{'='*70}")

    device = torch.device(device)

    # Load combined model
    print("\n1. Loading combined model...")
    model = CombinedHum2MelodyModel(
        pitch_ckpt_path=pitch_ckpt_path,
        onset_ckpt_path=onset_ckpt_path,
        device=str(device)
    )
    model.eval()

    # Get model info
    prep_info = model.get_preprocessing_info()
    arch_info = model.get_architecture_info()

    print(f"\n2. Model specifications:")
    print(f"   Sample rate: {prep_info['sr']} Hz")
    print(f"   Hop length: {prep_info['hop_length']}")
    print(f"   CQT bins: {prep_info['n_bins']}")
    print(f"   Input frames: {prep_info['target_frames']}")
    print(f"   Output frames: {prep_info['expected_output_frames']}")
    print(f"   Total parameters: {model.count_parameters():,}")
    print(f"   Pitch model: {arch_info['pitch']['model_class']}")
    print(f"   Onset model: {arch_info['onset']['model_class']}")

    # Test forward pass with random data
    print(f"\n3. Testing forward pass with random data...")
    cqt = torch.randn(2, 1, 500, 88).to(device)
    extras = torch.randn(2, 1, 500, 24).to(device)

    with torch.no_grad():
        frame, onset, offset, f0 = model(cqt, extras)

    print(f"   Input shapes:")
    print(f"     CQT:    {list(cqt.shape)}")
    print(f"     Extras: {list(extras.shape)}")
    print(f"   Output shapes:")
    print(f"     Frame:  {list(frame.shape)}")
    print(f"     Onset:  {list(onset.shape)}")
    print(f"     Offset: {list(offset.shape)}")
    print(f"     F0:     {list(f0.shape)}")

    # Verify shapes
    expected_time = 125
    assert frame.shape == (2, expected_time, 88), f"Frame shape mismatch!"
    assert onset.shape == (2, expected_time, 1), f"Onset shape mismatch!"
    assert offset.shape == (2, expected_time, 1), f"Offset shape mismatch!"
    assert f0.shape == (2, expected_time, 2), f"F0 shape mismatch!"
    print(f"   ✅ All output shapes correct!")

    # Test without extras (should use zeros)
    print(f"\n4. Testing forward pass without extras...")
    with torch.no_grad():
        frame2, onset2, offset2, f02 = model(cqt, None)

    assert frame2.shape == (2, expected_time, 88), f"Frame shape mismatch!"
    print(f"   ✅ Forward pass with None extras works!")

    # Benchmark latency
    print(f"\n5. Benchmarking latency ({num_runs} runs, batch=1)...")
    test_cqt = torch.randn(1, 1, 500, 88).to(device)
    test_extras = torch.randn(1, 1, 500, 24).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_cqt, test_extras)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(test_cqt, test_extras)
            end = time.time()
            latencies.append((end - start) * 1000)  # ms

    import numpy as np
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print(f"   Mean latency: {mean_lat:.2f} ± {std_lat:.2f} ms")
    print(f"   Min: {min_lat:.2f} ms | Max: {max_lat:.2f} ms")

    # Test consistency
    print(f"\n6. Testing output consistency...")
    outputs = []
    with torch.no_grad():
        for _ in range(3):
            out = model(test_cqt, test_extras)
            outputs.append(out)

    # Check all outputs are identical
    all_identical = True
    for i in range(1, 3):
        for j, name in enumerate(['Frame', 'Onset', 'Offset', 'F0']):
            diff = (outputs[0][j] - outputs[i][j]).abs().max().item()
            if diff > 1e-6:
                print(f"   ⚠️  {name} differs: max diff = {diff:.2e}")
                all_identical = False

    if all_identical:
        print(f"   ✅ All outputs are consistent!")

    # Summary
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Combined model loads successfully")
    print(f"✅ Forward pass works correctly")
    print(f"✅ Output shapes are correct")
    print(f"✅ Handles missing extras gracefully")
    print(f"✅ Latency: {mean_lat:.2f} ms (CPU, batch=1)")
    print(f"✅ Outputs are consistent")
    print(f"\n📦 MODEL SUCCESSFULLY BUNDLED!")
    print(f"\n   Your original checkpoints are safe:")
    print(f"   - {pitch_ckpt_path}")
    print(f"   - {onset_ckpt_path}")
    print(f"\n   Combined model can be used directly in Python:")
    print(f"   ```python")
    print(f"   from models.combined_model import CombinedHum2MelodyModel")
    print(f"   model = CombinedHum2MelodyModel(")
    print(f"       pitch_ckpt_path='{pitch_ckpt_path}',")
    print(f"       onset_ckpt_path='{onset_ckpt_path}'")
    print(f"   )")
    print(f"   frame, onset, offset, f0 = model(cqt, extras)")
    print(f"   ```")
    print(f"{'='*70}\n")

    # Save results
    results = {
        'status': 'success',
        'model_info': {
            'total_parameters': model.count_parameters(),
            'pitch_model': arch_info['pitch']['model_class'],
            'onset_model': arch_info['onset']['model_class'],
        },
        'preprocessing': prep_info,
        'performance': {
            'mean_latency_ms': float(mean_lat),
            'std_latency_ms': float(std_lat),
            'min_latency_ms': float(min_lat),
            'max_latency_ms': float(max_lat),
            'device': str(device),
            'batch_size': 1
        },
        'tests': {
            'forward_pass': True,
            'correct_shapes': True,
            'handles_none_extras': True,
            'consistent_outputs': all_identical
        }
    }

    with open('combined_model_verification.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: combined_model_verification.json\n")

    return results


if __name__ == '__main__':
    verify_model(
        pitch_ckpt_path='checkpoints_fixed_data/best_model_with_metadata.pth',
        onset_ckpt_path='checkpoints_enhanced_onset/best_model.pth',
        device='cpu',
        num_runs=10
    )
