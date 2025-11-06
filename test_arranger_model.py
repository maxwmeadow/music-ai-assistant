"""
Test script for ArrangerTransformer model

Runs comprehensive tests to verify model functionality including:
- Forward pass with dummy data
- Multi-instrument output splitting
- Gradient flow
- Parameter counting
- Variable sequence lengths
- Padding mask support
"""

import torch
import torch.nn as nn
from backend.models.arranger_model import ArrangerTransformer, create_padding_mask


def test_basic_forward_pass():
    """Test basic forward pass with dummy data."""
    print("=" * 70)
    print("TEST 1: Basic Forward Pass")
    print("=" * 70)

    model = ArrangerTransformer(
        input_dim=4,
        model_dim=256,
        num_heads=8,
        num_layers=6,
        num_instruments=3
    )

    batch_size = 2
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, 4)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, 12)")

    try:
        output = model(dummy_input)
        print(f"Actual output shape: {output.shape}")

        assert output.shape == (batch_size, seq_len, 12), \
            f"Output shape mismatch! Expected (2, 32, 12), got {output.shape}"

        print("[PASS] Forward pass successful!")
        print(f"[PASS] Output shape correct: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False


def test_instrument_splitting():
    """Test multi-instrument output splitting."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Instrument Output Splitting")
    print("=" * 70)

    model = ArrangerTransformer()

    batch_size = 2
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, 4)

    try:
        instruments = model.predict_instruments(dummy_input)

        print("Instrument outputs:")
        for name, tensor in instruments.items():
            print(f"  {name}: {tensor.shape}")
            assert tensor.shape == (batch_size, seq_len, 4), \
                f"{name} shape mismatch! Expected (2, 32, 4), got {tensor.shape}"

        # Verify we have all three instruments
        expected_instruments = {'bass', 'pads', 'counter_melody'}
        assert set(instruments.keys()) == expected_instruments, \
            f"Instrument mismatch! Expected {expected_instruments}, got {set(instruments.keys())}"

        print("[PASS] All instrument outputs have correct shape (batch=2, seq=32, features=4)")
        print("[PASS] All three instruments present: bass, pads, counter_melody")
        return True
    except Exception as e:
        print(f"[FAIL] Instrument splitting failed: {e}")
        return False


def test_gradient_flow():
    """Test that gradients flow through the model properly."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Flow")
    print("=" * 70)

    model = ArrangerTransformer(
        input_dim=4,
        model_dim=128,  # Smaller for faster testing
        num_heads=4,
        num_layers=2
    )

    batch_size = 2
    seq_len = 16
    dummy_input = torch.randn(batch_size, seq_len, 4)
    dummy_target = torch.randn(batch_size, seq_len, 12)

    try:
        # Forward pass
        output = model(dummy_input)

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, dummy_target)

        print(f"Loss value: {loss.item():.4f}")

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = False
        total_grad_norm = 0.0
        params_with_grad = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                params_with_grad += 1

        print(f"[PASS] Parameters with gradients: {params_with_grad}")
        print(f"[PASS] Total gradient norm: {total_grad_norm:.4f}")
        print(f"[PASS] Average gradient norm: {total_grad_norm / params_with_grad:.6f}")

        assert has_gradients, "No gradients found in model!"
        print("[PASS] Gradient flow successful!")
        return True
    except Exception as e:
        print(f"[FAIL] Gradient flow test failed: {e}")
        return False


def test_variable_sequence_lengths():
    """Test model with variable sequence lengths."""
    print("\n" + "=" * 70)
    print("TEST 4: Variable Sequence Lengths")
    print("=" * 70)

    model = ArrangerTransformer(max_seq_len=128)

    test_lengths = [8, 16, 32, 64, 128]

    try:
        for seq_len in test_lengths:
            dummy_input = torch.randn(1, seq_len, 4)
            output = model(dummy_input)

            assert output.shape == (1, seq_len, 12), \
                f"Shape mismatch for seq_len={seq_len}! Got {output.shape}"

            print(f"  seq_len={seq_len:3d}: [PASS] Output shape {output.shape}")

        print("[PASS] Model handles variable sequence lengths correctly!")
        return True
    except Exception as e:
        print(f"[FAIL] Variable sequence length test failed: {e}")
        return False


def test_padding_mask():
    """Test padding mask functionality."""
    print("\n" + "=" * 70)
    print("TEST 5: Padding Mask Support")
    print("=" * 70)

    model = ArrangerTransformer()

    batch_size = 4
    max_len = 32
    actual_lengths = torch.tensor([32, 24, 16, 8])  # Variable actual lengths

    try:
        # Create input with padding
        dummy_input = torch.randn(batch_size, max_len, 4)

        # Create padding mask
        padding_mask = create_padding_mask(actual_lengths, max_len)

        print(f"Actual lengths: {actual_lengths.tolist()}")
        print(f"Padding mask shape: {padding_mask.shape}")
        print(f"Padding mask:\n{padding_mask.int()}")

        # Forward pass with mask
        output = model(dummy_input, src_key_padding_mask=padding_mask)

        assert output.shape == (batch_size, max_len, 12), \
            f"Output shape mismatch! Got {output.shape}"

        print(f"[PASS] Output shape with padding: {output.shape}")
        print("[PASS] Padding mask applied successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Padding mask test failed: {e}")
        return False


def test_instrument_conditioning():
    """Test instrument type conditioning."""
    print("\n" + "=" * 70)
    print("TEST 6: Instrument Conditioning")
    print("=" * 70)

    model = ArrangerTransformer()

    batch_size = 4
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, 4)

    # Different instrument IDs for each batch element
    instrument_ids = torch.tensor([0, 1, 2, 3])  # piano, guitar, synth, strings

    try:
        # Forward pass with instrument conditioning
        output = model(dummy_input, instrument_ids=instrument_ids)

        assert output.shape == (batch_size, seq_len, 12), \
            f"Output shape mismatch! Got {output.shape}"

        print(f"Instrument IDs: {instrument_ids.tolist()}")
        print(f"Output shape: {output.shape}")
        print("[PASS] Instrument conditioning successful!")
        return True
    except Exception as e:
        print(f"[FAIL] Instrument conditioning test failed: {e}")
        return False


def test_model_info():
    """Test model info and parameter counting."""
    print("\n" + "=" * 70)
    print("TEST 7: Model Information & Parameter Counting")
    print("=" * 70)

    model = ArrangerTransformer(
        input_dim=4,
        model_dim=256,
        num_heads=8,
        num_layers=6
    )

    try:
        info = model.get_model_info()

        print("Model Configuration:")
        for key, value in info.items():
            if key == 'total_parameters':
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")

        param_count = model.count_parameters()
        print(f"\n[PASS] Total trainable parameters: {param_count:,}")

        # Verify parameter count is reasonable (should be several million)
        assert param_count > 1_000_000, "Parameter count seems too low!"
        assert param_count < 100_000_000, "Parameter count seems too high!"

        print("[PASS] Parameter count is reasonable!")
        return True
    except Exception as e:
        print(f"[FAIL] Model info test failed: {e}")
        return False


def test_realistic_music_scenario():
    """Test with realistic musical note data."""
    print("\n" + "=" * 70)
    print("TEST 8: Realistic Musical Scenario")
    print("=" * 70)

    model = ArrangerTransformer()

    try:
        # Simulate a simple melody: C major scale
        # Features: [pitch (MIDI), start_time (normalized), duration (normalized), velocity (normalized)]
        melody_notes = torch.tensor([
            [60, 0.0, 0.25, 0.8],   # C4
            [62, 0.25, 0.25, 0.8],  # D4
            [64, 0.5, 0.25, 0.8],   # E4
            [65, 0.75, 0.25, 0.8],  # F4
            [67, 1.0, 0.25, 0.8],   # G4
            [69, 1.25, 0.25, 0.8],  # A4
            [71, 1.5, 0.25, 0.8],   # B4
            [72, 1.75, 0.5, 0.9],   # C5
        ])

        # Normalize features to 0-1 range (model expects normalized inputs)
        melody_normalized = melody_notes.clone()
        melody_normalized[:, 0] = melody_normalized[:, 0] / 127.0  # MIDI pitch
        melody_normalized[:, 1] = melody_normalized[:, 1] / 2.0    # Time
        melody_normalized[:, 2] = melody_normalized[:, 2] / 4.0    # Duration
        # Velocity already normalized

        # Add batch dimension
        input_batch = melody_normalized.unsqueeze(0)  # (1, 8, 4)

        print(f"Input melody shape: {input_batch.shape}")
        print(f"Input notes (first 3):\n{input_batch[0, :3]}")

        # Generate arrangement
        model.eval()
        with torch.no_grad():
            instruments = model.predict_instruments(input_batch)

        print("\nGenerated arrangement:")
        for inst_name, inst_output in instruments.items():
            print(f"  {inst_name}: {inst_output.shape}")
            print(f"    First note: pitch={inst_output[0, 0, 0]:.3f}, "
                  f"time={inst_output[0, 0, 1]:.3f}, "
                  f"duration={inst_output[0, 0, 2]:.3f}, "
                  f"velocity={inst_output[0, 0, 3]:.3f}")

        print("[PASS] Model successfully generated arrangement for realistic melody!")
        return True
    except Exception as e:
        print(f"[FAIL] Realistic scenario test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("ARRANGER TRANSFORMER MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        test_basic_forward_pass,
        test_instrument_splitting,
        test_gradient_flow,
        test_variable_sequence_lengths,
        test_padding_mask,
        test_instrument_conditioning,
        test_model_info,
        test_realistic_music_scenario
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n[FAIL] Test {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n*** ALL TESTS PASSED! Model is ready for integration. ***")
    else:
        print(f"\n!!! WARNING: {total - passed} test(s) failed. Please review. !!!")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)