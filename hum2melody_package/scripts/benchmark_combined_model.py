"""
Benchmark Combined Model

Measures latency and memory usage of the combined hum2melody model.

Tests:
1. Inference latency (CPU and GPU)
2. Memory usage
3. Throughput (samples/second)

Acceptance criteria:
- Latency < 100ms per sample (batch=1, GPU)
- Memory < 1GB (GPU)
- Consistent outputs across multiple runs

Usage:
    python scripts/benchmark_combined_model.py \\
        --model combined_hum2melody.pt \\
        --device cuda \\
        --num-runs 100
"""

import torch
import argparse
from pathlib import Path
import sys
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("⚠️  GPUtil not available, GPU memory tracking disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available, CPU memory tracking disabled")


def benchmark_latency(model, device, batch_size=1, num_runs=100, warmup_runs=10):
    """Benchmark inference latency."""
    print(f"\n{'='*70}")
    print(f"LATENCY BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {num_runs}")

    # Create test inputs
    cqt = torch.randn(batch_size, 1, 500, 88).to(device)
    extras = torch.randn(batch_size, 1, 500, 24).to(device)

    # Warmup
    print(f"\nWarming up...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(cqt, extras)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running benchmark...")
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            _ = model(cqt, extras)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

    # Statistics
    import numpy as np
    latencies = np.array(latencies)

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\nResults:")
    print(f"  Mean:   {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"  Median: {p50:.2f} ms")
    print(f"  Min:    {min_latency:.2f} ms")
    print(f"  Max:    {max_latency:.2f} ms")
    print(f"  P95:    {p95:.2f} ms")
    print(f"  P99:    {p99:.2f} ms")

    # Throughput
    throughput = (batch_size * num_runs) / (sum(latencies) / 1000)
    print(f"\nThroughput: {throughput:.2f} samples/second")

    # Check acceptance criteria
    target_latency = 100  # ms
    if mean_latency < target_latency:
        print(f"✅ Latency within target (< {target_latency} ms)")
    else:
        print(f"⚠️  Latency exceeds target ({mean_latency:.2f} ms > {target_latency} ms)")
        print(f"   Consider: FP16, CPU offload, or sequential execution")

    return {
        'mean': mean_latency,
        'std': std_latency,
        'min': min_latency,
        'max': max_latency,
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'throughput': throughput
    }


def benchmark_memory(model, device, batch_size=1):
    """Benchmark memory usage."""
    print(f"\n{'='*70}")
    print(f"MEMORY BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Create test inputs
    cqt = torch.randn(batch_size, 1, 500, 88).to(device)
    extras = torch.randn(batch_size, 1, 500, 24).to(device)

    # Measure memory before inference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Get memory before
        mem_before = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

        # Run inference
        with torch.no_grad():
            _ = model(cqt, extras)

        # Get memory after
        mem_after = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

        print(f"\nGPU Memory:")
        print(f"  Before inference: {mem_before:.1f} MB")
        print(f"  After inference:  {mem_after:.1f} MB")
        print(f"  Peak usage:       {mem_peak:.1f} MB")

        # Get total GPU memory if GPUtil available
        if GPUTIL_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                total_mem = gpu.memoryTotal
                used_mem = gpu.memoryUsed
                print(f"  Total available:  {total_mem:.0f} MB")
                print(f"  Currently used:   {used_mem:.0f} MB")
            except Exception as e:
                print(f"  (Could not get GPU info: {e})")

        # Check acceptance criteria
        target_memory = 1024  # MB (1GB)
        if mem_peak < target_memory:
            print(f"✅ Memory within target (< {target_memory} MB)")
        else:
            print(f"⚠️  Memory exceeds target ({mem_peak:.1f} MB > {target_memory} MB)")
            print(f"   Consider: FP16 or moving onset model to CPU")

        return {
            'before': mem_before,
            'after': mem_after,
            'peak': mem_peak
        }

    else:
        # CPU memory
        if PSUTIL_AVAILABLE:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)

            print(f"\nCPU Memory:")
            print(f"  Process RSS: {mem_mb:.1f} MB")

            return {
                'rss': mem_mb
            }
        else:
            print("\n⚠️  psutil not available, cannot measure CPU memory")
            return {}


def test_consistency(model, device, num_runs=5):
    """Test that model produces consistent outputs."""
    print(f"\n{'='*70}")
    print(f"CONSISTENCY TEST")
    print(f"{'='*70}")
    print(f"Testing {num_runs} runs with same input...")

    # Create test input
    cqt = torch.randn(1, 1, 500, 88).to(device)
    extras = torch.randn(1, 1, 500, 24).to(device)

    # Run multiple times
    outputs = []
    with torch.no_grad():
        for _ in range(num_runs):
            out = model(cqt, extras)
            outputs.append(out)

    # Check all outputs are identical
    print(f"\nComparing outputs...")
    all_identical = True

    for i in range(1, num_runs):
        for j, (name, idx) in enumerate([('Frame', 0), ('Onset', 1), ('Offset', 2), ('F0', 3)]):
            diff = (outputs[0][idx] - outputs[i][idx]).abs().max().item()
            if diff > 1e-6:
                print(f"  ⚠️  {name} differs between run 0 and {i}: max diff = {diff:.2e}")
                all_identical = False

    if all_identical:
        print(f"✅ All {num_runs} runs produced identical outputs")
    else:
        print(f"⚠️  Outputs differ across runs (may indicate non-determinism)")

    return all_identical


def main():
    parser = argparse.ArgumentParser(description="Benchmark combined model")
    parser.add_argument(
        '--model',
        type=str,
        default='combined_hum2melody.pt',
        help='Path to exported TorchScript model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to benchmark on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for benchmarking'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs'
    )
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=10,
        help='Number of warmup runs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save benchmark results JSON'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COMBINED MODEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")

    device = torch.device(args.device)

    # Load model
    print(f"\nLoading model...")
    try:
        model = torch.jit.load(args.model, map_location=device)
        model.eval()
        print(f"✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Run benchmarks
    results = {}

    # 1. Latency
    results['latency'] = benchmark_latency(
        model, device,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )

    # 2. Memory
    results['memory'] = benchmark_memory(
        model, device,
        batch_size=args.batch_size
    )

    # 3. Consistency
    results['consistency'] = test_consistency(model, device, num_runs=5)

    # Summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"\nLatency:")
    print(f"  Mean: {results['latency']['mean']:.2f} ms")
    print(f"  P95:  {results['latency']['p95']:.2f} ms")
    print(f"  Throughput: {results['latency']['throughput']:.2f} samples/sec")

    if device.type == 'cuda':
        print(f"\nMemory:")
        print(f"  Peak: {results['memory']['peak']:.1f} MB")
    elif PSUTIL_AVAILABLE:
        print(f"\nMemory:")
        print(f"  RSS: {results['memory']['rss']:.1f} MB")

    print(f"\nConsistency: {'✅ Pass' if results['consistency'] else '⚠️  Fail'}")

    # Save results if requested
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
