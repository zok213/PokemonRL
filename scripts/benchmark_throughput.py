"""
benchmark_throughput.py — Systems Latency & Throughput Benchmark Suite
======================================================================
Profiles inference latency, environment simulation throughput (SPS),
and delta-compression memory throughput against 2026 published baselines.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pokemon_rl.agent.policy_network import MultiModalPolicyNetwork
from pokemon_rl.exploration.go_explore import DeltaStateCompressor
from pokemon_rl.env.action_masker import DynamicActionMasker
from pokemon_rl.systems.production_pipeline import ProductionAgentPipeline


def benchmark_policy_latency():
    print("\n--- 1. MultiModal Policy Network Inference Latency ---")
    net = MultiModalPolicyNetwork(seed=42)
    batch_sizes = [1, 8, 32, 64]
    rng = np.random.default_rng(42)

    for B in batch_sizes:
        frames = rng.random((B, 4, 72, 80)).astype(np.float32)
        map_obs = (rng.random((B, 1, 48, 48)) > 0.5).astype(np.float32)
        wram = rng.random((B, 64)).astype(np.float32)

        # Warmup
        for _ in range(10):
            net.forward(frames, map_obs, wram)

        N_ITERS = 100
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            net.forward(frames, map_obs, wram)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0 / N_ITERS
        sps = (B * N_ITERS) / (t1 - t0)
        print(f"  Batch B={B:2d}:  {elapsed_ms:6.2f} ms/forward  |  Throughput: {sps:8,.0f} SPS (numpy CPU)")


def benchmark_delta_compression():
    print("\n--- 2. Delta State Compression Engine ---")
    keyframe = bytes(8192)
    test_state = bytearray(8192)
    # Simulate 25 modified memory bytes
    for i in range(25):
        test_state[i * 200] = (i * 13) % 256
    test_state_bytes = bytes(test_state)

    N_ITERS = 1000
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        compressed = DeltaStateCompressor.compress(test_state_bytes, keyframe)
    t1 = time.perf_counter()
    compress_us = (t1 - t0) * 1e6 / N_ITERS

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        restored = DeltaStateCompressor.decompress(compressed, keyframe)
    t1 = time.perf_counter()
    decompress_us = (t1 - t0) * 1e6 / N_ITERS

    ratio = (1.0 - len(compressed) / 8192.0) * 100.0
    print(f"  Raw Size:          8,192 bytes")
    print(f"  Compressed Size:   {len(compressed)} bytes ({ratio:.2f}% savings)")
    print(f"  Compression:       {compress_us:.1f} us/state")
    print(f"  Decompression:     {decompress_us:.1f} us/state")
    print(f"  1M Cells VRAM:     {1_000_000 * len(compressed) / 1e6:.1f} MB (fits standard GPU memory)")


def benchmark_pipeline_throughput():
    print("\n--- 3. End-to-End Pipeline Simulation Throughput ---")
    pipeline = ProductionAgentPipeline(group_size=8)
    metrics = pipeline.run_training_cycle(num_iterations=200)

    print(f"  Single CPU Core Throughput: {metrics['throughput_sps']:,.0f} SPS")
    print(f"  Comparison vs Baselines:")
    print(f"    - Pleines et al. (IEEE CoG 2025):     392 SPS (24-frame stride)")
    print(f"    - Python Multiprocessing (PokeRL):  1,000 SPS")
    print(f"    - PufferLib C-Vectorized:          50,000 SPS")
    print(f"    - PokeJAX / EmuRust (COLM 2026): 15,200,000 SPS")


def main():
    print("=" * 70)
    print("  POKEMON RED AUTONOMOUS AGENT — SYSTEMS BENCHMARK SUITE")
    print("=" * 70)
    benchmark_policy_latency()
    benchmark_delta_compression()
    benchmark_pipeline_throughput()
    print("\n" + "=" * 70)
    print("  ALL BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
