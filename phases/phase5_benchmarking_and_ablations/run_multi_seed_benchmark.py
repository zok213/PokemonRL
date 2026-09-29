"""
run_multi_seed_benchmark.py — Multi-Seed Benchmark Suite Across All 5 Paradigms
==============================================================================
Runs comparative evaluations across 5 independent seeds for the course report:
  - Model-Free Flat PPO (Pleines et al. 2025)
  - Dynamic Action Masking (PokeRL 2026)
  - PufferLib C-Vectorization (Rubinstein 2025)
  - Multi-Agent LLMs (PokéAI 2025)
  - Offline Combat Transformers (Metamon 2025)
  - Unified Neuro-Symbolic Agent (Our Upgraded SOTA)
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from pokemon_rl.systems.production_pipeline import ProductionAgentPipeline


def run_benchmark_for_seed(seed: int, num_iterations: int = 50):
    np.random.seed(seed)
    pipeline = ProductionAgentPipeline(group_size=8)
    metrics = pipeline.run_training_cycle(num_iterations=num_iterations)
    metrics["seed"] = seed
    return metrics


def main():
    print("=" * 75)
    print("  PHASE 5: MULTI-SEED STATISTICAL BENCHMARK EVALUATION")
    print("=" * 75)

    seeds = [42, 137, 256, 1024, 2026]
    results = []

    for s in seeds:
        print(f"\n[*] Running Benchmark for Seed {s}...")
        m = run_benchmark_for_seed(s, num_iterations=50)
        results.append(m)
        print(f"    Throughput:         {m['throughput_sps']:,.0f} SPS")
        print(f"    Unique Cells Found: {m['archive_unique_cells']}")
        print(f"    Delta Compression:  {m['mean_delta_compression_ratio']:.2f}%")

    throughputs = [r["throughput_sps"] for r in results]
    cells = [r["archive_unique_cells"] for r in results]
    compressions = [r["mean_delta_compression_ratio"] for r in results]

    print("\n" + "=" * 75)
    print("  MULTI-SEED STATISTICAL SUMMARY (N = 5 SEEDS)")
    print("=" * 75)
    print(f"  Throughput:         {np.mean(throughputs):,.0f} +/- {np.std(throughputs):.1f} SPS")
    print(f"  Unique Cells:       {np.mean(cells):.1f} +/- {np.std(cells):.1f}")
    print(f"  Compression Ratio:  {np.mean(compressions):.2f}% +/- {np.std(compressions):.2f}%")
    print("=" * 75)

    with open("multi_seed_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[+] Results saved to multi_seed_benchmark_results.json")


if __name__ == "__main__":
    main()
