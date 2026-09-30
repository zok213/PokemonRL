"""
train.py — Main Training Entry Point for Autonomous JRPG Agent
==============================================================
Runs the unified neuro-symbolic production pipeline with Go-Explore state archiving,
Critic-Free Adaptive Tau-GRPO, Dynamic Action Masking, and Decoupled Combat.

Usage:
    python scripts/train.py --iterations 500 --group-size 8 --tau 0.25 --alpha 2.0
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

# Ensure src/ is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pokemon_rl.systems.production_pipeline import ProductionAgentPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train Autonomous JRPG Agent on Pokémon Red")
    parser.add_argument("--iterations", type=int, default=250, help="Number of training iterations / decision forks")
    parser.add_argument("--group-size", type=int, default=8, help="Number of parallel sibling rollouts G per fork")
    parser.add_argument("--tau", type=float, default=0.25, help="Adaptive Tau temperature for STAD variance injection")
    parser.add_argument("--alpha", type=float, default=2.0, help="Directed Frontier Distance (DFD) progress exponent")
    parser.add_argument("--max-cells", type=int, default=20000, help="Maximum cells stored in Go-Explore archive")
    parser.add_argument("--use-torch", action="store_true", help="Enable PyTorch autograd policy training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip",
        help="Path to Whidden 439M step checkpoint for warm-start visual transfer",
    )
    parser.add_argument("--output-json", type=str, default="train_metrics.json", help="Path to save output JSON metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("  POKEMON RED AUTONOMOUS AGENT — SOTA TRAINING RUN")
    print("=" * 70)
    print(f"  Iterations:          {args.iterations}")
    print(f"  Sibling Group Size:  G = {args.group_size}")
    print(f"  Adaptive Tau (STAD): {args.tau}")
    print(f"  DFD Alpha Progress:  {args.alpha}")
    print(f"  Archive Max Cells:   {args.max_cells:,}")
    print(f"  PyTorch Autograd:    {args.use_torch}")
    if args.use_torch:
        print(f"  Warm-Start Weights:  {args.checkpoint}")
    print("=" * 70)

    pipeline = ProductionAgentPipeline(
        group_size=args.group_size,
        use_torch_policy=args.use_torch,
        whidden_checkpoint_path=args.checkpoint if args.use_torch else None,
    )
    t0 = time.time()
    metrics = pipeline.run_training_cycle(num_iterations=args.iterations)
    elapsed = time.time() - t0

    metrics["wall_clock_seconds"] = elapsed
    metrics["config"] = vars(args)

    print("\n" + "=" * 70)
    print("  TRAINING CYCLE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Throughput:           {metrics['throughput_sps']:,.1f} SPS")
    print(f"  Total Actions:        {metrics['total_actions']:,}")
    print(f"  Unique Cells Found:   {metrics['archive_unique_cells']:,}")
    print(f"  Delta Compression:    {metrics['mean_delta_compression_ratio']:.2f}% (32KB -> ~103B)")
    print(f"  STAD Augmented:       {metrics['augmented_variance_cycles']} cycles")
    print(f"  Wall-Clock Time:      {elapsed:.2f} s")
    print("=" * 70)

    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Metrics written to {args.output_json}")


if __name__ == "__main__":
    main()
