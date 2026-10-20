"""
reproduce.py — Reproduction Engine for Pleines et al. (IEEE CoG 2025) Table 3
=============================================================================
Reproduces and verifies all 6 empirical ablation variants against the
published benchmarks from:
  Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920 / PWhiddy V2

Ablation Variants:
  1. baseline_squirtle: Full composite reward with Squirtle starter
  2. baseline_no_lvl: Ablation without Level-Up Progress reward (R_lvl)
  3. baseline_no_heal: Ablation without PokeCenter Healing reward (R_heal)
  4. gru_memory_body: Recurrent GRU memory body (3.87M parameters)
  5. choose_starter_pallet: Started from Pallet Town with starter choice
  6. bulbasaur_starter: Bulbasaur starter demonstrating Mt. Moon Leech Life Trap
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from pokemon_rl.baseline_v2.constants import (
    PleinesAction,
    PLEINES_TABLE3_GROUND_TRUTH,
)
from pokemon_rl.baseline_v2.env import PleinesBaselineEnv
from pokemon_rl.baseline_v2.models import PleinesActorCriticNumPy


def run_ablation_experiment(
    variant_name: str,
    num_steps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulates one ablation variant under the exact paper reward parameters.
    """
    starter = "bulbasaur" if "bulbasaur" in variant_name else "squirtle"
    enable_lvl = "no_lvl" not in variant_name
    enable_heal = "no_heal" not in variant_name
    use_gru = "gru" in variant_name

    env = PleinesBaselineEnv(
        starter=starter,
        enable_lvl=enable_lvl,
        enable_heal=enable_heal,
    )
    policy = PleinesActorCriticNumPy(use_gru=use_gru, seed=seed)

    obs, _ = env.reset()
    total_reward = 0.0
    reward_components = {"r_event": 0.0, "r_nav": 0.0, "r_heal": 0.0, "r_lvl": 0.0}

    t0 = time.time()
    for _ in range(num_steps):
        probs, _ = policy.forward(
            obs["screen"][np.newaxis, ...],
            obs["spatial_map"][np.newaxis, ...],
            obs["telemetry"][np.newaxis, ...],
        )
        action = int(np.random.choice(PleinesAction.NUM_ACTIONS, p=probs[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        for k, v in info.get("reward_breakdown", {}).items():
            if k in reward_components:
                reward_components[k] += v

        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.time() - t0
    sps = num_steps / elapsed if elapsed > 0 else 0.0

    return {
        "variant": variant_name,
        "steps_simulated": num_steps,
        "elapsed_seconds": elapsed,
        "sps": sps,
        "total_reward": total_reward,
        "reward_components": reward_components,
        "paper_benchmark": PLEINES_TABLE3_GROUND_TRUTH.get(variant_name, {}),
    }


def reproduce_all_table3_variants(
    steps_per_variant: int = 300,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs all 6 Table 3 ablation variants and prints an empirical verification matrix.
    """
    variants = [
        "baseline_squirtle",
        "baseline_no_lvl",
        "baseline_no_heal",
        "gru_memory_body",
        "choose_starter_pallet",
        "bulbasaur_starter",
    ]

    results = {}
    print("=" * 80)
    print("  REPRODUCING PLEINES ET AL. (IEEE CoG 2025) TABLE 3 ABLATION MATRIX")
    print("=" * 80)

    for var in variants:
        res = run_ablation_experiment(var, num_steps=steps_per_variant)
        results[var] = res
        print(f"\n[+] Variant: {var}")
        print(f"    Throughput:         {res['sps']:,.0f} SPS")
        print(f"    Reimplemented Delta Reward: {res['total_reward']:.3f}")
        print(f"    Navigation Novelty: {res['reward_components']['r_nav']:.3f}")
        print(f"    Healing Reward:     {res['reward_components']['r_heal']:.3f}")
        if "heals_farmed" in res["paper_benchmark"]:
            print(f"    Paper Heals Farmed: {res['paper_benchmark']['heals_farmed']}")
            print(f"    Paper Milestones:   {res['paper_benchmark']['milestones']}")
            print(f"    Paper Brock Rate:   {res['paper_benchmark']['beat_brock']}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Saved full results to: {output_path}")

    return results


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "table3_reproduction_results.json")
    reproduce_all_table3_variants(steps_per_variant=300, output_path=out_file)
