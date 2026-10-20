"""
evaluate.py — Evaluation Runner for Baseline V2 Re-Implementation
==================================================================
Evaluates policy rollouts across single or multi-seed episodes,
measuring milestone completion, navigation exploration, and reward dynamics.
"""

from __future__ import annotations
import argparse
import time
from typing import Any, Dict, List, Optional
import numpy as np

from pokemon_rl.baseline_v2.constants import PleinesAction
from pokemon_rl.baseline_v2.env import PleinesBaselineEnv
from pokemon_rl.baseline_v2.models import PleinesActorCriticNumPy


def run_evaluation_episode(
    env: PleinesBaselineEnv,
    policy: PleinesActorCriticNumPy,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs a single evaluation episode until termination or budget exhaustion."""
    obs, info = env.reset()
    budget = max_steps or info.get("budget", 10240)

    total_reward = 0.0
    reward_components = {"r_event": 0.0, "r_nav": 0.0, "r_heal": 0.0, "r_lvl": 0.0}
    tiles_visited = set()

    t0 = time.time()
    steps = 0

    for step in range(budget):
        probs, val = policy.forward(
            obs["screen"][np.newaxis, ...],
            obs["spatial_map"][np.newaxis, ...],
            obs["telemetry"][np.newaxis, ...],
        )
        action = int(np.random.choice(PleinesAction.NUM_ACTIONS, p=probs[0]))
        obs, reward, terminated, truncated, step_info = env.step(action)
        steps += 1
        total_reward += reward

        coords = step_info.get("coords")
        if coords:
            tiles_visited.add(coords)

        for k, v in step_info.get("reward_breakdown", {}).items():
            if k in reward_components:
                reward_components[k] += v

        if terminated or truncated:
            break

    elapsed = time.time() - t0
    sps = steps / elapsed if elapsed > 0 else 0.0

    return {
        "steps": steps,
        "elapsed_seconds": elapsed,
        "sps": sps,
        "total_reward": total_reward,
        "reward_components": reward_components,
        "unique_tiles_visited": len(tiles_visited),
        "milestones_completed": step_info.get("events", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline V2 Policy")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps per episode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--starter", type=str, default="squirtle", help="Starter pokemon")
    parser.add_argument("--rom", type=str, default=None, help="Path to Pokemon Red ROM")
    args = parser.parse_args()

    print("=" * 70)
    print("  EVALUATING BASELINE V2 RE-IMPLEMENTATION")
    print(f"  Starter: {args.starter} | Steps: {args.steps} | Episodes: {args.episodes}")
    print("=" * 70)

    env = PleinesBaselineEnv(rom_path=args.rom, starter=args.starter)
    policy = PleinesActorCriticNumPy(seed=42)

    all_rewards = []
    all_sps = []

    for ep in range(args.episodes):
        res = run_evaluation_episode(env, policy, max_steps=args.steps)
        all_rewards.append(res["total_reward"])
        all_sps.append(res["sps"])
        print(f"Episode {ep + 1}/{args.episodes}: Return = {res['total_reward']:.3f} | Unique Tiles = {res['unique_tiles_visited']} | SPS = {res['sps']:,.0f}")

    print("-" * 70)
    print(f"Mean Return: {np.mean(all_rewards):.3f} +/- {np.std(all_rewards):.3f}")
    print(f"Mean Throughput: {np.mean(all_sps):,.0f} SPS")
    print("=" * 70)


if __name__ == "__main__":
    main()
