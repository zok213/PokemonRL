"""
run_v2_jev_hybrid_interactive.py — Live Interactive Runner for V2 Weights + Jev Affordances
==========================================================================================
Runs the flagship hybrid architecture:
  - Loads Peter Whidden's 26.2M-step pretrained PPO checkpoint (poke_26214400.zip)
  - Applies Jev-style Residual Policy Steering (TacticalBattleController + WorldGraphNavigator)
  - Renders live Game Boy graphics via PyBoy SDL2 window and outputs real-time steering telemetry.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path

# Add repo root, src, and external/PokemonRedExperiments/v2 to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = REPO_ROOT / "external" / "PokemonRedExperiments" / "v2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from red_gym_env_v2 import RedGymEnv
from pokemon_rl.agent.v2_jev_hybrid_policy import BaselineV2JevSteeredPolicy, ACTION_NAMES
from interactive.constants import MAP_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline V2 + Jev Affordance Hybrid Interactive Runner")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without SDL2 window")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to execute (0 = infinite)")
    parser.add_argument("--alpha", type=float, default=1.2, help="Jev affordance steering strength (default: 1.2)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path to PPO checkpoint .zip")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve ROM path
    rom_candidates = [
        REPO_ROOT / "roms" / "pokemon_red.gb",
        REPO_ROOT / "PokemonRed.gb",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "PokemonRed.gb",
    ]
    rom_path = next((str(p) for p in rom_candidates if p.exists()), None)
    if not rom_path:
        print("[!] Error: No Pokemon Red ROM found in roms/ or external/!")
        sys.exit(1)

    # 2. Resolve Init Save State
    state_candidates = [
        REPO_ROOT / "external" / "PokemonRedExperiments" / "has_pokedex_nballs.state",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "init.state",
        REPO_ROOT / "saves" / "has_pokedex_nballs.state",
    ]
    state_path = next((str(p) for p in state_candidates if p.exists()), None)

    # 3. Resolve Checkpoint
    ckpt_path = args.checkpoint or str(V2_DIR / "runs" / "poke_26214400.zip")

    print("=" * 75)
    print("  POKEMONRL — BASELINE V2 PRETRAINED WEIGHTS + JEV AFFORDANCE STEERING")
    print("  Architecture: Residual Policy Optimization (V2 Motor Prior + Jev Supervisor)")
    print(f"  V2 Checkpoint: {Path(ckpt_path).name} (26.2M steps PPO)")
    print(f"  ROM: {Path(rom_path).name} | State: {Path(state_path).name if state_path else 'Fresh Game'}")
    print(f"  Steering Strength (alpha): {args.alpha} | Mode: {'Headless' if args.headless else 'Interactive SDL2 Window'}")
    print("=" * 75, flush=True)

    # 4. Initialize Environment
    sess_path = V2_DIR / f"session_hybrid_{int(time.time())}"
    env_config = {
        "headless": args.headless,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": state_path,
        "max_steps": args.steps if args.steps > 0 else 2**23,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": rom_path,
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False
    }

    env = RedGymEnv(env_config)

    # 5. Initialize Steered Hybrid Policy
    policy = BaselineV2JevSteeredPolicy(
        checkpoint_path=ckpt_path,
        default_alpha=args.alpha
    )

    obs, info = env.reset()
    step_count = 0

    print("\n[+] Starting live policy execution loop...\n", flush=True)

    try:
        while True:
            step_count += 1

            # Read WRAM Snapshot directly from Game Boy RAM
            map_id = env.read_m(0xD35E)
            cur_x = env.read_m(0xD362)
            cur_y = env.read_m(0xD361)
            in_battle = env.read_m(0xD057) != 0
            battle_type = env.read_m(0xD057)
            badges = env.read_m(0xD356)
            text_box = env.read_m(0xCF13)
            font_loaded = (env.read_m(0xCFC4) & 1) != 0
            border_tile = env.read_m(0xC3A0 + 17 * 20 + 19) == 0x7E
            is_text_active = (font_loaded or border_tile) and (text_box != 0)

            hp_val = env.read_hp(0xD16C)
            max_hp = max(1, env.read_hp(0xD18D))
            lvl = env.read_m(0xD18C)
            moves = [env.read_m(0xD173), env.read_m(0xD174), env.read_m(0xD175), env.read_m(0xD176)]

            opp_species = env.read_m(0xCFE5)
            opp_hp = env.read_hp(0xCFE6)

            snapshot = {
                "map_id": map_id,
                "x": cur_x,
                "y": cur_y,
                "in_battle": in_battle,
                "battle_type": battle_type,
                "badges": badges,
                "text_box_id": 1 if is_text_active else 0,
                "text_box_active": is_text_active,
                "active_mon": {
                    "hp": hp_val,
                    "max_hp": max_hp,
                    "level": lvl,
                    "moves": moves
                },
                "opponent_species": opp_species,
                "opponent_hp": opp_hp
            }

            # Hybrid Decision: V2 Logits + Jev Affordances
            decision = policy.decide(snapshot, obs=obs)

            # Step Environment
            obs, rewards, terminated, truncated, info = env.step(decision.action_idx)

            # Print telemetry periodically
            if step_count % 5 == 0 or in_battle or decision.is_overridden:
                map_str = MAP_NAMES.get(map_id, f"Map_{map_id}")
                v2_top = max(decision.v2_probs.keys(), key=lambda k: decision.v2_probs[k])
                v2_pct = int(decision.v2_probs[v2_top] * 100)
                steered_pct = int(decision.steered_probs[decision.action_name] * 100)
                override_str = "YES [Steered]" if decision.is_overridden else "No [Prior Agree]"

                print(
                    f"Step {step_count:4d} | {map_str:12s} ({cur_x:2d},{cur_y:2d}) | HP={hp_val}/{max_hp} Lv={lvl} | "
                    f"V2: {v2_top} ({v2_pct}%) -> Chosen: {decision.action_name} ({steered_pct}%) | Override: {override_str} | "
                    f"Context: {decision.affordance_situation}",
                    flush=True
                )

            if not args.headless:
                env.render()

            if truncated or (args.steps > 0 and step_count >= args.steps):
                break

    except KeyboardInterrupt:
        print("\n[*] Gracefully stopping hybrid runner...")
    finally:
        env.close()
        print(f"[+] Session complete. Total steps executed: {step_count}")


if __name__ == "__main__":
    main()
