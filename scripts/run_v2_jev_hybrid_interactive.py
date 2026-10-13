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
    parser.add_argument("--state", type=str, default=None, help="Custom path to Game Boy .state file")
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
    if args.state and Path(args.state).exists():
        state_path = str(Path(args.state).resolve())
    else:
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
    saved_pewter = bool(args.state and "pewter" in args.state)
    saved_badge = False
    saved_route3 = False
    saved_route4 = False
    saved_mt_moon = False
    saved_cerulean = bool(args.state and "cerulean" in args.state)
    saved_pokecenter_heal = False
    saved_cascade_badge = False

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

            # In battle active Pokemon HP (0xD015) vs overworld lead party HP (0xD16C)
            battle_hp = env.read_hp(0xD015) if in_battle else hp_val
            battle_max_hp = max(1, env.read_hp(0xD023)) if in_battle else max_hp

            # Party count and party HPs
            party_count = env.read_m(0xD163)
            party_hps = [env.read_hp(0xD16C + i * 44) for i in range(party_count)] if party_count > 0 else [hp_val]

            # Battle menu coordinates, moves, and PP values
            menu_y = env.read_m(0xCC24)
            menu_x = env.read_m(0xCC25)
            cur_item = env.read_m(0xCC26)
            battle_moves = [env.read_m(0xD01C + i) for i in range(4)]
            party_moves = [env.read_m(0xD173 + i) for i in range(4)]
            moves = battle_moves if in_battle and any(m > 0 for m in battle_moves) else party_moves
            move_pps = [env.read_m(0xD02D + i) for i in range(4)] if in_battle else [35, 40, 25, 30]

            opp_species = env.read_m(0xCFE5)
            opp_hp = env.read_hp(0xCFE6)
            is_evolving = env.read_m(0xCCD3) != 0
            first_mons_not_out = env.read_m(0xD11D) != 0 if in_battle else False

            # Check if interactive arrow cursor (0xED) is actively rendered in wTileMap
            c_lo = env.read_m(0xCC30)
            c_hi = env.read_m(0xCC31)
            c_ptr = c_lo | (c_hi << 8)
            tile_at_cursor = env.read_m(c_ptr) if (0xC3A0 <= c_ptr < 0xC508) else 0
            menu_cursor_active = (tile_at_cursor == 0xED) if in_battle else True

            # Check if any player move is currently disabled by Disable (wPlayerDisabledMove at 0xD06D)
            dis_byte = env.read_m(0xD06D) if in_battle else 0
            disabled_move_slot = (((dis_byte >> 4) & 0x0F) - 1) if (dis_byte & 0xF0) != 0 else -1

            # Check player direction and fossil acquisition (via event flags OR Bag inventory: 0x29 Dome, 0x2A Helix)
            player_dir = env.read_m(0xC109)
            bag_count = env.read_m(0xD31D)
            bag_items = [env.read_m(0xD31E + 2 * i) for i in range(bag_count)]
            fossil_byte = env.read_m(0xD5A6 + 549 // 8)
            event_fossil = bool(fossil_byte & ((1 << (549 % 8)) | (1 << (550 % 8))))
            has_fossil = event_fossil or (0x29 in bag_items) or (0x2A in bag_items)

            snapshot = {
                "map_id": map_id,
                "x": cur_x,
                "y": cur_y,
                "player_dir": player_dir,
                "has_fossil": has_fossil,
                "in_battle": in_battle,
                "battle_type": battle_type,
                "badges": badges,
                "text_box_id": 1 if is_text_active else 0,
                "text_box_active": is_text_active,
                "first_mons_not_out": first_mons_not_out,
                "menu_cursor_active": menu_cursor_active,
                "is_evolving": is_evolving,
                "battle_menu_y": menu_y,
                "battle_menu_x": menu_x,
                "battle_cur_item": cur_item,
                "move_pps": move_pps,
                "party_hps": party_hps,
                "disabled_move_slot": disabled_move_slot,
                "active_mon": {
                    "hp": battle_hp,
                    "max_hp": battle_max_hp,
                    "level": lvl,
                    "moves": moves
                },
                "opponent_species": opp_species,
                "opponent_hp": opp_hp,
                "opp_def_mod": env.read_m(0xCD2F) if in_battle else 7
            }

            # Read real-time collision matrix from PyBoy
            coll_matrix = env.pyboy.game_area_collision() if hasattr(env.pyboy, "game_area_collision") else None

            # Hybrid Decision: V2 Logits + Jev Affordances
            decision = policy.decide(snapshot, obs=obs, collision_matrix=coll_matrix)

            # Step Environment
            obs, rewards, terminated, truncated, info = env.step(decision.action_idx)

            # Print telemetry periodically
            if step_count % 5 == 0 or in_battle or decision.is_overridden:
                map_str = MAP_NAMES.get(map_id, f"Map_{map_id}")
                v2_top = max(decision.v2_probs.keys(), key=lambda k: decision.v2_probs[k])
                v2_pct = int(decision.v2_probs[v2_top] * 100)
                steered_pct = int(decision.steered_probs[decision.action_name] * 100)
                override_str = "YES [Steered]" if decision.is_overridden else "No [Prior Agree]"
                hp_str = f"BattleHP={battle_hp}/{battle_max_hp}" if in_battle else f"LeadHP={hp_val}/{max_hp}"

                print(
                    f"Step {step_count:4d} | {map_str:12s} ({cur_x:2d},{cur_y:2d}) | {hp_str} Lv={lvl} | "
                    f"V2: {v2_top} ({v2_pct}%) -> Chosen: {decision.action_name} ({steered_pct}%) | Override: {override_str} | "
                    f"Context: {decision.affordance_situation}",
                    flush=True
                )

            if not args.headless:
                env.render()

            # Milestone Save States
            if map_id == 2 and not saved_pewter:
                saved_pewter = True
                save_file = REPO_ROOT / "saves" / "pewter_city.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: ENTERED PEWTER CITY! Saved: {save_file.name}\n", flush=True)

            if (badges & 1 != 0) and not saved_badge:
                saved_badge = True
                save_file = REPO_ROOT / "saves" / "boulder_badge.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: BOULDER BADGE ACQUIRED! Saved: {save_file.name}\n", flush=True)

            if map_id == 14 and not saved_route3:
                saved_route3 = True
                save_file = REPO_ROOT / "saves" / "route3.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: ENTERED ROUTE 3! Saved: {save_file.name}\n", flush=True)

            if map_id == 15 and cur_x < 22 and not saved_route4:
                saved_route4 = True
                save_file = REPO_ROOT / "saves" / "route4.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: ENTERED ROUTE 4! Saved: {save_file.name}\n", flush=True)

            if map_id in (59, 60, 61) and not saved_mt_moon:
                saved_mt_moon = True
                save_file = REPO_ROOT / "saves" / "mt_moon.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: ENTERED MT. MOON! Saved: {save_file.name}\n", flush=True)

            if step_count % 100 == 0:
                ckpt_file = REPO_ROOT / "saves" / "checkpoint_progress.state"
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_file, "wb") as f:
                    env.pyboy.save_state(f)

            if map_id == 64 and hp_val == max_hp and hp_val > 0 and not saved_pokecenter_heal:
                saved_pokecenter_heal = True
                save_file = REPO_ROOT / "saves" / "cerulean_healed.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(f"\n[+] MILESTONE ACHIEVED: PARTY RESTORED AT CERULEAN POKÉCENTER! Saved: {save_file.name}\n", flush=True)

            if map_id == 3 and not saved_cerulean:
                saved_cerulean = True
                save_file = REPO_ROOT / "saves" / "cerulean_city.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(
                    f"\n{'='*75}\n"
                    f"  [+] MILESTONE ACHIEVED: CERULEAN CITY REACHED AT STEP {step_count}!\n"
                    f"  Saved: {save_file.name}\n"
                    f"{'='*75}\n",
                    flush=True
                )

            if (badges & 2 != 0) and not saved_cascade_badge:
                saved_cascade_badge = True
                save_file = REPO_ROOT / "saves" / "cascade_badge.state"
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as f:
                    env.pyboy.save_state(f)
                print(
                    f"\n{'='*75}\n"
                    f"  [+] MISSION ACCOMPLISHED: CASCADE BADGE ACQUIRED AT STEP {step_count}!\n"
                    f"  Defeated Gym Leader Misty! Boulder Badge (Bit 0) + Cascade Badge (Bit 1) = 0x03!\n"
                    f"  Saved Final State: {save_file.name}\n"
                    f"{'='*75}\n",
                    flush=True
                )
                break

            if truncated or (args.steps > 0 and step_count >= args.steps):
                break

    except KeyboardInterrupt:
        print("\n[*] Gracefully stopping hybrid runner...")
    finally:
        env.close()
        print(f"[+] Session complete. Total steps executed: {step_count}")


if __name__ == "__main__":
    main()
