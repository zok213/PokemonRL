"""
run_v2_jev_hybrid_interactive.py — Live Interactive Runner for V2 Weights + Jev Affordances
==========================================================================================
Runs the flagship hybrid architecture:
  - Loads Peter Whidden's 26.2M-step pretrained PPO checkpoint (poke_26214400.zip)
  - Applies Jev-style Residual Policy Steering (TacticalBattleController + WorldGraphNavigator)
  - Optional Online Self-Tuning GRPO (DecisionForkGRPO + OnlineGRPOTrainer)
  - Continuous progression across all 8 Gym Badges and Indigo Plateau Elite Four
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
from pokemon_rl.agent.grpo_steering_policy import DecisionForkGRPO, OnlineGRPOTrainer
from pokemon_rl.env.wram_map import RAMMap
from interactive.constants import MAP_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline V2 + Jev Affordance Hybrid Interactive Runner")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without SDL2 window")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to execute (0 = infinite)")
    parser.add_argument("--alpha", type=float, default=1.2, help="Jev affordance steering strength (default: 1.2)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom path to PPO checkpoint .zip")
    parser.add_argument("--state", type=str, default=None, help="Custom path to Game Boy .state file")
    parser.add_argument("--grpo", action="store_true", help="Enable learned steering gain via online GRPO")
    parser.add_argument("--all-badges", action="store_true", default=True, help="Run continuously across all 8 badges and Hall of Fame")
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

    # 4. Optional Online GRPO Initialization
    grpo_controller = DecisionForkGRPO(group_size=8) if args.grpo else None
    grpo_trainer = OnlineGRPOTrainer(grpo_controller) if grpo_controller else None

    print("=" * 75)
    print("  POKEMONRL — BASELINE V2 PRETRAINED WEIGHTS + JEV AFFORDANCE STEERING")
    print("  Architecture: Residual Policy Optimization (V2 Motor Prior + Jev Supervisor)")
    print(f"  V2 Checkpoint: {Path(ckpt_path).name} (26.2M steps PPO)")
    print(f"  ROM: {Path(rom_path).name} | State: {Path(state_path).name if state_path else 'Fresh Game'}")
    print(f"  Steering Mode: {'Online GRPO Learned' if args.grpo else f'Fixed alpha={args.alpha}'}")
    print(f"  Run Scope: {'All 8 Badges & Indigo Plateau' if args.all_badges else 'Target Cascade Badge'}")
    print(f"  Display: {'Headless' if args.headless else 'Interactive SDL2 Window'}")
    print("=" * 75, flush=True)

    # 5. Initialize Environment
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

    # 6. Initialize Steered Hybrid Policy
    policy = BaselineV2JevSteeredPolicy(
        checkpoint_path=ckpt_path,
        default_alpha=args.alpha,
        grpo_controller=grpo_controller
    )

    obs, info = env.reset()
    step_count = 0

    # Milestone tracking bitmasks
    saved_pewter = bool(args.state and "pewter" in args.state)
    saved_boulder = bool(args.state and "boulder" in args.state)
    saved_route3 = False
    saved_route4 = False
    saved_mt_moon = False
    saved_cerulean = bool(args.state and "cerulean" in args.state)
    saved_pokecenter_heal = False
    saved_cascade = False
    saved_thunder = False
    saved_rainbow = False
    saved_soul = False
    saved_marsh = False
    saved_volcano = False
    saved_earth = False
    saved_hall_of_fame = False

    print("\n[+] Starting live policy execution loop...\n", flush=True)

    try:
        while True:
            step_count += 1

            # Read WRAM Snapshot directly from Game Boy RAM
            map_id = env.read_m(RAMMap.CURRENT_MAP)
            cur_x = env.read_m(RAMMap.PLAYER_X)
            cur_y = env.read_m(RAMMap.PLAYER_Y)
            in_battle = env.read_m(RAMMap.IS_IN_BATTLE) != 0
            battle_type = env.read_m(RAMMap.IS_IN_BATTLE)
            badges = env.read_m(RAMMap.OBTAINED_BADGES)
            text_box = env.read_m(RAMMap.TEXT_BOX_ID)
            font_loaded = (env.read_m(0xCFC4) & 1) != 0
            border_tile = env.read_m(0xC3A0 + 17 * 20 + 19) == 0x7E
            is_text_active = (font_loaded or border_tile) and (text_box != 0)

            hp_val = env.read_hp(RAMMap.PARTY_HP_BASE)
            max_hp = max(1, env.read_hp(RAMMap.PARTY_MAX_HP_BASE))
            lvl = env.read_m(0xD18C)

            battle_hp = env.read_hp(0xD015) if in_battle else hp_val
            battle_max_hp = max(1, env.read_hp(0xD023)) if in_battle else max_hp

            party_count = env.read_m(RAMMap.NUM_PARTY_MONS)
            party_hps = [env.read_hp(RAMMap.PARTY_HP_BASE + i * 44) for i in range(party_count)] if party_count > 0 else [hp_val]

            menu_y = env.read_m(0xCC24)
            menu_x = env.read_m(0xCC25)
            cur_item = env.read_m(0xCC26)

            battle_moves = [env.read_m(0xD01C + i) for i in range(4)]
            party_moves = [env.read_m(0xD173 + i) for i in range(4)]
            moves = battle_moves if in_battle and any(m > 0 for m in battle_moves) else party_moves
            move_pps = [env.read_m(0xD02D + i) for i in range(4)] if in_battle else [35, 40, 25, 30]

            opp_species = env.read_m(RAMMap.ENEMY_SPECIES)
            opp_hp = env.read_hp(0xCFE6)
            is_evolving = env.read_m(0xCCD3) != 0
            first_mons_not_out = env.read_m(0xD11D) != 0 if in_battle else False

            c_lo = env.read_m(0xCC30)
            c_hi = env.read_m(0xCC31)
            c_ptr = c_lo | (c_hi << 8)
            tile_at_cursor = env.read_m(c_ptr) if (0xC3A0 <= c_ptr < 0xC508) else 0
            menu_cursor_active = (tile_at_cursor == 0xED) if in_battle else True

            dis_byte = env.read_m(0xD06D) if in_battle else 0
            disabled_move_slot = (((dis_byte >> 4) & 0x0F) - 1) if (dis_byte & 0xF0) != 0 else -1

            player_dir = env.read_m(RAMMap.PLAYER_FACING)
            bag_count = env.read_m(RAMMap.NUM_BAG_ITEMS)
            bag_items = [env.read_m(RAMMap.BAG_ITEMS_START + 2 * i) for i in range(bag_count)]

            fossil_byte = env.read_m(0xD5A6 + 549 // 8)
            event_fossil = bool(fossil_byte & ((1 << (549 % 8)) | (1 << (550 % 8))))
            has_fossil = event_fossil or (RAMMap.ITEM_DOME_FOSSIL in bag_items) or (RAMMap.ITEM_HELIX_FOSSIL in bag_items)

            snapshot = {
                "map_id": map_id,
                "x": cur_x,
                "y": cur_y,
                "player_dir": player_dir,
                "has_fossil": has_fossil,
                "in_battle": in_battle,
                "battle_type": battle_type,
                "badges": badges,
                "bag": bag_items,
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

            coll_matrix = env.pyboy.game_area_collision() if hasattr(env.pyboy, "game_area_collision") else None

            # Hybrid Decision: V2 Logits + Jev Affordances (+ optional GRPO)
            decision = policy.decide(snapshot, obs=obs, collision_matrix=coll_matrix)

            # Step Environment
            obs, rewards, terminated, truncated, info = env.step(decision.action_idx)

            # Online GRPO Training Step (if active)
            if grpo_trainer:
                grpo_trainer.record_transition(snapshot, decision.steering_weight, rewards)
                metrics = grpo_trainer.maybe_train_step()
                if metrics and step_count % 100 == 0:
                    print(f"[*] Online GRPO Update: Loss={metrics['loss_total']:.4f}, KL={metrics['kl_div']:.4f}, MeanAdv={metrics['mean_advantage']:.3f}", flush=True)

            # Periodic Telemetry Log
            if step_count % 10 == 0 or in_battle or decision.is_overridden:
                map_str = MAP_NAMES.get(map_id, f"Map_{map_id}")
                v2_top = max(decision.v2_probs.keys(), key=lambda k: decision.v2_probs[k])
                v2_pct = int(decision.v2_probs[v2_top] * 100)
                steered_pct = int(decision.steered_probs[decision.action_name] * 100)
                override_str = "YES [Steered]" if decision.is_overridden else "No [Prior Agree]"
                hp_str = f"BattleHP={battle_hp}/{battle_max_hp}" if in_battle else f"LeadHP={hp_val}/{max_hp}"

                print(
                    f"Step {step_count:5d} | {map_str:12s} ({cur_x:2d},{cur_y:2d}) Badges={badges:02X} | {hp_str} Lv={lvl} | "
                    f"V2: {v2_top} ({v2_pct}%) -> {decision.action_name} ({steered_pct}%) | Alpha={decision.steering_weight:.2f} | "
                    f"Context: {decision.affordance_situation}",
                    flush=True
                )

            if not args.headless:
                env.render()

            # Checkpoint Save every 500 steps
            if step_count % 500 == 0:
                ckpt_file = REPO_ROOT / "saves" / "checkpoint_progress.state"
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_file, "wb") as f:
                    env.pyboy.save_state(f)

            # -----------------------------------------------------------------
            # ALL-KANTO MILESTONE SAVE STATE AUTOMATION
            # -----------------------------------------------------------------
            def save_milestone(filename: str, desc: str):
                save_file = REPO_ROOT / "saves" / filename
                save_file.parent.mkdir(parents=True, exist_ok=True)
                with open(save_file, "wb") as sf:
                    env.pyboy.save_state(sf)
                print(f"\n{'='*75}\n  [+] MILESTONE ACHIEVED: {desc} AT STEP {step_count}!\n  Saved: {save_file.name}\n{'='*75}\n", flush=True)

            if map_id == 2 and not saved_pewter:
                saved_pewter = True
                save_milestone("pewter_city.state", "ENTERED PEWTER CITY")

            if (badges & RAMMap.BADGE_BOULDER) and not saved_boulder:
                saved_boulder = True
                save_milestone("boulder_badge.state", "BOULDER BADGE ACQUIRED (BROCK DEFEATED)")

            if map_id == 14 and not saved_route3:
                saved_route3 = True
                save_milestone("route3.state", "ENTERED ROUTE 3")

            if map_id == 15 and cur_x < 22 and not saved_route4:
                saved_route4 = True
                save_milestone("route4.state", "ENTERED ROUTE 4")

            if map_id in (59, 60, 61) and not saved_mt_moon:
                saved_mt_moon = True
                save_milestone("mt_moon.state", "ENTERED MT. MOON")

            if map_id == 64 and hp_val == max_hp and hp_val > 0 and not saved_pokecenter_heal:
                saved_pokecenter_heal = True
                save_milestone("cerulean_healed.state", "CERULEAN POKÉCENTER HEALED")

            if map_id == 3 and not saved_cerulean:
                saved_cerulean = True
                save_milestone("cerulean_city.state", "CERULEAN CITY REACHED")

            if (badges & RAMMap.BADGE_CASCADE) and not saved_cascade:
                saved_cascade = True
                save_milestone("cascade_badge.state", "CASCADE BADGE ACQUIRED (MISTY DEFEATED)")
                if not args.all_badges:
                    break

            if (badges & RAMMap.BADGE_THUNDER) and not saved_thunder:
                saved_thunder = True
                save_milestone("thunder_badge.state", "THUNDER BADGE ACQUIRED (LT. SURGE DEFEATED)")

            if (badges & RAMMap.BADGE_RAINBOW) and not saved_rainbow:
                saved_rainbow = True
                save_milestone("rainbow_badge.state", "RAINBOW BADGE ACQUIRED (ERIKA DEFEATED)")

            if (badges & RAMMap.BADGE_SOUL) and not saved_soul:
                saved_soul = True
                save_milestone("soul_badge.state", "SOUL BADGE ACQUIRED (KOGA DEFEATED)")

            if (badges & RAMMap.BADGE_MARSH) and not saved_marsh:
                saved_marsh = True
                save_milestone("marsh_badge.state", "MARSH BADGE ACQUIRED (SABRINA DEFEATED)")

            if (badges & RAMMap.BADGE_VOLCANO) and not saved_volcano:
                saved_volcano = True
                save_milestone("volcano_badge.state", "VOLCANO BADGE ACQUIRED (BLAINE DEFEATED)")

            if (badges & RAMMap.BADGE_EARTH) and not saved_earth:
                saved_earth = True
                save_milestone("earth_badge.state", "EARTH BADGE ACQUIRED (GIOVANNI DEFEATED)")

            if map_id == 118 and not saved_hall_of_fame:
                saved_hall_of_fame = True
                save_milestone("hall_of_fame.state", "POKÉMON LEAGUE CHAMPION! HALL OF FAME REGISTERED")
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
