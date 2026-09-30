"""
run_interactive_phase1.py — Interactive Live Visualization of Phase 1 Baseline (Pleines et al.)
=================================================================================================
Provides a live visual dashboard for inspecting the Pleines et al. (IEEE CoG 2025)
empirical baseline in real-time or step-by-step.

Visualizations:
  1. 2D Overworld Map (Pallet Town & Route 1) with real-time player positioning (@)
  2. 48x48 Spatial Memory Visited Grid
  3. Action Probabilities Distribution (PPO Policy Actor Head)
  4. Work RAM Telemetry HUD (HP, Level, Badges, Step Budget)
  5. Composite Linear Reward Breakdown (R_event, R_nav, R_heal, R_lvl)

Usage:
  python phases/phase1_baseline_reimplementation/run_interactive_phase1.py
  python phases/phase1_baseline_reimplementation/run_interactive_phase1.py --steps 50 --fps 8
  python phases/phase1_baseline_reimplementation/run_interactive_phase1.py --manual
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Ensure Windows terminal supports UTF-8 Unicode characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src and phase1 directory to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pleines_baseline_env import PleinesPokemonRedEnv, PleinesAction
from pleines_ppo_policy import PleinesActorCriticPolicy


# ANSI Color Codes for Rich Terminal Output
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BG_BLUE = "\033[44m"


# Pallet Town & Route 1 Overworld Layout (16x16 grid)
# P = Player Home, R = Rival Home, L = Oak Lab, G = Grass, F = Fence, T = Town, W = Water
MAP_GRID_PALLET = [
    "GGGGGGG..GGGGGGG", # Y=0 (Route 1 Entrance)
    "GGGGGGG..GGGGGGG", # Y=1 (Route 1 Path)
    "FFFFFFF..FFFFFFF", # Y=2 (Fence North)
    "T..............T", # Y=3 (Pallet North)
    "T..[P]....[R]..T", # Y=4 (Player House & Rival House)
    "T..[P]....[R]..T", # Y=5
    "T..............T", # Y=6
    "T..............T", # Y=7
    "T........[L]...T", # Y=8 (Prof Oak's Lab)
    "T........[L]...T", # Y=9
    "T..............T", # Y=10
    "FFFFFF....FFFFFF", # Y=11 (Fence South)
    "WWWWWW....WWWWWW", # Y=12 (Water Shoreline)
    "WWWWWW....WWWWWW", # Y=13
    "WWWWWWWWWWWWWWWW", # Y=14
    "WWWWWWWWWWWWWWWW", # Y=15
]


def render_ascii_overworld(player_x: int, player_y: int, visited_coords: set) -> str:
    """Renders a colorful 16x16 ASCII overworld grid with player coordinates."""
    lines = []
    lines.append(f"{CYAN}┌────────────────── 2D OVERWORLD TOPOLOGY ──────────────────┐{RESET}")

    for y in range(16):
        row_chars = []
        for x in range(16):
            if x == player_x and y == player_y:
                row_chars.append(f"{RED}{BOLD} @ {RESET}")
            elif (x, y, 0) in visited_coords:
                row_chars.append(f"{YELLOW} · {RESET}")
            else:
                base = MAP_GRID_PALLET[y][x]
                if base == 'W':
                    row_chars.append(f"{BLUE} ~ {RESET}")
                elif base == 'G':
                    row_chars.append(f"{GREEN} \" {RESET}")
                elif base == 'F':
                    row_chars.append(f"{DIM} # {RESET}")
                elif base in '[]PRL':
                    row_chars.append(f"{WHITE} {base} {RESET}")
                else:
                    row_chars.append(" . ")
        row_str = "".join(row_chars)
        lines.append(f"│ {row_str} │")

    lines.append(f"{CYAN}└───────────────────────────────────────────────────────────┘{RESET}")
    return "\n".join(lines)


def render_policy_distribution(probs: np.ndarray, chosen_action: int) -> str:
    """Renders an action probability bar chart."""
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
    lines = []
    lines.append(f"{CYAN}┌─────────────── PPO POLICY ACTION PROBS ───────────────┐{RESET}")

    for idx, name in enumerate(action_names):
        p = probs[idx]
        bar_len = int(round(p * 20))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        marker = f"{RED}{BOLD} ▶ {RESET}" if idx == chosen_action else "   "
        pct = f"{p*100:5.1f}%"
        color = GREEN if idx == chosen_action else WHITE
        lines.append(f"│{marker}{color}{name:<6s}{RESET} {bar} {pct}                     │")

    lines.append(f"{CYAN}└───────────────────────────────────────────────────────┘{RESET}")
    return "\n".join(lines)


def render_telemetry_hud(
    step: int,
    budget: int,
    x: int,
    y: int,
    map_id: int,
    hp: int,
    max_hp: int,
    level: int,
    badges: int,
    events: int,
    total_reward: float,
    breakdown: Dict[str, float],
    value_estimate: float,
) -> str:
    """Renders comprehensive HUD with WRAM state and Pleines reward decomposition."""
    hp_pct = max(0.0, min(1.0, hp / max(1, max_hp)))
    hp_bars = int(round(hp_pct * 12))
    hp_bar_str = f"[{GREEN}{'█' * hp_bars}{RED}{'░' * (12 - hp_bars)}{RESET}] {hp}/{max_hp}"

    budget_pct = max(0.0, min(1.0, (budget - step) / max(1, budget)))
    b_bars = int(round(budget_pct * 12))
    budget_bar_str = f"[{CYAN}{'█' * b_bars}{DIM}{'░' * (12 - b_bars)}{RESET}] {step}/{budget}"

    lines = []
    lines.append(f"{CYAN}┌────────────────── WRAM TELEMETRY HUD ─────────────────┐{RESET}")
    lines.append(f"│ Position:     X={x:2d}, Y={y:2d}  (Map {map_id:2d}: Pallet Town)        │")
    lines.append(f"│ Lead Party:   Squirtle (Lv. {level:2d})  HP: {hp_bar_str:<23s} │")
    lines.append(f"│ Step Budget:  {budget_bar_str:<39s} │")
    lines.append(f"│ Badges:       {badges:2d}/8     Storyline Milestones: {events:2d}/7          │")
    lines.append(f"│ Critic V(s):  {value_estimate:+.4f}                                │")
    lines.append(f"├──────────────── COMPOSITE REWARD BREAKDOWN ───────────┤")
    lines.append(f"│ Cumulative:   {BOLD}{YELLOW}{total_reward:+8.4f}{RESET}                                │")
    lines.append(f"│   R_event:    {breakdown.get('r_event', 0.0):+8.4f}  (Gyms & Oak's Parcel)     │")
    lines.append(f"│   R_nav:      {breakdown.get('r_nav', 0.0):+8.4f}  (Spatial Novelty +0.005)  │")
    lines.append(f"│   R_heal:     {breakdown.get('r_heal', 0.0):+8.4f}  (PokeCenter Delta-HP)     │")
    lines.append(f"│   R_lvl:      {breakdown.get('r_lvl', 0.0):+8.4f}  (Party Level Growth)      │")
    lines.append(f"{CYAN}└───────────────────────────────────────────────────────┘{RESET}")
    return "\n".join(lines)


def run_interactive_phase1(
    num_steps: int = 100,
    fps: float = 6.0,
    manual_mode: bool = False,
    starter: str = "squirtle",
):
    """
    Main interactive loop for Phase 1 baseline execution.
    """
    print(f"{BOLD}{BG_BLUE} POKÉMON RED RL: PHASE 1 INTERACTIVE VISUALIZER {RESET}")
    print(f"Target: Pleines et al. (IEEE CoG 2025 / IEEE Xplore Doc. 11114399)")
    print(f"Configuration: Starter={starter.upper()}, Dynamic Budget=10,240 + 2,048*N_events\n")

    env = PleinesPokemonRedEnv(starter=starter, enable_lvl=True, enable_heal=True)
    policy = PleinesActorCriticPolicy(seed=42)

    obs, info = env.reset()
    total_reward = 0.0
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]

    delay = 1.0 / max(fps, 0.1) if fps > 0 else 0.0

    print("Controls:")
    print("  [Auto Mode] Running automatically. Press Ctrl+C to halt.")
    print("  [Manual Mode] Enter u/d/l/r/a/b/s to steer player, or press Enter to follow policy.")
    time.sleep(1.0)

    try:
        for step in range(1, num_steps + 1):
            # 1. Forward pass through Pleines Actor-Critic Policy
            probs, val = policy.forward(
                obs["screen"][np.newaxis, ...],
                obs["spatial_map"][np.newaxis, ...],
                obs["telemetry"][np.newaxis, ...],
            )
            model_action = int(np.random.choice(PleinesAction.NUM_ACTIONS, p=probs[0]))

            # 2. Determine action (policy or manual override)
            if manual_mode:
                user_in = input(f"\nStep {step}/{num_steps} [Policy chooses {action_names[model_action]}]. Input (u/d/l/r/a/b/s or Enter to accept): ").strip().lower()
                manual_map = {
                    'u': PleinesAction.UP, 'd': PleinesAction.DOWN,
                    'l': PleinesAction.LEFT, 'r': PleinesAction.RIGHT,
                    'a': PleinesAction.A, 'b': PleinesAction.B, 's': PleinesAction.START
                }
                action = manual_map.get(user_in, model_action)
            else:
                action = model_action

            # 3. Environment Step (24 emulation frames)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward

            # 4. Extract telemetry for display
            px = env.wram[0xD362 - 0xC000]
            py = env.wram[0xD361 - 0xC000]
            map_id = env.wram[0xD35E - 0xC000]
            badges = env.wram[0xD356 - 0xC000]
            cur_hp = env.current_hp[0]
            max_hp = env.max_hp[0]
            level = env.levels[0]
            events = env.events_completed
            budget = step_info["budget"]
            breakdown = step_info["reward_breakdown"]

            # Clear screen in terminal for smooth animation
            # os.system('cls' if os.name == 'nt' else 'clear')
            print("\033[H\033[J", end="")

            print(f"{BOLD}POKÉMON RED — PHASE 1 INTERACTIVE RE-IMPLEMENTATION (Pleines et al. 2025){RESET}")
            print(f"Step: {step:4d}/{num_steps} | Cadence: 24 Frames/Action (~392 SPS) | Mode: {'Manual Override' if manual_mode else 'Autonomous PPO'}")
            print("=" * 65)

            # Print Overworld ASCII Map
            print(render_ascii_overworld(px, py, env.reward_fn.visited_coords))

            # Print Policy Action Distribution
            print(render_policy_distribution(probs[0], action))

            # Print WRAM Telemetry HUD
            print(render_telemetry_hud(
                step=step, budget=budget, x=px, y=py, map_id=map_id,
                hp=cur_hp, max_hp=max_hp, level=level, badges=badges,
                events=events, total_reward=total_reward, breakdown=breakdown,
                value_estimate=float(val[0])
            ))

            print(f"\n{BOLD}Action Executed:{RESET} [{action_names[action]}] | {BOLD}Step Reward:{RESET} {reward:+.4f} | {BOLD}Novel Tiles:{RESET} {len(env.reward_fn.visited_coords)}")

            if terminated or truncated:
                print(f"\n{YELLOW}[!] Episode Finished at Step {step}! Budget exhausted or goal reached.{RESET}")
                obs, _ = env.reset()
                time.sleep(1.5)

            if not manual_mode and delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}[*] Visualization paused by user.{RESET}")

    print("\n" + "=" * 65)
    print(f"  INTERACTIVE RUN SUMMARY:")
    print(f"    Total Steps Simulated:    {step}")
    print(f"    Unique Tiles Discovered:  {len(env.reward_fn.visited_coords)}")
    print(f"    Total Accumulated Reward: {total_reward:+.4f}")
    print(f"    Final Player Position:    X={px}, Y={py} (Map {map_id})")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokémon Red RL Phase 1 Interactive Visualizer")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to simulate")
    parser.add_argument("--fps", type=float, default=5.0, help="Visualization frames per second")
    parser.add_argument("--manual", action="store_true", help="Enable manual user input override per step")
    parser.add_argument("--starter", type=str, default="squirtle", choices=["squirtle", "bulbasaur", "charmander"], help="Starter Pokémon")
    args = parser.parse_args()

    run_interactive_phase1(
        num_steps=args.steps,
        fps=args.fps,
        manual_mode=args.manual,
        starter=args.starter,
    )
