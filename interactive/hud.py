"""
interactive/hud.py — Real-Time Terminal Telemetry HUD
======================================================
Flicker-free ANSI terminal display of:
  - Location, party stats, exploration count
  - Whidden-aligned reward matrix breakdown (per-step + cumulative)
  - PPO policy action probability histogram
  - Active subsystem mode
"""

from __future__ import annotations
import sys
from typing import Dict, Any, List, Optional
import numpy as np
from interactive.constants import WHIDDEN_ACTION_NAMES

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TerminalHUD:
    """Renders Game Boy RL telemetry to the terminal at each step."""

    def __init__(self, emulation_speed: int = 16):
        self.emulation_speed = emulation_speed

    def render(
        self,
        step_count:              int,
        telemetry:               Dict[str, Any],
        action_idx:              int,
        probs:                   np.ndarray,
        step_reward:             float,
        cumulative_reward:       float,
        visited_count:           int,
        subsystem_str:           str,
        agent_enabled:           bool,
        reward_matrix:           Optional[Dict[str, float]] = None,
        reward_matrix_cumulative:Optional[Dict[str, float]] = None,
        rm_state:                str = "WHIDDEN_ALIGNED",
        objective_desc:          str = "Explore & Gain Events",
        dist_to_goal:            int = 0,
    ):
        names    = WHIDDEN_ACTION_NAMES
        act_name = names[action_idx] if 0 <= action_idx < len(names) else "NONE"

        # HP bar
        hp     = telemetry.get("hp", 0)
        max_hp = telemetry.get("max_hp", 1)
        hp_pct = max(0.0, min(1.0, hp / max_hp))
        hp_bar = "█" * int(hp_pct * 15) + "░" * (15 - int(hp_pct * 15))

        # ANSI: jump to top-left, clear screen (flicker-free)
        print("\033[2J\033[H", end="")
        W = 74
        sep = "=" * W
        dash = "-" * W

        print(sep)
        print(f"  POKÉMON RED — LIVE PYBOY RL HUD  ({self.emulation_speed}x Speed)")
        print(sep)
        print(f"  Step:       {step_count:>8,}   (24 ticks/action = {step_count*24:,} frames)")
        print(f"  Location:   {telemetry['map_name']!s:<22}  Map={telemetry['map_id']:3d}  X={telemetry['x']:3d}  Y={telemetry['y']:3d}")
        print(f"  Party:      Lv.{telemetry['level']:3d}  HP [{hp_bar}] {hp}/{max_hp}  Badges: {telemetry['badges']}/8")
        print(f"  Explore:    {visited_count:,} unique tiles discovered")
        print(f"  Subsystem:  {subsystem_str}")
        print(dash)

        # Reward matrix
        if reward_matrix and reward_matrix_cumulative:
            rm = reward_matrix
            cm = reward_matrix_cumulative
            print("  REWARD MATRIX (Whidden-aligned + Anti-stagnation):")
            print(f"    State/Objective: [{rm_state}]  {objective_desc}")
            rows = [
                ("Event Flags (RM)",  "event"),
                ("Level Gain",        "level"),
                ("Badge Milestone",   "badge"),
                ("Opponent Level",    "op_level"),
                ("Heal Bonus",        "heal"),
                ("Death Penalty",     "dead"),
                ("Coord Exploration", "explore"),
                ("Anti-Stagnation",   "stagnation"),
                ("Menu Penalty",      "menu_pen"),
            ]
            for label, key in rows:
                sv = rm.get(key, 0.0)
                cv = cm.get(key, 0.0)
                if abs(sv) > 1e-7 or abs(cv) > 1e-5:
                    mark = "▶" if abs(sv) > 1e-7 else " "
                    print(f"   {mark} {label:<24} step: {sv:+8.4f}   cumul: {cv:+9.3f}")
            print(f"    {'NET STEP REWARD':<26} {step_reward:+8.4f}   TOTAL: {cumulative_reward:+9.3f}")
        else:
            print(f"    Step Reward: {step_reward:+.4f}   Total: {cumulative_reward:+.4f}")

        print(dash)
        print("  PPO ACTION PROBABILITIES (439M WHIDDEN PRETRAINED):")
        for i, (name, p) in enumerate(zip(names, probs)):
            ptr = "▶ " if i == action_idx else "  "
            bar = "█" * int(p * 26) + "░" * (26 - int(p * 26))
            print(f"   {ptr}{name:<6} [{bar}] {p*100:5.1f}%")
        print(dash)
        mode = "AUTONOMOUS PPO" if agent_enabled else "MANUAL HUMAN OVERRIDE"
        print(f"  Mode: {mode}   [M] Toggle  [Esc/Close] Exit")
        print(sep)
