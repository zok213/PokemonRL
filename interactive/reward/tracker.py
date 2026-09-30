"""
interactive/reward/tracker.py — RewardTracker: Unified Reward Orchestrator
===========================================================================
Ties together WhiddenReward + AntiStagnation + coordinate exploration
into one clean interface for session.py.

Also tracks:
  - Unique (map, x, y) coordinate set for exploration count
  - Cumulative per-component breakdown for HUD display

Usage:
    tracker = RewardTracker()
    step_reward = tracker.step(mem)       # call after each emulator tick
    snapshot    = tracker.snapshot(mem)   # call to get full telemetry dict
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Any, Optional
import numpy as np

from interactive.wram.reader import (
    read_map_id, read_xy, read_badges, read_party_level,
    read_hp, read_max_hp, read_party_size, read_moves,
    is_in_battle, is_menu_or_text_active,
    read_text_box_id, read_menu_type, read_joy_ignore,
)
from interactive.wram.addresses import LEVELS_ADDRS
from interactive.reward.whidden import WhiddenRewardState, WhiddenStepReward, compute_whidden_step
from interactive.reward.anti_stagnation import (
    AntiStagnationState, AntiStagnationReward, compute_anti_stagnation,
    is_stagnant, is_menu_locked,
)
from interactive.constants import MAP_NAMES


# Exploration reward per new unique coordinate.
# Raised 10× from the original 0.0005 to make tile discovery a dominant signal.
# At 0.005 per tile, 1 new tile ≈ 1 level-up (0.1 scaled), making exploration
# the primary early-game driver rather than heal oscillation.
EXPLORE_REWARD_PER_TILE = 0.005


class RewardTracker:
    """
    Single entry point for reward computation and telemetry.

    Combines:
      - WhiddenRewardState (event flags, level, badge, op_level, heal, dead)
      - AntiStagnationState (stagnation + menu penalty)
      - Coordinate exploration set

    All reward sub-components are exposed for HUD display.
    """

    def __init__(self):
        self._whidden   = WhiddenRewardState()
        self._antistag  = AntiStagnationState()

        self.seen_coords: Set[Tuple[int, int, int]] = set()
        self.cumulative_reward: float = 0.0

        # Per-step and cumulative breakdown matrices (for HUD)
        self.reward_matrix: Dict[str, float] = self._zero_matrix()
        self.reward_matrix_cumulative: Dict[str, float] = self._zero_matrix()

        # HUD shims
        self.active_objective_desc: str = "Explore & Gain Events"
        self.dist_to_goal: int = 0
        self.reward_machine = _DummyRM()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, mem) -> float:
        """
        Compute reward for the current WRAM state.
        Call AFTER the emulator has ticked (post-action state).

        Returns:
            float: total scaled step reward
        """
        # Coordinate exploration
        cur_map = read_map_id(mem)
        x, y    = read_xy(mem)
        in_battle = is_in_battle(mem)
        in_menu   = is_menu_or_text_active(mem)

        tile_key = (cur_map, x, y)
        is_novel = tile_key not in self.seen_coords
        if is_novel:
            self.seen_coords.add(tile_key)
        r_explore = EXPLORE_REWARD_PER_TILE if (is_novel and not in_menu and not in_battle) else 0.0

        # Whidden rewards
        whidden: WhiddenStepReward = compute_whidden_step(mem, self._whidden)

        # Anti-stagnation penalties
        antistag: AntiStagnationReward = compute_anti_stagnation(mem, self._antistag, is_novel=is_novel)

        # Total
        step_total = whidden.total + r_explore + antistag.total

        # Update matrices
        self.reward_matrix = {
            "event":      whidden.event,
            "level":      whidden.level,
            "badge":      whidden.badge,
            "op_level":   whidden.op_lvl,
            "heal":       whidden.heal,
            "dead":       whidden.dead,
            "explore":    r_explore,
            "stagnation": antistag.stagnation,
            "menu_pen":   antistag.menu_pen,
            "total":      step_total,
        }
        for k, v in self.reward_matrix.items():
            self.reward_matrix_cumulative[k] = self.reward_matrix_cumulative.get(k, 0.0) + v

        self.cumulative_reward += step_total
        return step_total

    def snapshot(self, mem) -> Dict[str, Any]:
        """
        Return a complete telemetry snapshot dict from live WRAM.
        Used by session.py to feed the HUD and observation builder.
        """
        cur_map    = read_map_id(mem)
        x, y       = read_xy(mem)
        badges     = read_badges(mem)
        party_size = read_party_size(mem)
        level      = read_party_level(mem, 0) if party_size > 0 else 5
        hp         = read_hp(mem, 0)
        max_hp     = max(read_max_hp(mem, 0), 1)
        species_id = mem[0xD164] if party_size > 0 else 0
        move_ids, move_pps = read_moves(mem)
        in_battle  = is_in_battle(mem)
        in_menu    = is_menu_or_text_active(mem)
        tile_key   = (cur_map, x, y)

        return {
            "map_id":         cur_map,
            "map_name":       MAP_NAMES.get(cur_map, f"Map({cur_map})"),
            "x":              x,
            "y":              y,
            "badges":         badges,
            "hp":             hp,
            "max_hp":         max_hp,
            "level":          level,
            "species_id":     species_id,
            "move_ids":       move_ids,
            "move_pps":       move_pps,
            "is_novel":       tile_key not in self.seen_coords,
            "is_in_battle":   in_battle,
            "in_menu":        in_menu,
            # Legacy aliases (for backward-compat with session.py checks)
            "text_active":    in_menu,
            "is_menu_active": in_menu,
            "text_box_id":    read_text_box_id(mem),
            "menu_type":      read_menu_type(mem),
            "joy_ignore":     read_joy_ignore(mem),
        }

    def is_stagnant(self) -> bool:
        return is_stagnant(self._antistag)

    def is_menu_locked(self) -> bool:
        return is_menu_locked(self._antistag)

    @property
    def visited_count(self) -> int:
        return len(self.seen_coords)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_matrix() -> Dict[str, float]:
        return {
            "event": 0.0, "level": 0.0, "badge": 0.0,
            "op_level": 0.0, "heal": 0.0, "dead": 0.0,
            "explore": 0.0, "stagnation": 0.0, "menu_pen": 0.0,
            "total": 0.0,
        }


class _DummyRM:
    """HUD shim — shows a human-readable RM state string."""
    current_state: str = "WHIDDEN_ALIGNED"
