"""
reward.py — Composite Reward Formulation for Baseline V2 (Pleines et al., 2025)
================================================================================
Full mathematical formulation conforming to:
  R_t = R_event + R_nav + R_heal + R_lvl

Where:
  1. Milestone Event Reward:
     R_event = 2.0 * Delta N_events
  2. Navigation Novelty:
     R_nav   = 0.005 * I[c_t not in H_visited], where c_t = <wPlayerXCoord, wPlayerYCoord, wCurMap>
  3. Pokemon Center Healing Reward:
     R_heal  = 2.5 * sum_{i=1}^6 (HP_{i,t} - HP_{i,t-1})^+ / HP_{i,max}
  4. Level-Up Progress Reward:
     R_lvl   = 0.5 * Delta LevelPotential(PartyLevels)
     LevelPotential(L) = min( sum(L), (sum(L) - 22)/4 + 22 )
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from pokemon_rl.baseline_v2.constants import (
    WEIGHT_EVENT,
    WEIGHT_NAV,
    WEIGHT_HEAL,
    WEIGHT_LVL,
)


def compute_level_potential(levels: List[int]) -> float:
    """
    Computes piece-wise linear level potential as defined in Pleines et al. (2025).
    Below sum(levels) <= 22: slope = 1.0 (encourages early leveling)
    Above sum(levels) > 22:  slope = 0.25 (diminishing returns to prevent grinding traps)
    """
    total_levels = sum(levels)
    if total_levels <= 22:
        return float(total_levels)
    return float((total_levels - 22) / 4.0 + 22.0)


def compute_event_reward(current_events: int, last_events: int) -> float:
    """Computes R_event = 2.0 * max(0, Delta N_events)."""
    delta = max(0, current_events - last_events)
    return WEIGHT_EVENT * float(delta)


def compute_nav_reward(coords: Tuple[int, int, int], visited_coords: Set[Tuple[int, int, int]]) -> Tuple[float, bool]:
    """
    Computes R_nav = 0.005 if coordinate (x, y, map_id) has not been visited yet.
    Returns (reward, is_new_tile).
    """
    if coords not in visited_coords:
        return WEIGHT_NAV, True
    return 0.0, False


def compute_heal_reward(
    current_hp: List[int],
    last_hp: List[int],
    max_hp: List[int],
) -> float:
    """
    Computes R_heal = 2.5 * sum_i (HP_gain_i / MaxHP_i).
    Only positive HP gains are counted; damage does not penalize healing reward.
    """
    hp_gain_ratio = 0.0
    for i in range(min(len(current_hp), len(last_hp), len(max_hp))):
        m_hp = max(1, max_hp[i])
        diff = current_hp[i] - last_hp[i]
        if diff > 0:
            hp_gain_ratio += diff / float(m_hp)
    return WEIGHT_HEAL * hp_gain_ratio


def compute_level_reward(current_levels: List[int], last_levels: List[int]) -> float:
    """Computes R_lvl = 0.5 * max(0, Delta LevelPotential)."""
    curr_pot = compute_level_potential(current_levels)
    last_pot = compute_level_potential(last_levels)
    delta = max(0.0, curr_pot - last_pot)
    return WEIGHT_LVL * delta


class PleinesCompositeReward:
    """
    Stateful manager for the composite linear reward from Pleines et al. (IEEE CoG 2025).
    Maintains visited tile hash set, event counters, and party telemetry baselines.
    """

    def __init__(self, enable_lvl: bool = True, enable_heal: bool = True):
        self.enable_lvl = enable_lvl
        self.enable_heal = enable_heal
        self.visited_coords: Set[Tuple[int, int, int]] = set()
        self.last_event_count: int = 0
        self.last_party_hp: List[int] = [0] * 6
        self.party_max_hp: List[int] = [100] * 6
        self.last_party_levels: List[int] = [5] + [0] * 5

    def reset(self, initial_levels: List[int] = None, initial_hp: List[int] = None, max_hp: List[int] = None):
        """Resets all stateful tracking baselines for a new episode."""
        self.visited_coords.clear()
        self.last_event_count = 0
        self.last_party_hp = list(initial_hp) if initial_hp else [20] + [0] * 5
        self.party_max_hp = list(max_hp) if max_hp else [20] + [0] * 5
        self.last_party_levels = list(initial_levels) if initial_levels else [5] + [0] * 5

    def compute_reward(
        self,
        coords: Tuple[int, int, int],          # (x, y, map_id)
        current_events: int,                   # cumulative count of completed milestones
        current_hp: List[int],                 # current HP across 6 party slots
        max_hp: List[int],                     # max HP across 6 party slots
        current_levels: List[int],             # levels across 6 party slots
    ) -> Tuple[float, Dict[str, float]]:
        """
        Step-level reward computation returning (total_reward, breakdown_dict).
        """
        # 1. Milestone Event Reward
        r_event = compute_event_reward(current_events, self.last_event_count)
        self.last_event_count = current_events

        # 2. Navigation Novelty
        r_nav, is_new = compute_nav_reward(coords, self.visited_coords)
        if is_new:
            self.visited_coords.add(coords)

        # 3. Pokemon Center Healing Reward
        r_heal = 0.0
        if self.enable_heal:
            r_heal = compute_heal_reward(current_hp, self.last_party_hp, max_hp)
        self.last_party_hp = list(current_hp)
        self.party_max_hp = list(max_hp)

        # 4. Level-Up Progress Reward
        r_lvl = 0.0
        if self.enable_lvl:
            r_lvl = compute_level_reward(current_levels, self.last_party_levels)
        self.last_party_levels = list(current_levels)

        total_reward = r_event + r_nav + r_heal + r_lvl
        breakdown = {
            "r_event": r_event,
            "r_nav": r_nav,
            "r_heal": r_heal,
            "r_lvl": r_lvl,
            "total": total_reward,
        }
        return total_reward, breakdown
