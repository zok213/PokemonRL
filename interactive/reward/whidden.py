"""
interactive/reward/whidden.py — Whidden-Aligned Monotone-Max Reward Components
===============================================================================
Replicates get_game_state_reward() from PokemonRedExperiments/red_gym_env.py
exactly so that the pretrained 439M policy sees reward signals consistent with
its training distribution.

All reward components are MONOTONE MAX (they can only go up, never down),
matching the baseline. Delta = new_max - old_max is emitted each step.

Components:
  event:   delta in max(all_events_bitcount)          — game progress
  level:   delta in max(level_sum_scaled)              — party growth
  badge:   delta in (badge_count * 5.0)                — milestone
  op_lvl:  delta in max(opp_level - 5) * 0.2          — combat depth
  heal:    HP fraction increase * 4.0                  — survival
  dead:    -0.1 * death_count cumulative               — death penalty

Scale: all × 0.1 (matches Whidden env's eturn new_reward * 0.1).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from interactive.wram.reader import (
    read_all_events_reward, read_levels_sum, read_badges,
    read_max_opp_level, read_hp_fraction, read_party_size,
)

EXPLORE_THRESH = 22
LEVEL_SCALE    = 4
REWARD_SCALE   = 0.1   # Whidden multiplier


@dataclass
class WhiddenRewardState:
    """
    Mutable state for Whidden monotone-max reward tracking.
    Reset this on environment reset.
    """
    max_event_rew:     int   = 0
    max_level_rew:     float = 0.0
    max_op_level:      int   = 0
    total_healing_rew: float = 0.0
    died_count:        int   = 0
    last_health:       float = 1.0
    last_party_size:   int   = 0
    badge_abs:         float = 0.0  # last badge reward (for delta)
    dead_abs:          float = 0.0  # last dead reward (for delta)


@dataclass
class WhiddenStepReward:
    """Per-step breakdown of each Whidden reward component (unscaled)."""
    event:   float = 0.0
    level:   float = 0.0
    badge:   float = 0.0
    op_lvl:  float = 0.0
    heal:    float = 0.0
    dead:    float = 0.0
    total:   float = 0.0   # sum × REWARD_SCALE

    def asdict(self) -> dict:
        return {
            "event":  self.event,
            "level":  self.level,
            "badge":  self.badge,
            "op_lvl": self.op_lvl,
            "heal":   self.heal,
            "dead":   self.dead,
            "total":  self.total,
        }


def compute_whidden_step(mem, state: WhiddenRewardState) -> WhiddenStepReward:
    """
    Compute one step of Whidden-aligned rewards from live WRAM memory.
    Mutates state to update running trackers.

    Args:
        mem:   PyBoy memory object (supports mem[addr] -> int)
        state: mutable WhiddenRewardState (updated in place)

    Returns:
        WhiddenStepReward with per-component values and scaled total.
    """
    # --- Event Flags (monotone max) ---
    cur_event    = read_all_events_reward(mem)
    new_max_ev   = max(state.max_event_rew, cur_event)
    r_event      = float(new_max_ev - state.max_event_rew)
    state.max_event_rew = new_max_ev

    # --- Level Sum (monotone max, scaled like Whidden's get_levels_reward) ---
    level_sum = read_levels_sum(mem)
    if level_sum < EXPLORE_THRESH:
        cur_level_scaled = float(level_sum)
    else:
        cur_level_scaled = float((level_sum - EXPLORE_THRESH) / LEVEL_SCALE + EXPLORE_THRESH)
    new_max_level = max(state.max_level_rew, cur_level_scaled)
    r_level       = new_max_level - state.max_level_rew
    state.max_level_rew = new_max_level

    # --- Badge Count (5× per badge, emit delta) ---
    cur_badges    = read_badges(mem)
    cur_badge_abs = float(cur_badges) * 5.0
    r_badge       = cur_badge_abs - state.badge_abs
    state.badge_abs = cur_badge_abs

    # --- Opponent Level (monotone max) ---
    cur_opp    = max(read_max_opp_level(mem) - 5, 0)
    new_max_op = max(state.max_op_level, cur_opp)
    r_op_lvl   = float(new_max_op - state.max_op_level) * 0.2
    state.max_op_level = new_max_op

    # --- Heal Reward (HP increase when party unchanged) ---
    # SCALE: 0.5 (reduced from 4.0).
    # Rationale: at scale 4.0, one full PokéCenter heal = +0.4, whereas
    # one new tile = +0.0005 — 800× imbalance causes a Healing Trap where
    # the agent oscillates near a PokéCenter. At 0.5, a full heal ≈ 50 novel
    # tiles, making spatial exploration the dominant gradient signal.
    HEAL_SCALE = 0.5
    cur_health  = read_hp_fraction(mem)
    cur_party   = read_party_size(mem)
    r_heal = 0.0
    if cur_health > state.last_health and cur_party == state.last_party_size:
        if state.last_health > 0.0:
            r_heal = (cur_health - state.last_health) * HEAL_SCALE
            state.total_healing_rew += r_heal
        else:
            # Revived from 0 HP = death counted
            state.died_count += 1
    state.last_health     = cur_health
    state.last_party_size = cur_party

    # --- Death Penalty (cumulative, emit delta) ---
    cur_dead_abs = -0.1 * state.died_count
    r_dead       = cur_dead_abs - state.dead_abs
    state.dead_abs = cur_dead_abs

    # --- Scaled Total ---
    raw = r_event + r_level + r_badge + r_op_lvl + r_heal + r_dead
    total = raw * REWARD_SCALE

    return WhiddenStepReward(
        event=r_event, level=r_level, badge=r_badge,
        op_lvl=r_op_lvl, heal=r_heal, dead=r_dead,
        total=total,
    )
