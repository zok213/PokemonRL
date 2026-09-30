"""
interactive/reward/anti_stagnation.py — Anti-Stagnation & Menu Penalty
=======================================================================
Detects two failure modes that cause the agent to get stuck:

  1. POSITION STAGNATION / LOCAL OSCILLATION:
     Agent fails to discover new tiles, or oscillates between <=2 tiles
     for more than STAGNATION_THRESHOLD steps. Emits a negative reward
     to push the agent out of dead-ends.

  2. MENU LOCK:
     Agent is inside a menu/dialogue for more than MENU_THRESHOLD
     consecutive steps. Emits a negative reward to discourage idle text farming.

These are separate from WhiddenReward so each can be tuned or inspected independently.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque
from interactive.wram.reader import read_map_id, read_xy, is_menu_or_text_active, is_in_battle

# Tuning parameters
STAGNATION_THRESHOLD = 20    # steps without new ground before penalty fires (was 25)
# Escalating stagnation penalty: starts at -0.02, grows to -0.10 after 100 stagnation steps.
# A flat penalty (-0.01) was too weak against the heal reward (+0.4). Escalation forces
# proportional pressure the longer the agent stays stuck.
STAGNATION_BASE_PENALTY  = -0.02
STAGNATION_PENALTY       = STAGNATION_BASE_PENALTY  # alias for tests/evaluators
STAGNATION_SCALE_PER_100 = -0.08  # additional penalty per 100 steps of stagnation

MENU_THRESHOLD = 1           # consecutive menu steps before penalty fires (was 3)
MENU_PENALTY   = -0.05       # per step inside menu lock (was -0.01, raised 5×)


@dataclass
class AntiStagnationState:
    """Mutable state for stagnation & menu tracking. Reset on env reset."""
    stagnation_steps: int = 0
    menu_lock_steps:  int = 0
    recent_coords:    deque = field(default_factory=lambda: deque(maxlen=16))


@dataclass
class AntiStagnationReward:
    """Per-step anti-stagnation reward components."""
    stagnation: float = 0.0
    menu_pen:   float = 0.0
    total:      float = 0.0

    def asdict(self) -> dict:
        return {"stagnation": self.stagnation, "menu_pen": self.menu_pen}


def compute_anti_stagnation(
    mem, state: AntiStagnationState, is_novel: bool = False
) -> AntiStagnationReward:
    """
    Compute stagnation + menu penalties from live WRAM.
    Mutates state in place.

    Args:
        mem:      PyBoy memory object
        state:    mutable AntiStagnationState
        is_novel: True if the current tile is a newly discovered unique coordinate

    Returns:
        AntiStagnationReward with penalty values.
    """
    cur_map  = read_map_id(mem)
    cur_x, cur_y = read_xy(mem)
    cur_tile = (cur_map, cur_x, cur_y)
    in_battle = is_in_battle(mem)
    in_menu   = is_menu_or_text_active(mem)

    # --- Position Stagnation / Oscillation ---
    state.recent_coords.append(cur_tile)
    r_stagnation = 0.0

    if is_novel:
        # Novel tile discovered — reset stagnation counter
        state.stagnation_steps = 0
    else:
        # If moving between <= 2 distinct tiles in the last 16 steps, count as stagnation
        unique_recent = len(set(state.recent_coords))
        if unique_recent <= 2:
            state.stagnation_steps += 1
        elif state.stagnation_steps > 0:
            state.stagnation_steps -= 1  # slowly decay when moving across distinct tiles

        if state.stagnation_steps > STAGNATION_THRESHOLD:
            # Escalating penalty: −0.02 base, grows by −0.08 per 100 stagnation steps
            # This makes staying stuck progressively more expensive than any heal reward.
            r_stagnation = (
                STAGNATION_BASE_PENALTY
                + STAGNATION_SCALE_PER_100 * min(state.stagnation_steps / 100.0, 1.0)
            )

    # --- Menu Lock ---
    r_menu_pen = 0.0
    if in_menu and not in_battle:
        state.menu_lock_steps += 1
        if state.menu_lock_steps >= MENU_THRESHOLD:
            r_menu_pen = MENU_PENALTY
    else:
        state.menu_lock_steps = 0

    total = r_stagnation + r_menu_pen
    return AntiStagnationReward(stagnation=r_stagnation, menu_pen=r_menu_pen, total=total)


def is_stagnant(state: AntiStagnationState) -> bool:
    """True if agent has been at same tile beyond threshold."""
    return state.stagnation_steps > STAGNATION_THRESHOLD


def is_menu_locked(state: AntiStagnationState) -> bool:
    """True if agent has been in menu beyond threshold."""
    return state.menu_lock_steps >= MENU_THRESHOLD
