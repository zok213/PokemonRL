"""
test_reward_machine.py — Unit Tests for 16-State Reward Machine
==============================================================
Verifies Healing Trap immunity, LTL milestone graph transitions, and PBRS potential monotonicity.
"""

import pytest
from pokemon_rl.agent.reward_machine import (
    RewardMachine,
    WRAMReader,
    RM_STATES,
    RM_WIN_STATE,
    RM_START_STATE,
    RM_TRANSITION_REWARDS,
    RM_STATE_POTENTIAL,
    build_wram_with_badges_and_flags,
    RAMMap,
)


def test_rm_initial_state():
    rm = RewardMachine()
    assert rm.current_state == RM_START_STATE
    assert rm.total_rm_reward == 0.0
    assert not rm.is_terminal()


def test_healing_trap_immunity():
    """
    HEALING TRAP IMMUNITY THEOREM:
    sigma_R(u, u) = 0.0 for ALL states in U.
    In-state actions (healing at PokeCenter, grinding, menu cycling) emit strictly 0.0 reward.
    """
    rm = RewardMachine()
    empty_wram = build_wram_with_badges_and_flags(badges=0)
    reader = WRAMReader(empty_wram)

    for step in range(250):
        new_state, reward = rm.step(reader)
        assert reward == 0.0, f"Reward emitted on self-loop at step {step}: {reward}"
        assert new_state == RM_START_STATE

    assert rm.total_rm_reward == 0.0


def test_oaks_parcel_transition():
    rm = RewardMachine()
    wram = build_wram_with_badges_and_flags(
        badges=0,
        event_flags={RAMMap.EVENT_GOT_OAKS_PARCEL: True}
    )
    reader = WRAMReader(wram)
    new_state, reward = rm.step(reader)
    assert new_state == "U1_OAKS_PARCEL"
    assert reward == 5.0
    assert rm.milestone_depth() == 1


def test_badge_transition_sequence():
    """Simulate acquiring Brock's Boulder Badge from Pokedex state."""
    rm = RewardMachine()
    rm.current_state = "U2_POKEDEX"

    wram = build_wram_with_badges_and_flags(
        badges=RAMMap.BADGE_BOULDER,
        event_flags={
            RAMMap.EVENT_BEAT_BROCK: True,
            RAMMap.EVENT_GOT_BOULDER_BADGE: True,
        }
    )
    reader = WRAMReader(wram)
    new_state, reward = rm.step(reader)
    assert new_state == "U3_BOULDER_BADGE"
    assert reward == 100.0


def test_pbrs_potential_monotonicity():
    """PBRS potential must strictly increase along canonical quest progression."""
    progression = [
        "U0_PALLET_TOWN",
        "U1_OAKS_PARCEL",
        "U2_POKEDEX",
        "U3_BOULDER_BADGE",
        "U4_CASCADE_BADGE",
        "U5_THUNDER_BADGE",
        "U6_RAINBOW_BADGE",
        "U7_SOUL_BADGE",
        "U8_MARSH_BADGE",
        "U9_VOLCANO_BADGE",
        "U10_EARTH_BADGE",
    ]
    potentials = [RM_STATE_POTENTIAL[s] for s in progression]
    for i in range(len(potentials) - 1):
        assert potentials[i + 1] > potentials[i]
        # Shaping F = gamma * Phi(s') - Phi(s) must be positive at gamma=0.997
        f_shaping = 0.997 * potentials[i + 1] - potentials[i]
        assert f_shaping > 0.0
