"""
test_v2_jev_hybrid_policy.py
============================
Unit tests for BaselineV2JevSteeredPolicy:
Verifies that 26.2M-step pretrained Baseline V2 weights are cleanly loaded,
and that Jev-style affordance steering resolves combat legality and overworld waypoints.
"""

import pytest
import torch
from pathlib import Path

from pokemon_rl.agent.v2_jev_hybrid_policy import (
    BaselineV2JevSteeredPolicy,
    HybridDecision,
    ACTION_NAMES,
)


def test_v2_hybrid_policy_loads_checkpoint():
    """Verify that poke_26214400.zip is found and loaded into memory."""
    policy = BaselineV2JevSteeredPolicy()
    assert policy.checkpoint_path is not None
    assert policy.checkpoint_path.exists()
    assert policy.v2_model is not None


def test_v2_hybrid_policy_overworld_steering():
    """
    Verify that in Pallet Town (Map 0), when target is Route 1 (UP),
    the affordance head steers the policy north even if V2 raw prior favors another direction.
    """
    policy = BaselineV2JevSteeredPolicy()
    snapshot = {
        "map_id": 0,
        "x": 10,
        "y": 8,
        "badges": 0,
        "in_battle": False,
        "text_box_id": 0,
        "active_mon": {"hp": 20, "max_hp": 20, "level": 5}
    }

    decision = policy.decide(snapshot, obs=None)
    assert isinstance(decision, HybridDecision)
    assert decision.action_name == "UP"
    assert decision.steered_probs["UP"] > 0.80
    assert "north_to_route1" in decision.affordance_situation


def test_v2_hybrid_policy_trainer_combat_legality():
    """
    Verify that in a trainer battle (battle_type == 2),
    RUN and B are suppressed and attack (A) dominates.
    """
    policy = BaselineV2JevSteeredPolicy()
    snapshot = {
        "map_id": 51,
        "x": 10,
        "y": 10,
        "badges": 0,
        "in_battle": True,
        "battle_type": 2,  # Trainer battle!
        "text_box_id": 0,
        "opponent_species": 0x70,  # Weedle
        "opponent_hp": 15,
        "active_mon": {"hp": 10, "max_hp": 20, "level": 6, "moves": [10, 52]}
    }

    decision = policy.decide(snapshot, obs=None)
    assert decision.action_name == "A"
    assert decision.steered_probs["A"] > 0.90
    assert decision.steered_probs["B"] < 0.05
    assert decision.steered_probs["START"] < 0.01


def test_v2_hybrid_policy_anti_stagnation_scaling():
    """Verify that when position remains stuck, steering weight alpha dynamically increases."""
    policy = BaselineV2JevSteeredPolicy(default_alpha=1.0)
    snapshot = {
        "map_id": 12,
        "x": 8,
        "y": 28,
        "badges": 0,
        "in_battle": False,
        "text_box_id": 0,
    }

    # Step 1: initial position
    d1 = policy.decide(snapshot, obs=None)
    assert policy.stuck_counter == 0
    assert d1.steering_weight == 1.0

    # Step 2: same position -> stuck_counter increments
    d2 = policy.decide(snapshot, obs=None)
    assert policy.stuck_counter == 1

    # Step 3: same position again -> stuck_counter = 2, alpha scales up
    d3 = policy.decide(snapshot, obs=None)
    assert policy.stuck_counter == 2
    assert d3.steering_weight > 1.0
    assert "Anti-Stagnation Active" in d3.affordance_situation


def test_v2_hybrid_policy_dialogue_loop_breaker():
    """Verify that prolonged text dialogues switch from A to B to prevent infinite re-trigger loops."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 39,
        "x": 2,
        "y": 4,
        "in_battle": False,
        "text_box_active": True,
        "text_box_id": 1,
    }

    # Steps 1 to 3: advance text rapidly with Button A
    for step in range(1, 4):
        d = policy.decide(snap, obs=None)
        assert d.action_name == "A"
        assert "Advancing text with Button A" in d.affordance_situation

    # Step 4: anti-loop dismiss latch engages, chooses B
    d4 = policy.decide(snap, obs=None)
    assert d4.action_name == "B"
    assert "Anti-loop dismiss active" in d4.affordance_situation


def test_v2_hybrid_policy_post_dialogue_anti_retrigger_shield():
    """Verify that when dialogue closes, Button A is heavily suppressed to allow stepping away."""
    policy = BaselineV2JevSteeredPolicy()
    snap_dialogue = {
        "map_id": 39,
        "x": 2,
        "y": 4,
        "in_battle": False,
        "text_box_active": True,
        "text_box_id": 1,
    }
    policy.decide(snap_dialogue, obs=None)

    # Dialogue closes on next frame
    snap_closed = {
        "map_id": 39,
        "x": 2,
        "y": 4,
        "in_battle": False,
        "text_box_active": False,
        "text_box_id": 0,
    }
    d_closed = policy.decide(snap_closed, obs=None)
    assert "Anti-Retrigger Active" in d_closed.affordance_situation
    assert d_closed.action_name != "A"
    assert d_closed.steered_probs["A"] < 0.05
    # Subgoal in Blue's House at (2, 4) is south (2, 7) -> DOWN
    assert d_closed.action_name == "DOWN"


def test_v2_hybrid_policy_blues_house_exit_navigation():
    """Verify that inside Blue's House (Map 39), Red is guided south to the exit mat at (2, 7)."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 39,
        "x": 2,
        "y": 5,
        "in_battle": False,
        "text_box_id": 0,
    }
    d = policy.decide(snap, obs=None)
    assert d.action_name == "DOWN"
    assert "exit_blues_house" in d.affordance_situation


def test_v2_hybrid_policy_move_selection_navigation():
    """Verify that in trainer combat, the policy decisively steers Attack with Button A when cursor is aligned with optimal move."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 54,
        "x": 4,
        "y": 2,
        "in_battle": True,
        "battle_type": 2,
        "battle_menu_y": 7,  # Move selection menu (wTopMenuItemY == 7)
        "battle_cur_item": 2,  # Cursor aligned on Move 3 (Ember=52, 0-based index 2)
        "move_pps": [35, 40, 25, 30],
        "active_mon": {"hp": 40, "max_hp": 40, "level": 15, "moves": [10, 45, 52, 43]}
    }
    d0 = policy.decide(snap, obs=None)
    assert d0.action_name == "A"
    assert "attack with Button A" in d0.affordance_situation
    assert d0.steered_probs["A"] > 0.90


def test_v2_hybrid_policy_depleted_pp_move_navigation():
    """Verify that when Move 1 is depleted (0 PP), policy navigates cursor to Move 3 (Ember)."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 54,
        "x": 4,
        "y": 2,
        "in_battle": True,
        "battle_type": 2,
        "battle_menu_y": 7,  # Move selection menu (wTopMenuItemY == 7)
        "battle_cur_item": 0,  # Cursor on Move 1 (0-based index 0)
        "move_pps": [0, 40, 25, 30],  # Move 1 has 0 PP, Move 3 (Ember=52) has 25 PP
        "active_mon": {"hp": 25, "max_hp": 40, "level": 15, "moves": [10, 45, 52, 43]}
    }
    d = policy.decide(snap, obs=None)
    assert d.action_name == "DOWN"
    assert "Move DOWN" in d.affordance_situation and "Move 3" in d.affordance_situation
    assert d.steered_probs["DOWN"] > 0.85

    # When cursor reaches Move 3 (0-based index 2), it presses A
    snap["battle_cur_item"] = 2
    d_aligned = policy.decide(snap, obs=None)
    assert d_aligned.action_name == "A"
    assert "attack with Button A" in d_aligned.affordance_situation


def test_v2_hybrid_policy_battle_stall_unjamming():
    """Verify that if battle state stalls for >= 6 steps, unjamming watchdog activates."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 54,
        "x": 4,
        "y": 2,
        "in_battle": True,
        "battle_type": 2,
        "battle_cur_item": 0,
        "opponent_hp": 30,
        "active_mon": {"hp": 20, "max_hp": 40, "level": 15, "moves": [10, 45, 52, 43]}
    }
    # Execute 7 steps (first step sets baseline, next 6 steps increment counter to 6)
    for _ in range(7):
        d = policy.decide(snap, obs=None)

    assert policy.battle_stall_counter >= 6
    assert "Combat: Menu/PP stall detected" in d.affordance_situation


def test_v2_hybrid_policy_bubble_move_priority():
    """Verify that Bubble (Move 145) is prioritized over Tackle (33) and Tail Whip (39)."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 54,  # Pewter Gym
        "x": 4,
        "y": 2,
        "in_battle": True,
        "battle_type": 2,
        "battle_menu_y": 7,  # In move selection menu
        "battle_cur_item": 0,  # Cursor on Move 1 (Tackle, 0-based index 0)
        "move_pps": [35, 30, 30, 0],
        "active_mon": {"hp": 28, "max_hp": 28, "level": 9, "moves": [33, 39, 145, 0]}
    }
    # Should navigate DOWN to Move 3 (Bubble)
    d = policy.decide(snap, obs=None)
    assert d.action_name == "DOWN"
    assert "Selecting Move 3" in d.affordance_situation
    assert d.steered_probs["DOWN"] > 0.85

    # Once cursor reaches Move 3 (0-based index 2), should press Button A to unleash Bubble
    snap["battle_cur_item"] = 2
    d2 = policy.decide(snap, obs=None)
    assert d2.action_name == "A"
    assert "Executing attack with Button A" in d2.affordance_situation


def test_v2_hybrid_policy_warp_doorway_boost():
    """Verify that at warp doorway thresholds (e.g. Pokécenter entrance), steering weight scales to >=2.8 and forces entry."""
    policy = BaselineV2JevSteeredPolicy()
    snap = {
        "map_id": 2,  # Pewter City
        "x": 13,
        "y": 26,  # Standing right before Pokécenter door at (13, 25)
        "badges": 0,
        "in_battle": False,
        "text_box_id": 0,
        "active_mon": {"hp": 0, "max_hp": 28, "level": 9, "moves": [33, 39, 145, 0]}  # Fainted mon needs healing
    }
    d = policy.decide(snap, obs=None)
    assert d.action_name == "UP"
    assert d.steering_weight >= 2.8
    assert "Warp Doorway Transition" in d.affordance_situation
    assert d.steered_probs["UP"] > 0.90

