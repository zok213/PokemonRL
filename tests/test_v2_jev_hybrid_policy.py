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
