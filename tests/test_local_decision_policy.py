"""
test_local_decision_policy.py
=============================
Unit tests for LocalAffordancePolicy, JevosLocalClient, and SMDPDecisionHarness.
Verifies 100% local execution, probability normalization, and RL gradient backward pass.
"""

import pytest
import torch
import torch.nn.functional as F

from pokemon_rl.agent.local_decision_policy import (
    LocalAffordancePolicy,
    JevosLocalClient,
    SMDPDecisionHarness,
    AffordanceBranch,
    LocalDecision,
)


def test_local_affordance_policy_forward():
    policy = LocalAffordancePolicy(state_dim=32, hidden_dim=64, embed_dim=32)
    state = torch.randn(1, 32)
    options = {
        "use_move_scratch": "attack with Scratch, a Normal move, neutral against Water",
        "use_move_growl": "use Growl, lowers opponent Attack",
        "switch_to_squirtle": "switch to Squirtle, level 5, healthy",
    }

    probs, aux, logits = policy.forward(state, options)

    assert len(probs) == 3
    assert set(probs.keys()) == set(options.keys())
    assert pytest.approx(sum(probs.values()), rel=1e-5) == 1.0
    for v in probs.values():
        assert 0.0 <= v <= 1.0

    assert "faints_this_turn" in aux
    assert 0.0 <= aux["faints_this_turn"] <= 1.0
    assert "should_flee" in aux
    assert 0.0 <= aux["should_flee"] <= 1.0


def test_local_affordance_policy_decide_latency():
    policy = LocalAffordancePolicy(state_dim=32, hidden_dim=64, embed_dim=32)
    state = torch.zeros(1, 32)
    options = {
        "use_move_ember": "attack with Ember, a Fire move, super effective against Grass",
        "use_move_scratch": "attack with Scratch, a Normal move",
    }

    decision = policy.decide(state, options)
    assert isinstance(decision, LocalDecision)
    assert decision.selected_option in options
    assert decision.confidence >= 0.5
    assert decision.latency_ms < 50.0  # Runs in single-digit ms
    assert decision.model_source == "local_pytorch"


def test_local_policy_reinforcement_learning_backward():
    """Verify that the local policy can be trained via local policy gradients / RL."""
    policy = LocalAffordancePolicy(state_dim=32, hidden_dim=64, embed_dim=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    state = torch.randn(1, 32)
    options = {
        "option_a": "good choice, super effective",
        "option_b": "bad choice, no effect",
    }

    optimizer.zero_grad()
    probs, aux, logits = policy.forward(state, options)

    # Simulate an RL reward signal: reward option_a (+1.0) and penalize option_b (-1.0)
    advantage = torch.tensor([1.0, -1.0])
    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
    loss = -(log_probs * advantage).sum()

    loss.backward()
    optimizer.step()

    # Verify policy parameters (state_encoder and option_encoder) received gradients
    for name, p in policy.named_parameters():
        if "auxiliary" not in name and p.requires_grad:
            assert p.grad is not None, f"Parameter {name} had no gradient"


def test_smdp_decision_harness_integration():
    harness = SMDPDecisionHarness()
    wram_snapshot = {
        "map_id": 1,
        "x": 10,
        "y": 12,
        "party_count": 2,
        "badges": 1,
        "in_battle": True,
        "active_mon": {"hp": 15, "max_hp": 30, "level": 12},
    }

    branch = AffordanceBranch(
        branch_type="battle",
        state_payload=wram_snapshot,
        legal_options={
            "use_move_thunderbolt": "attack with Thunderbolt, super effective",
            "run_away": "escape battle",
        },
    )

    decision = harness.step_branch(branch, wram_snapshot)
    assert decision.selected_option in branch.legal_options
    assert harness.decision_count == 1
    assert decision.latency_ms >= 0.0


def test_jevos_local_client_fallback_when_offline():
    """Verify that JevosLocalClient gracefully falls back when no local server is listening."""
    client = JevosLocalClient(base_url="http://127.0.0.1:9999", timeout=0.1)
    assert not client.is_available()

    branch = AffordanceBranch(
        branch_type="tie",
        state_payload={"x": 5, "y": 5},
        legal_options={"step_left": "walk left", "step_up": "walk up"},
    )
    decision = client.decide(branch)
    assert decision.selected_option in ("step_left", "step_up")
    assert "fallback" in decision.model_source
