"""
test_grpo_steering_and_jax_bridge.py
====================================
Comprehensive tests for:
  1. DecisionForkGRPO: Critic-free Group Relative Policy Optimization, G=8 sibling sampling,
     zero-mean advantage normalization, and reverse-KL loss regularization.
  2. Integration of learned steering gain alpha into BaselineV2JevSteeredPolicy.
  3. PokeJAXVectorBridge: Shared-memory zero-copy buffer vectorization across parallel environments.
  4. 4-Tier Verification Suite (Karten, Appapogu, & Jin, COLM 2026):
     - Tier 1: Invariant and property tests
     - Tier 2: Deterministic interaction tests
     - Tier 3: Rollout total variation divergence
     - Tier 4: Sim-to-sim policy transfer
"""

import math
import numpy as np
import pytest
import torch

from pokemon_rl.agent.grpo_steering_policy import SteeringGainNetwork, DecisionForkGRPO
from pokemon_rl.agent.v2_jev_hybrid_policy import BaselineV2JevSteeredPolicy
from pokemon_rl.env.poke_jax_bridge import SharedMemoryVectorBuffer, PokeJAXVectorBridge, VectorBridgeVerifier


# =============================================================================
# 1. DECISION FORK GRPO TESTS
# =============================================================================

def test_grpo_feature_extraction():
    state = {
        "dx_goal": 10.0,
        "dy_goal": -5.0,
        "stagnation_count": 5,
        "action_entropy": 0.8,
        "in_battle": False,
        "is_warp": True,
        "is_interaction": False,
    }
    feat = DecisionForkGRPO.extract_features(state)
    assert feat.shape == (1, 8)
    # Check bounded values
    assert -1.0 <= feat[0, 0].item() <= 1.0  # dx
    assert -1.0 <= feat[0, 1].item() <= 1.0  # dy
    assert 0.0 <= feat[0, 2].item() <= 1.0   # dist
    assert 0.0 <= feat[0, 3].item() <= 1.0   # stagnation
    assert 0.0 <= feat[0, 4].item() <= 1.0   # entropy
    assert feat[0, 5].item() == 0.0          # in_battle
    assert feat[0, 6].item() == 1.0          # is_warp
    assert feat[0, 7].item() == 0.0          # is_interaction


def test_grpo_group_sampling_and_advantage_normalization():
    grpo = DecisionForkGRPO(group_size=8)
    feat = torch.zeros(1, 8)

    alphas, log_probs = grpo.sample_group(feat)
    assert alphas.shape == (8, 1)
    assert log_probs.shape == (8,)

    # All alphas bounded in [1.0, 4.0]
    assert torch.all(alphas >= 1.0)
    assert torch.all(alphas <= 4.0)

    # Simulated sibling rewards across G=8 rollouts
    rewards = [1.5, 0.2, -0.4, 2.1, 0.8, 1.2, -0.1, 0.5]
    advantages = grpo.compute_group_advantages(rewards)
    assert advantages.shape == (8,)

    # Mathematical property: sum of group-relative advantages is identically zero
    assert abs(float(torch.sum(advantages).item())) < 1e-5
    # Unit standard deviation
    assert abs(float(torch.std(advantages, unbiased=False).item()) - 1.0) < 1e-4


def test_grpo_policy_optimization_step():
    grpo = DecisionForkGRPO(group_size=8, learning_rate=1e-3)
    feat = torch.randn(1, 8)

    alphas, log_probs_old = grpo.sample_group(feat)
    rewards = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]

    metrics = grpo.update_policy(feat, alphas, log_probs_old, rewards)

    assert "loss_total" in metrics
    assert "loss_surr" in metrics
    assert "kl_div" in metrics
    assert not math.isnan(metrics["loss_total"])
    assert not math.isinf(metrics["loss_total"])


def test_grpo_integration_with_v2_jev_policy():
    grpo = DecisionForkGRPO(group_size=8)
    policy = BaselineV2JevSteeredPolicy(checkpoint_path=None, grpo_controller=grpo)

    snapshot = {
        "map_id": 12,
        "x": 10,
        "y": 20,
        "badges": 0,
        "hp": 20,
        "max_hp": 20,
        "in_battle": False,
        "text_box_id": 0,
    }

    decision = policy.decide(snapshot)
    assert decision.action_idx in range(len(policy.combat_controller.GEN1_MOVES_MAP) if False else 7)
    assert "GRPO Learned Alpha" in decision.affordance_situation
    assert 1.0 <= decision.steering_weight <= 4.0


# =============================================================================
# 2. POKEJAX VECTOR BRIDGE & 4-TIER VERIFICATION SUITE
# =============================================================================

def test_shared_memory_vector_buffer_contiguity():
    num_envs = 64
    buf = SharedMemoryVectorBuffer(num_envs=num_envs, obs_shape=(36, 40))
    buf.reset_all(map_id=1, x=15, y=20, badges=1)

    assert buf.obs_buffer.shape == (64, 36, 40)
    assert buf.telemetry_buffer.shape == (64, 10)
    assert buf.obs_buffer.flags.c_contiguous is True
    assert buf.telemetry_buffer.flags.c_contiguous is True
    assert np.all(buf.telemetry_buffer[:, 0] == 1)
    assert np.all(buf.telemetry_buffer[:, 1] == 15)
    assert np.all(buf.telemetry_buffer[:, 2] == 20)


def test_poke_jax_vector_bridge_batched_step():
    num_envs = 32
    bridge = PokeJAXVectorBridge(num_envs=num_envs)

    actions = np.zeros(num_envs, dtype=np.int8)  # Action 0: Down
    obs, rews, dones, telem = bridge.step(actions)

    assert obs.shape == (32, 36, 40)
    assert rews.shape == (32,)
    assert dones.shape == (32,)
    assert len(telem) == 32
    # Y-coordinate should have increased by 1
    assert telem[0]["y"] == 11
    assert telem[0]["steps"] == 1


def test_4_tier_verification_tier1_property_invariants():
    bridge = PokeJAXVectorBridge(num_envs=64)
    res = VectorBridgeVerifier.verify_tier1_property_invariants(bridge)

    assert res["tier1_coords_valid"] is True
    assert res["tier1_badges_valid"] is True
    assert res["tier1_hp_valid"] is True
    assert res["tier1_memory_contiguous"] is True
    assert res["tier1_passed"] is True


def test_4_tier_verification_tier2_interaction_determinism():
    res = VectorBridgeVerifier.verify_tier2_interaction_determinism(num_envs=16, num_steps=25)
    assert res["tier2_identical_parity"] is True
    assert res["tier2_passed"] is True


def test_4_tier_verification_tier3_rollout_divergence():
    bridge_a = PokeJAXVectorBridge(num_envs=16, seed=100)
    bridge_b = PokeJAXVectorBridge(num_envs=16, seed=100)

    res = VectorBridgeVerifier.verify_tier3_rollout_divergence(bridge_a, bridge_b, num_steps=20)
    assert res["tier3_total_variation"] == 0.0
    assert res["tier3_passed"] is True


def test_4_tier_verification_tier4_sim_to_sim_policy_transfer():
    bridge = PokeJAXVectorBridge(num_envs=16)

    # Simple heuristic policy function
    def policy_fn(telemetry: dict) -> int:
        if telemetry["y"] < 15:
            return 0  # Down
        return 3      # Right

    res = VectorBridgeVerifier.verify_tier4_policy_transfer(bridge, policy_fn, num_steps=15)
    assert res["tier4_valid_transitions"] is True
    assert res["tier4_passed"] is True
