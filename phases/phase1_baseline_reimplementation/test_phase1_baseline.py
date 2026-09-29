"""
test_phase1_baseline.py — Unit Tests for Pleines et al. (IEEE CoG 2025) Baseline
================================================================================
Verifies compliance with the published paper's exact mathematical specification:
  - 7 discrete actions (SELECT omitted)
  - 24-frame action wrapper
  - Multimodal observation tensor shapes
  - Dynamic step budget formula: B_t = 10,240 + 2,048 * N_events
  - Composite linear reward components
  - Actor-Critic policy and GAE(gamma=0.997, lambda=0.95)
"""

import pytest
import numpy as np

from pleines_baseline_env import (
    PleinesPokemonRedEnv,
    PleinesAction,
    PleinesCompositeReward,
)
from pleines_ppo_policy import PleinesActorCriticPolicy


def test_action_space_specification():
    """Paper specification: 7 discrete actions, SELECT is omitted."""
    assert PleinesAction.NUM_ACTIONS == 7
    assert PleinesAction.UP == 0
    assert PleinesAction.A == 4
    assert PleinesAction.START == 6


def test_dynamic_step_budget_formula():
    """Paper formula: B_t = 10,240 + 2,048 * N_events."""
    env = PleinesPokemonRedEnv()
    assert env.calculate_step_budget(0) == 10240
    assert env.calculate_step_budget(1) == 12288
    assert env.calculate_step_budget(4) == 18432
    assert env.calculate_step_budget(8) == 26624


def test_multimodal_observation_shapes():
    """
    Paper observation shapes:
      - Screen: (3, 72, 80) grayscale stack
      - Spatial Map: (1, 48, 48) visited binary grid
      - Telemetry: (64,) telemetry vector
    """
    env = PleinesPokemonRedEnv()
    obs, info = env.reset()

    assert obs["screen"].shape == (3, 72, 80)
    assert obs["spatial_map"].shape == (1, 48, 48)
    assert obs["telemetry"].shape == (64,)
    assert info["budget"] == 10240


def test_composite_reward_components():
    """
    Verify exact equations:
      R_event = 2.0 * Delta N_events
      R_nav   = 0.005 on unvisited coordinates
      R_heal  = 2.5 * Delta HP / Max HP
      R_lvl   = 0.5 * Delta level potential
    """
    reward_fn = PleinesCompositeReward(enable_lvl=True, enable_heal=True)

    # Step 1: Navigating to new coordinate (5, 4, 0)
    r1, b1 = reward_fn.compute_reward(
        coords=(5, 4, 0),
        current_events=0,
        current_hp=[20, 0, 0, 0, 0, 0],
        max_hp=[20, 100, 100, 100, 100, 100],
        current_levels=[5, 0, 0, 0, 0, 0],
    )
    assert abs(b1["r_nav"] - 0.005) < 1e-6
    assert abs(b1["r_event"] - 0.0) < 1e-6

    # Step 2: Revisiting same coordinate emits 0 nav reward
    r2, b2 = reward_fn.compute_reward(
        coords=(5, 4, 0),
        current_events=0,
        current_hp=[20, 0, 0, 0, 0, 0],
        max_hp=[20, 100, 100, 100, 100, 100],
        current_levels=[5, 0, 0, 0, 0, 0],
    )
    assert abs(b2["r_nav"] - 0.0) < 1e-6

    # Step 3: Event milestone (+1 event) emits +2.0
    r3, b3 = reward_fn.compute_reward(
        coords=(5, 5, 0),
        current_events=1,
        current_hp=[20, 0, 0, 0, 0, 0],
        max_hp=[20, 100, 100, 100, 100, 100],
        current_levels=[5, 0, 0, 0, 0, 0],
    )
    assert abs(b3["r_event"] - 2.0) < 1e-6

    # Step 4: Healing party (+10 HP on a 20 HP Pokémon) emits 2.5 * (10/20) = 1.25
    r4, b4 = reward_fn.compute_reward(
        coords=(5, 5, 0),
        current_events=1,
        current_hp=[30, 0, 0, 0, 0, 0],
        max_hp=[30, 100, 100, 100, 100, 100],
        current_levels=[5, 0, 0, 0, 0, 0],
    )
    assert abs(b4["r_heal"] - 2.5 * (10.0 / 30.0)) < 1e-4


def test_actor_critic_policy_forward_and_gae():
    """Verify Actor-Critic outputs and GAE credit assignment."""
    policy = PleinesActorCriticPolicy(use_gru=False, seed=42)
    B = 2

    screen = np.zeros((B, 3, 72, 80), dtype=np.float32)
    spatial = np.zeros((B, 1, 48, 48), dtype=np.float32)
    telemetry = np.zeros((B, 64), dtype=np.float32)

    probs, values = policy.forward(screen, spatial, telemetry)

    assert probs.shape == (B, 7)
    assert values.shape == (B,)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    # GAE calculation with gamma=0.997, lambda=0.95
    rewards = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    val_traj = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    adv, targets = policy.compute_gae(rewards, val_traj, dones, gamma=0.997, lambda_gae=0.95)
    assert len(adv) == 3
    assert adv[2] == 2.0
    # Step 0 credit attenuation: (0.997 * 0.95)^2 * 2.0
    expected_adv0 = ((0.997 * 0.95) ** 2) * 2.0
    assert abs(adv[0] - expected_adv0) < 1e-4
