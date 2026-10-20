"""
test_baseline_v2.py — Unit & Integration Test Suite for Baseline V2 Re-Implementation
======================================================================================
Verifies all mathematical formulations and architectural components from:
  Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920 / PWhiddy V2
"""

import pytest
import numpy as np

import pokemon_rl.baseline_v2 as b2


def test_baseline_v2_constants_and_action_space():
    """Verifies action space has exactly 7 discrete actions omitting SELECT."""
    assert b2.PleinesAction.NUM_ACTIONS == 7
    expected_actions = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
    for act_name in expected_actions:
        assert hasattr(b2.PleinesAction, act_name)
    assert not hasattr(b2.PleinesAction, "SELECT")

    # Stride verification: 8 hold + 16 release = 24 frames
    assert b2.FRAME_HOLD == 8
    assert b2.FRAME_RELEASE == 16
    assert b2.DECISION_STRIDE == 24


def test_baseline_v2_dynamic_budget():
    """Verifies dynamic budget formula: B_t = 10,240 + 2,048 * N_events."""
    env = b2.PleinesBaselineEnv()
    assert env.calculate_step_budget(0) == 10240
    assert env.calculate_step_budget(1) == 10240 + 2048 == 12288
    assert env.calculate_step_budget(4) == 10240 + 2048 * 4 == 18432


def test_baseline_v2_reward_components():
    """Verifies all 4 terms of composite linear reward: R = R_event + R_nav + R_heal + R_lvl."""
    # 1. Event reward: +2.0 per event
    assert b2.compute_event_reward(current_events=2, last_events=1) == 2.0
    assert b2.compute_event_reward(current_events=1, last_events=1) == 0.0

    # 2. Navigation reward: +0.005 on first visit, 0.0 on revisit
    visited = set()
    r_nav1, is_new1 = b2.compute_nav_reward((5, 4, 0), visited)
    assert r_nav1 == 0.005 and is_new1
    visited.add((5, 4, 0))
    r_nav2, is_new2 = b2.compute_nav_reward((5, 4, 0), visited)
    assert r_nav2 == 0.0 and not is_new2

    # 3. Healing reward: +2.5 * Delta HP / MaxHP
    # 10 HP gained on 20 MaxHP -> ratio = 0.5 -> 2.5 * 0.5 = 1.25
    r_heal = b2.compute_heal_reward(
        current_hp=[20, 0, 0, 0, 0, 0],
        last_hp=[10, 0, 0, 0, 0, 0],
        max_hp=[20, 100, 100, 100, 100, 100],
    )
    assert pytest.approx(r_heal, 1e-5) == 1.25

    # 4. Level-up reward: piece-wise linear
    assert b2.compute_level_potential([5, 0, 0, 0, 0, 0]) == 5.0
    assert b2.compute_level_potential([22, 0, 0, 0, 0, 0]) == 22.0
    # Level sum = 26: 22 + (26-22)/4 = 23.0
    assert b2.compute_level_potential([26, 0, 0, 0, 0, 0]) == 23.0
    r_lvl = b2.compute_level_reward([6, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0])
    assert pytest.approx(r_lvl, 1e-5) == 0.5 * 1.0


def test_baseline_v2_env_gym_lifecycle():
    """Verifies standard Gym lifecycle (reset -> step -> truncate) and observation shapes."""
    env = b2.PleinesBaselineEnv()
    obs, info = env.reset()

    # Observation contract
    assert "screen" in obs
    assert "spatial_map" in obs
    assert "telemetry" in obs

    assert obs["screen"].shape == (3, 72, 80)
    assert obs["spatial_map"].shape == (1, 48, 48)
    assert obs["telemetry"].shape == (64,)

    # Step
    next_obs, reward, terminated, truncated, step_info = env.step(int(b2.PleinesAction.RIGHT))
    assert not terminated
    assert not truncated
    assert "reward_breakdown" in step_info
    assert "r_nav" in step_info["reward_breakdown"]


def test_baseline_v2_numpy_actor_critic():
    """Verifies NumPy lightweight actor-critic forward pass and probability distribution."""
    policy = b2.PleinesActorCriticNumPy(seed=42)
    B = 2
    screen = np.zeros((B, 3, 72, 80), dtype=np.float32)
    spatial = np.zeros((B, 1, 48, 48), dtype=np.float32)
    telemetry = np.zeros((B, 64), dtype=np.float32)

    probs, values = policy.forward(screen, spatial, telemetry)
    assert probs.shape == (B, 7)
    assert values.shape == (B,)
    # Probabilities sum to 1.0
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(B), atol=1e-5)


def test_baseline_v2_torch_actor_critic():
    """Verifies PyTorch Actor-Critic forward and backward autograd passes."""
    if not b2.HAS_TORCH:
        pytest.skip("PyTorch not installed")

    import torch
    net = b2.PleinesActorCriticTorch(use_gru=False)
    screen = torch.randn(2, 3, 72, 80)
    spatial = torch.randn(2, 1, 48, 48)
    telemetry = torch.randn(2, 64)

    logits, values, _ = net(screen, spatial, telemetry)
    assert logits.shape == (2, 7)
    assert values.shape == (2,)

    # Backward pass
    loss = logits.sum() + values.sum()
    loss.backward()
    assert net.actor_head.weight.grad is not None
    assert net.critic_head.weight.grad is not None


def test_baseline_v2_gae_computation():
    """Verifies Generalized Advantage Estimation GAE(0.997, 0.95)."""
    T = 10
    rewards = np.ones(T, dtype=np.float32)
    values = np.zeros(T + 1, dtype=np.float32)
    dones = np.zeros(T, dtype=np.float32)

    advs, targets = b2.compute_gae(rewards, values, dones, gamma=0.997, lambda_gae=0.95)
    assert advs.shape == (T,)
    assert targets.shape == (T,)
    # Later advantages decay according to (gamma * lambda)
    assert advs[0] > advs[-1]


def test_baseline_v2_table3_reproduction_ablation():
    """Verifies Table 3 ablation runner for standard and ablated variants."""
    res_squirtle = b2.run_ablation_experiment("baseline_squirtle", num_steps=50)
    assert res_squirtle["steps_simulated"] == 50
    assert res_squirtle["sps"] > 100
    assert "paper_benchmark" in res_squirtle

    res_no_heal = b2.run_ablation_experiment("baseline_no_heal", num_steps=50)
    assert res_no_heal["reward_components"]["r_heal"] == 0.0
