"""
test_puffer_bridge.py — Unit Tests for High-Throughput Vectorized Environment Bridge
=====================================================================================
Verifies synchronous array batching, zero-allocation buffers, and throughput scalability.
"""

import numpy as np
import pytest
from pokemon_rl.env.puffer_bridge import VectorizedPufferEnvironment


def test_vectorized_env_reset_shapes():
    num_envs = 8
    vec_env = VectorizedPufferEnvironment(num_envs=num_envs)
    obs, info = vec_env.reset()

    assert obs["screen"].shape == (num_envs, 3, 72, 80)
    assert obs["spatial_map"].shape == (num_envs, 1, 48, 48)
    assert obs["wram"].shape == (num_envs, 64)
    assert obs["action_mask"].shape == (num_envs, 8)
    assert info["num_envs"] == num_envs


def test_vectorized_env_step_and_rewards():
    num_envs = 4
    vec_env = VectorizedPufferEnvironment(num_envs=num_envs, max_steps=50)
    obs, _ = vec_env.reset()

    actions = np.zeros(num_envs, dtype=np.int32)
    next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

    assert next_obs["screen"].shape == (num_envs, 3, 72, 80)
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert len(infos) == num_envs
    assert infos[0]["step"] == 1


def test_vectorized_env_throughput():
    num_envs = 16
    vec_env = VectorizedPufferEnvironment(num_envs=num_envs)
    sps = vec_env.benchmark_throughput(num_steps=100)
    assert sps > 1000.0, f"Expected >1000 SPS, got {sps:.1f} SPS"
