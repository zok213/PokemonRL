"""
test_native_vectorizer.py — Tests for High-Performance Native Vectorization Engine
===================================================================================
Verifies zero-copy memory buffers, Numba LLVM parallel execution, and PyTorch tensor wrapping.
"""

import time
import pytest
import numpy as np
import torch
from pokemon_rl.env.native_vectorizer import NativeVectorEngine, HAS_NUMBA


def test_native_vector_engine_init():
    num_envs = 8
    engine = NativeVectorEngine(num_envs=num_envs)

    assert engine.num_envs == 8
    assert engine.screen_obs.shape == (8, 3, 72, 80)
    assert engine.wram_obs.shape == (8, 64)
    assert engine.action_masks.shape == (8, 8)
    assert engine.rewards.shape == (8,)
    assert engine.dones.shape == (8,)


def test_native_vector_engine_step():
    num_envs = 16
    engine = NativeVectorEngine(num_envs=num_envs)

    # Set mock wJoyIgnore in raw WRAM for env 0: suppress Action A (bit 0 = 0x01)
    engine.raw_wram_buffer[0, 0x0D6B] = 0x01

    actions = np.array([4] * num_envs, dtype=np.uint8) # All UP
    obs, rewards, dones, masks = engine.step(actions)

    assert obs["screen"].shape == (16, 3, 72, 80)
    assert obs["wram"].shape == (16, 64)
    assert rewards.shape == (16,)
    assert dones.shape == (16,)
    assert masks.shape == (16, 8)

    # Env 0 should have Action A masked out if Numba JIT decoded joy_ignore
    if HAS_NUMBA:
        assert masks[0, 0] == False # Action A disabled
        assert masks[0, 1] == True  # Action B enabled


def test_to_torch_tensors():
    num_envs = 4
    engine = NativeVectorEngine(num_envs=num_envs)
    actions = np.array([0, 1, 2, 3], dtype=np.uint8)
    engine.step(actions)

    tensors = engine.to_torch_tensors(device="cpu")

    assert isinstance(tensors["screen"], torch.Tensor)
    assert tensors["screen"].shape == (4, 3, 72, 80)
    assert tensors["screen"].dtype == torch.float32

    assert isinstance(tensors["wram"], torch.Tensor)
    assert tensors["wram"].shape == (4, 64)

    assert isinstance(tensors["action_mask"], torch.Tensor)
    assert tensors["action_mask"].shape == (4, 8)
    assert tensors["action_mask"].dtype == torch.bool


def test_native_throughput_benchmark():
    """Verify that native parallel vectorization reaches high SPS."""
    num_envs = 32
    engine = NativeVectorEngine(num_envs=num_envs)
    actions = np.zeros((num_envs,), dtype=np.uint8)

    # Warmup
    engine.step(actions)

    num_steps = 100
    start = time.perf_counter()
    for _ in range(num_steps):
        engine.step(actions)
    elapsed = time.perf_counter() - start

    total_actions = num_steps * num_envs
    sps = total_actions / elapsed if elapsed > 0 else 0.0

    print(f"\n[NativeVectorEngine Throughput]: {sps:,.0f} SPS ({num_envs} vectorized environments)")
    # Must exceed 2,000 SPS easily on native batch processing
    assert sps > 2000.0
