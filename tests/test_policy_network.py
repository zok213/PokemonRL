"""
test_policy_network.py — Unit Tests for Multimodal Policy Network
=================================================================
Verifies forward pass tensor shapes, softmax normalization, action masking,
and non-zero STAD policy entropy computation.
"""

import pytest
import numpy as np
from pokemon_rl.agent.policy_network import (
    MultiModalPolicyNetwork,
    extract_wram_vector,
)


def test_network_shapes_and_probabilities():
    net = MultiModalPolicyNetwork(seed=42)
    B = 4
    rng = np.random.default_rng(42)

    frames = rng.random((B, 4, 72, 80)).astype(np.float32)
    map_obs = (rng.random((B, 1, 48, 48)) > 0.5).astype(np.float32)
    wram = rng.random((B, 64)).astype(np.float32)

    probs, logits = net.forward(frames, map_obs, wram)

    assert probs.shape == (B, 8)
    assert logits.shape == (B, 8)
    # Probabilities must sum to 1.0 along action dimension
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0)


def test_network_action_masking():
    net = MultiModalPolicyNetwork(seed=42)
    B = 2
    rng = np.random.default_rng(42)

    frames = rng.random((B, 4, 72, 80)).astype(np.float32)
    map_obs = (rng.random((B, 1, 48, 48)) > 0.5).astype(np.float32)
    wram = rng.random((B, 64)).astype(np.float32)

    # Mask action 4 (Action.A)
    mask = np.ones((B, 8), dtype=bool)
    mask[:, 4] = False

    probs, _ = net.forward(frames, map_obs, wram, action_mask=mask)
    assert np.allclose(probs[:, 4], 0.0, atol=1e-7)


def test_stad_policy_entropy_strictly_positive():
    """STAD policy entropy must be strictly positive for non-degenerate policies."""
    net = MultiModalPolicyNetwork(seed=42)
    B = 4
    rng = np.random.default_rng(42)

    frames = rng.random((B, 4, 72, 80)).astype(np.float32)
    map_obs = (rng.random((B, 1, 48, 48)) > 0.5).astype(np.float32)
    wram = rng.random((B, 64)).astype(np.float32)

    actions, log_probs, entropy = net.sample_action(frames, map_obs, wram)

    assert actions.shape == (B,)
    assert log_probs.shape == (B,)
    assert entropy.shape == (B,)
    assert np.all(entropy > 0.0)


def test_wram_telemetry_vector_extraction():
    wram_bytes = bytearray(8192)
    wram_bytes[0xD35E - 0xC000] = 12  # Cerulean City Map ID
    wram_bytes[0xD356 - 0xC000] = 0x03  # 2 badges

    vec = extract_wram_vector(bytes(wram_bytes))
    assert vec.shape == (64,)
    assert abs(vec[0] - 12.0 / 255.0) < 1e-5
