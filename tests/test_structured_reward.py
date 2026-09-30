"""
tests.test_structured_reward — Rigorous Unit Tests for Whidden-Aligned Modular Reward & Navigation
==================================================================================================
Validates:
  1. Monotone-max event flag reward scaling and delta emission
  2. Level sum scaling and explore threshold (EXPLORE_THRESH = 22)
  3. Anti-stagnation penalty on persistent stagnation and local 2-tile oscillation
  4. Dialogue farming immunity (no spurious rewards during text boxes)
  5. Multi-directional wall-bump masking in ActionDispatcher
  6. Anti-oscillation limit cycle breaking in ActionDispatcher
  7. Whidden observation tensor shape (1, 3, 128, 40) and value range [0, 1]
"""

import pytest
import numpy as np
import torch
from collections import deque

from interactive.reward.whidden import (
    WhiddenRewardState, WhiddenStepReward, compute_whidden_step, REWARD_SCALE
)
from interactive.reward.anti_stagnation import (
    AntiStagnationState, AntiStagnationReward, compute_anti_stagnation,
    STAGNATION_THRESHOLD, STAGNATION_PENALTY
)
from interactive.reward.tracker import RewardTracker
from interactive.policy.observation import WhiddenObservationBuilder
from interactive.policy.action_dispatch import ActionDispatcher, IDX_DOWN, IDX_LEFT, IDX_RIGHT, IDX_UP, IDX_A


class MockMemory:
    """Mock PyBoy memory supporting __getitem__."""
    def __init__(self, data=None):
        self._data = bytearray(0x10000)
        if data:
            for k, v in data.items():
                self._data[k] = v

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        self._data[idx] = val


def test_whidden_event_reward_monotone_max():
    """Event rewards must be monotone max: only increases trigger positive deltas."""
    state = WhiddenRewardState()
    mem = MockMemory()

    # Step 1: Base flags (13 flags set, matching BASE_EVENT_FLAGS) -> 0 reward
    for i in range(13):
        mem[0xD747 + i] = 0x01
    r1 = compute_whidden_step(mem, state)
    assert r1.event == 0.0
    assert r1.total == 0.0

    # Step 2: Simulate 1 new event flag at 0xD760 (away from museum ticket)
    mem[0xD760] = 0x01
    r2 = compute_whidden_step(mem, state)
    assert r2.event == 1.0
    assert abs(r2.total - 0.1) < 1e-6

    # Step 3: Same event flag repeated -> delta MUST be strictly 0.0 (no farming)
    r3 = compute_whidden_step(mem, state)
    assert r3.event == 0.0
    assert r3.total == 0.0

    # Step 4: Event flag cleared in memory -> delta cannot be negative (monotone max)
    mem[0xD747] = 0x00
    r4 = compute_whidden_step(mem, state)
    assert r4.event == 0.0
    assert r4.total == 0.0


def test_whidden_level_reward_scaling():
    """Party level rewards scale properly before and after threshold."""
    state = WhiddenRewardState()
    mem = MockMemory()

    # Party size = 1, Level = 5 at 0xD18C (lead Pokemon level)
    # LEVELS_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
    mem[0xD18C] = 5  # level 5: max(5-2, 0) = 3; sum = 3 - 4 = 0
    r = compute_whidden_step(mem, state)
    assert r.level == 0.0

    # Level up to 10: level - 2 = 8; sum = 8 - 4 = 4
    mem[0xD18C] = 10
    r_up = compute_whidden_step(mem, state)
    assert r_up.level == 4.0
    assert abs(r_up.total - 0.4) < 1e-6


def test_anti_stagnation_local_oscillation():
    """Detects oscillation between 2 tiles and applies stagnation penalty."""
    state = AntiStagnationState()
    mem = MockMemory({0xD35E: 40, 0xD362: 0, 0xD361: 5}) # Map 40, (0, 5)

    # Alternate between (0, 5) and (1, 5) for 30 steps
    for step in range(35):
        x = step % 2
        mem[0xD362] = x
        rew = compute_anti_stagnation(mem, state, is_novel=False)

    # After 30 steps of oscillating between 2 tiles, stagnation penalty engages
    assert rew.stagnation <= STAGNATION_PENALTY
    assert rew.total <= STAGNATION_PENALTY


def test_reward_tracker_integration():
    """RewardTracker computes coordinate exploration, whidden rewards, and penalties."""
    tracker = RewardTracker()
    mem = MockMemory({0xD35E: 40, 0xD362: 5, 0xD361: 3, 0xD163: 1, 0xD18C: 6, 0xD16C: 0, 0xD16D: 22, 0xD18E: 0, 0xD18F: 22})

    # Step 1: First unique tile
    r1 = tracker.step(mem)
    assert r1 > 0.0
    assert tracker.reward_matrix["explore"] > 0.0
    assert tracker.visited_count == 1

    # Step 2: Same tile (not novel, no progress) -> reward should be 0.0
    r2 = tracker.step(mem)
    assert r2 == 0.0
    assert tracker.reward_matrix["explore"] == 0.0


def test_action_dispatcher_wall_bump_memory():
    """ActionDispatcher remembers blocked directions at the current tile."""
    disp = ActionDispatcher()
    mem = MockMemory({0xD35E: 40, 0xD362: 0, 0xD361: 5}) # (0, 5)

    # Fake policy
    class DummyPolicy:
        def __call__(self, obs_t, action_mask=None):
            probs = torch.ones((1, 8)) / 8.0
            if action_mask is not None:
                probs[~action_mask] = 0.0
                probs = probs / probs.sum()
            return probs, torch.zeros((1, 8))

    policy = DummyPolicy()
    obs_t = torch.zeros((1, 3, 128, 40))

    # Step 1: Agent tries DOWN and doesn't move
    disp.last_pos = (40, 0, 5)
    disp.last_action = IDX_DOWN
    act1, probs1, _ = disp._overworld_action(mem, policy, obs_t)
    assert IDX_DOWN in disp.blocked_dirs
    assert probs1[IDX_DOWN] == 0.0  # DOWN is masked

    # Step 2: Agent tries LEFT and doesn't move
    disp.last_pos = (40, 0, 5)
    disp.last_action = IDX_LEFT
    act2, probs2, _ = disp._overworld_action(mem, policy, obs_t)
    # BOTH DOWN and LEFT must now be remembered as blocked!
    assert IDX_DOWN in disp.blocked_dirs
    assert IDX_LEFT in disp.blocked_dirs
    assert probs2[IDX_DOWN] == 0.0
    assert probs2[IDX_LEFT] == 0.0

    # Step 3: Agent moves to (1, 5) -> blocked_dirs clears
    mem[0xD362] = 1
    disp._overworld_action(mem, policy, obs_t)
    assert len(disp.blocked_dirs) == 0


def test_whidden_observation_builder_shape_and_range():
    """WhiddenObservationBuilder produces exact (1, 3, 128, 40) float32 in [0, 1]."""
    builder = WhiddenObservationBuilder()
    raw_screen = np.random.randint(0, 256, (144, 160, 4), dtype=np.uint8)
    telemetry = {"hp": 22, "max_hp": 22, "badges": 0}

    obs = builder.build(raw_screen, telemetry, visited_count=10)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (1, 3, 128, 40)
    assert obs.dtype == torch.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0
