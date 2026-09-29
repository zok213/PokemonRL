"""
test_action_masker.py — Unit Tests for Dynamic Action Masker
============================================================
Verifies hardware wJoyIgnore zero-leak masking, text-box restriction, and wall-bump latch.
"""

import pytest
import numpy as np
from pokemon_rl.env.action_masker import DynamicActionMasker
from pokemon_rl.env.wram_map import Action, RAMMap


def test_hardware_joy_ignore_mask():
    """
    Zero-Leak Hardware Mask:
    When wJoyIgnore has bit set for Action.A (1 << 4), Action.A MUST be masked False.
    """
    masker = DynamicActionMasker()

    def mock_reader(addr):
        if addr == RAMMap.JOY_IGNORE:
            return 1 << Action.A  # Suppress button A
        return 0

    mask = masker.compute_action_mask(mock_reader)
    assert mask[Action.A] is np.False_ or not mask[Action.A]
    assert mask[Action.B] is np.True_ or mask[Action.B]
    assert mask[Action.UP] is np.True_ or mask[Action.UP]


def test_text_box_dialogue_restriction():
    """During overworld dialogue (wTextBoxID != 0), directional movement and START must be suppressed."""
    masker = DynamicActionMasker()

    def mock_reader(addr):
        if addr == RAMMap.TEXT_BOX_ID:
            return 1  # Dialog active
        return 0

    mask = masker.compute_action_mask(mock_reader)
    assert not mask[Action.UP]
    assert not mask[Action.DOWN]
    assert not mask[Action.LEFT]
    assert not mask[Action.RIGHT]
    assert not mask[Action.START]
    # A and B must remain enabled to progress and clear dialogue
    assert mask[Action.A]
    assert mask[Action.B]


def test_wall_bump_stagnation_latch():
    """Attempting the same direction into a wall latch-suppresses that direction."""
    masker = DynamicActionMasker()
    pos = (5, 4)

    # Initial position registration followed by two wall-bump stagnation steps
    masker.update_spatial_telemetry(pos, Action.UP)
    masker.update_spatial_telemetry(pos, Action.UP)
    masker.update_spatial_telemetry(pos, Action.UP)

    def mock_reader(addr):
        return 0

    mask = masker.compute_action_mask(mock_reader)
    assert not mask[Action.UP]  # Suppressed due to consecutive stagnation
    assert mask[Action.DOWN]    # Other directions remain allowed
