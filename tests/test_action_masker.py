"""
test_action_masker.py — Unit Tests for Dynamic Action Masker
============================================================
Verifies hardware wJoyIgnore zero-leak masking, text-box restriction, and wall-bump latch.
"""

import pytest
import numpy as np
from pokemon_rl.env.action_masker import DynamicActionMasker
from pokemon_rl.env.wram_map import Action, RAMMap, ACTION_TO_HW_BIT, HW_BIT_A, HW_BIT_UP


def test_hardware_joy_ignore_mask():
    """
    Zero-Leak Hardware Mask:
    When wJoyIgnore has bit set for A (bit 0 = 0x01 in hardware.inc B_PAD_A),
    Action.A MUST be masked False, while other actions remain enabled.
    """
    masker = DynamicActionMasker()

    def mock_reader(addr):
        if addr == RAMMap.JOY_IGNORE:
            return 1 << HW_BIT_A  # Bit 0: Hardware A Button suppression
        return 0

    mask = masker.compute_action_mask(mock_reader)
    assert mask[Action.A] is np.False_ or not mask[Action.A]
    assert mask[Action.B] is np.True_ or mask[Action.B]
    assert mask[Action.UP] is np.True_ or mask[Action.UP]

    # Test UP suppression (Bit 6: B_PAD_UP = 0x40 in hardware.inc)
    def mock_reader_up(addr):
        if addr == RAMMap.JOY_IGNORE:
            return 1 << HW_BIT_UP
        return 0

    mask_up = masker.compute_action_mask(mock_reader_up)
    assert not mask_up[Action.UP]
    assert mask_up[Action.A]
    assert mask_up[Action.DOWN]


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
