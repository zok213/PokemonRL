"""
test_wram_map.py — Unit Tests for LR35902 WRAM Register Map
===========================================================
Verifies canonical memory addresses against pret/pokered hardware disassembly.
"""

import pytest
from pokemon_rl.env.wram_map import Action, RAMMap


def test_action_enum():
    """Verify joypad button ordering and count."""
    assert Action.UP == 0
    assert Action.DOWN == 1
    assert Action.LEFT == 2
    assert Action.RIGHT == 3
    assert Action.A == 4
    assert Action.B == 5
    assert Action.START == 6
    assert Action.SELECT == 7
    assert Action.NUM_ACTIONS == 8


def test_canonical_wram_addresses():
    """Verify critical WRAM addresses against pret/pokered disassembly."""
    # Spatial
    assert RAMMap.CUR_MAP == 0xD35E
    assert RAMMap.X_POS == 0xD362
    assert RAMMap.Y_POS == 0xD361

    # Progression
    assert RAMMap.OBTAINED_BADGES == 0xD356

    # Battle & text
    assert RAMMap.IS_IN_BATTLE == 0xD057
    assert RAMMap.TEXT_BOX_ID == 0xCF13

    # Hardware input suppression mask (discovered in pret/pokered)
    assert RAMMap.JOY_IGNORE == 0xCD6B

    # Canonical Safari Zone 16-bit countdown (0xD70D-0xD70E, NOT 0xDA38)
    assert RAMMap.SAFARI_STEPS_LO == 0xD70D
    assert RAMMap.SAFARI_STEPS_HI == 0xD70E

    # Event flags base
    assert RAMMap.EVENT_FLAGS_BASE == 0xD747


def test_badge_reading():
    """Verify bitfield parsing for obtained badges."""
    wram = bytearray(8192)
    # Set Boulder (bit 0) and Cascade (bit 1)
    badge_offset = RAMMap.OBTAINED_BADGES - 0xC000
    wram[badge_offset] = RAMMap.BADGE_BOULDER | RAMMap.BADGE_CASCADE

    assert RAMMap.read_badge_count(bytes(wram)) == 2


def test_safari_step_reading():
    """Verify 16-bit little-endian reading of Safari steps."""
    wram = bytearray(8192)
    # 500 = 0x01F4 -> lo = 0xF4 (244), hi = 0x01 (1)
    lo_offset = RAMMap.SAFARI_STEPS_LO - 0xC000
    hi_offset = RAMMap.SAFARI_STEPS_HI - 0xC000
    wram[lo_offset] = 0xF4
    wram[hi_offset] = 0x01

    assert RAMMap.read_safari_steps(bytes(wram)) == 500
