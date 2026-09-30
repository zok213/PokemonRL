"""
interactive/wram/reader.py — Low-Level WRAM Memory Read Helpers
================================================================
All functions take mem — any object supporting mem[int] -> int
(i.e., pyboy.memory or a plain bytes/bytearray for testing).

These are pure functions with zero side effects — ideal for unit testing.

IMPORTANT NOTES on Gen 1 WRAM:
  - wMenuType (0xCC24): NOT a reliable indicator of an open menu.
    In Gen 1, this field holds the current 'script context type' and is
    often set to 2 during normal overworld movement. DO NOT use alone.
  - wTextBoxID (0xCF13): Reliable. Non-zero ONLY when a text box/dialogue
    is actually being displayed on screen. Zero during free movement.
  - wJoyIgnore (0xCD6B): Reliable for text/cutscene. Bits 4-7 (0xF0)
    are set when the ROM engine suppresses directional input during
    scrolling text, cutscenes, or NPC approach animation.
    CAUTION: briefly non-zero during tile-walk transition animation (~4 frames).
"""

from __future__ import annotations
from interactive.wram.addresses import *


# ---------------------------------------------------------------------------
# Core Primitives
# ---------------------------------------------------------------------------

def read_u8(mem, addr: int) -> int:
    """Read unsigned 8-bit value."""
    return mem[addr] & 0xFF


def read_u16_be(mem, addr: int) -> int:
    """Read unsigned 16-bit big-endian value (e.g. HP)."""
    return (mem[addr] << 8) | mem[addr + 1]


def bit_count(n: int) -> int:
    """Count number of 1-bits (popcount)."""
    return bin(n).count("1")


def read_bit(mem, addr: int, bit: int) -> bool:
    """Read a single bit (0-7) from a WRAM byte."""
    return bool(mem[addr] & (1 << bit))


# ---------------------------------------------------------------------------
# Spatial Navigation
# ---------------------------------------------------------------------------

def read_map_id(mem) -> int:
    return mem[MAP_N_ADDR]


def read_xy(mem) -> tuple:
    return mem[X_POS_ADDR], mem[Y_POS_ADDR]


# ---------------------------------------------------------------------------
# Battle / Menu State
# ---------------------------------------------------------------------------

def is_in_battle(mem) -> bool:
    return mem[IS_IN_BATTLE_ADDR] != 0


def read_text_box_id(mem) -> int:
    return mem[TEXT_BOX_ID_ADDR]


def read_menu_type(mem) -> int:
    """
    wMenuType (0xCC24). WARNING: In Gen 1, this is often 2 during
    normal overworld play. Do NOT use this alone to detect menus.
    """
    return mem[MENU_TYPE_ADDR]


def read_joy_ignore(mem) -> int:
    return mem[JOY_IGNORE_ADDR]


def is_text_box_active(mem) -> bool:
    """
    True when a real text box is being displayed.
    wTextBoxID (0xCF13) is the MOST reliable dialogue indicator in Gen 1:
      0 = no text box
      1-255 = a specific text window type is open
    """
    return mem[TEXT_BOX_ID_ADDR] != 0


def is_directions_suppressed(mem) -> bool:
    """
    True when ROM engine is suppressing directional d-pad input.
    Bits 4-7 of wJoyIgnore (0xCD6B) are set during:
      - Active text scrolling / NPC dialogue
      - Cutscenes / overworld scripted events
      - Walk-off-ledge animation (~4 frames transient)
    """
    return (mem[JOY_IGNORE_ADDR] & 0xF0) != 0


def is_menu_or_text_active(mem) -> bool:
    """
    CORRECTED menu/dialogue detection for Gen 1 Pokémon Red.

    Reliable indicators (confirmed by WRAM trace):
      1. wTextBoxID (0xCF13) != 0  — a text window IS open on screen.
         Zero during all free overworld movement. Zero false positives.
      2. wJoyIgnore (0xCD6B) & 0xF0 — directional inputs suppressed.
         Transient (~1 frame) during map tile transitions.
         Only trust if text_box ALSO fires, or use text_box alone.

    wMenuType (0xCC24): PERMANENTLY 2 during normal overworld play in
    Gen 1 Pokémon Red. Completely unreliable as a menu indicator. IGNORED.

    Strategy: Use ONLY wTextBoxID as the primary indicator.
    joypad_ignore is supplementary (helps catch rapid NPC approach cutscenes
    where text_box hasn't flipped yet), but ONLY when non-zero text_box
    suggests we're mid-dialogue.
    """
    if is_in_battle(mem):
        return False   # battle engine manages its own flow
    # Primary: wTextBoxID is the gold standard — no false positives
    return mem[TEXT_BOX_ID_ADDR] != 0


# ---------------------------------------------------------------------------
# Party / Health
# ---------------------------------------------------------------------------

def read_party_size(mem) -> int:
    return mem[PARTY_SIZE_ADDR]


def read_party_level(mem, slot: int = 0) -> int:
    """Read the level of party slot 0-5."""
    return mem[LEVELS_ADDRS[slot]] if slot < 6 else 5


def read_hp(mem, slot: int = 0) -> int:
    """Read current HP (16-bit big-endian) of party slot 0-5."""
    return read_u16_be(mem, HP_ADDRS[slot])


def read_max_hp(mem, slot: int = 0) -> int:
    """Read max HP (16-bit big-endian) of party slot 0-5."""
    return read_u16_be(mem, MAX_HP_ADDRS[slot])


def read_hp_fraction(mem) -> float:
    """Total HP fraction across all party members."""
    hp_sum     = sum(read_u16_be(mem, a) for a in HP_ADDRS)
    max_hp_sum = sum(read_u16_be(mem, a) for a in MAX_HP_ADDRS)
    return hp_sum / max(max_hp_sum, 1)


def read_levels_sum(mem) -> int:
    """
    Sum of all party levels minus startup offset.
    Matches Whidden baseline get_levels_sum():
      levels = [max(level - 2, 0) for level in party_levels]
      return max(sum(levels) - 4, 0)
    """
    levels = [max(mem[a] - 2, 0) for a in LEVELS_ADDRS]
    return max(sum(levels) - 4, 0)


def read_moves(mem) -> tuple:
    """Return (move_ids[4], move_pps[4]) for lead party member."""
    if mem[PARTY_SIZE_ADDR] == 0:
        return [33, 39, 0, 0], [35, 30, 0, 0]
    ids = [mem[MOVE_IDS_BASE + i] for i in range(4)]
    pps = [mem[MOVE_PPS_BASE + i] for i in range(4)]
    return ids, pps


# ---------------------------------------------------------------------------
# Badges & Events
# ---------------------------------------------------------------------------

def read_badges(mem) -> int:
    """Number of badges obtained (0-8)."""
    return bit_count(mem[BADGE_ADDR])


def read_all_events_reward(mem) -> int:
    """
    Bit-count all event flag bytes 0xD747..0xD886, subtract base flags
    and museum ticket. Exactly replicates Whidden's get_all_events_reward().
    """
    museum_bit = bool(mem[MUSEUM_TICKET_ADDR] & (1 << MUSEUM_TICKET_BIT))
    total = sum(bit_count(mem[i]) for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END))
    return max(total - BASE_EVENT_FLAGS - int(museum_bit), 0)


def read_max_opp_level(mem) -> int:
    """Max level among all opponent party slots."""
    return max(mem[a] for a in OPP_LVL_ADDRS)
