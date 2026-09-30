"""
wram_map.py — LR35902 Game Boy WRAM Register Map
=================================================
Canonical WRAM addresses sourced from pret/pokered disassembly:
  https://github.com/pret/pokered

All addresses are for the Game Boy Color/DMG memory space:
  0xC000–0xDFFF = 8KB Work RAM (WRAM0 + WRAM1)

This module is the SINGLE SOURCE OF TRUTH for all WRAM constants.
Import from here in all other modules — never hardcode addresses inline.
"""

from __future__ import annotations
from enum import IntEnum


class Action(IntEnum):
    """
    Game Boy joypad button enumeration.
    Matches pret/pokered button bit ordering.
    """
    UP     = 0
    DOWN   = 1
    LEFT   = 2
    RIGHT  = 3
    A      = 4
    B      = 5
    START  = 6
    SELECT = 7
    NUM_ACTIONS = 8


class RAMMap:
    """
    LR35902 Work RAM address constants for Pokémon Red.
    All addresses in range 0xC000–0xDFFF (WRAM).

    Usage:
        from pokemon_rl.env.wram_map import RAMMap
        map_id = pyboy.memory[RAMMap.CUR_MAP]
    """

    # =========================================================================
    # SPATIAL NAVIGATION
    # =========================================================================
    CUR_MAP     = 0xD35E   # wCurMap       — Current Map ID (0–247)
    X_POS       = 0xD362   # wXCoord       — Player X tile coordinate
    Y_POS       = 0xD361   # wYCoord       — Player Y tile coordinate

    # =========================================================================
    # PROGRESSION
    # =========================================================================
    OBTAINED_BADGES = 0xD356  # wObtainedBadges — 8-bit badge bitfield
                              #   bit 0 = Boulder (Brock)
                              #   bit 1 = Cascade (Misty)
                              #   bit 2 = Thunder  (Lt. Surge)
                              #   bit 3 = Rainbow  (Erika)
                              #   bit 4 = Soul     (Koga)
                              #   bit 5 = Marsh    (Sabrina)
                              #   bit 6 = Volcano  (Blaine)
                              #   bit 7 = Earth    (Giovanni)

    # Badge bitmasks (use with OBTAINED_BADGES)
    BADGE_BOULDER = 0x01
    BADGE_CASCADE = 0x02
    BADGE_THUNDER = 0x04
    BADGE_RAINBOW = 0x08
    BADGE_SOUL    = 0x10
    BADGE_MARSH   = 0x20
    BADGE_VOLCANO = 0x40
    BADGE_EARTH   = 0x80

    # =========================================================================
    # GAME STATE FLAGS
    # =========================================================================
    IS_IN_BATTLE  = 0xD057  # wIsInBattle    — 0=overworld, >0=battle active
    TEXT_BOX_ID   = 0xCF13  # wTextBoxID     — dialog/text window active
    MENU_ACTIVE   = 0xCC24  # wMenuType      — 0=no menu, >0=menu open
    CUR_MENU_ITEM = 0xCC26  # wCurrentMenuItem — cursor in menu/battle

    # =========================================================================
    # HARDWARE INPUT SUPPRESSION
    # =========================================================================
    JOY_IGNORE    = 0xCD6B  # wJoyIgnore     — HARDWARE INPUT MASK
                            # The Game Boy CPU writes a bitmask of buttons to
                            # silently discard during cutscenes/evolutions.
                            # bit 0=Up, 1=Down, 2=Left, 3=Right, 4=A, 5=B,
                            # 6=Start, 7=Select (matching Action enum).
                            # Read this BEFORE computing action masks for
                            # zero-leak suppression without heuristics.
                            # Reads as 0x00 during free overworld movement.

    # =========================================================================
    # SAFARI ZONE
    # =========================================================================
    SAFARI_STEPS_LO = 0xD70D  # wSafariSteps  — low byte  (canonical address)
    SAFARI_STEPS_HI = 0xD70E  # wSafariSteps  — high byte (16-bit countdown)
                              # NOTE: 0xDA38 (Rubinstein cheat) is WRONG.

    # =========================================================================
    # EVENT FLAGS (Quest Progression)
    # =========================================================================
    EVENT_FLAGS_BASE = 0xD747  # wEventFlags start (320 bytes = 2,560 bits)
    EVENT_FLAGS_END  = 0xD886  # wEventFlags end   (inclusive)

    # Key event flag bit addresses within wEventFlags:
    # Format: (byte_offset_from_0xD747, bit_index_0_7)
    EVENT_GOT_OAKS_PARCEL   = (0x0D, 0)  # 0xD754 bit 0 — picked up Oak's Parcel
    EVENT_DELIVERED_PARCEL  = (0x0D, 1)  # 0xD754 bit 1 — delivered to Oak
    EVENT_GOT_POKEDEX       = (0x0D, 2)  # 0xD754 bit 2 — received Pokédex
    EVENT_BEAT_BROCK        = (0x6A, 1)  # 0xD7B1 bit 1 — Brock defeated
    EVENT_GOT_BOULDER_BADGE = (0x6A, 2)  # 0xD7B1 bit 2 — Boulder Badge received
    EVENT_BEAT_MISTY        = (0xEE, 0)  # 0xD835 bit 0 — Misty defeated
    EVENT_GOT_CASCADE_BADGE = (0xEE, 1)  # 0xD835 bit 1 — Cascade Badge received
    EVENT_BEAT_LT_SURGE     = (0x7E, 0)  # 0xD7C5 bit 0 — Lt. Surge defeated
    EVENT_GOT_THUNDER_BADGE = (0x7E, 1)  # 0xD7C5 bit 1 — Thunder Badge received
    EVENT_BEAT_ERIKA        = (0x8A, 0)  # 0xD7D1 bit 0 — Erika defeated
    EVENT_GOT_RAINBOW_BADGE = (0x8A, 1)  # 0xD7D1 bit 1 — Rainbow Badge received
    EVENT_BEAT_KOGA         = (0xA2, 0)  # 0xD7E9 bit 0 — Koga defeated
    EVENT_GOT_SOUL_BADGE    = (0xA2, 1)  # 0xD7E9 bit 1 — Soul Badge received
    EVENT_BEAT_SABRINA      = (0xD4, 0)  # 0xD81B bit 0 — Sabrina defeated
    EVENT_GOT_MARSH_BADGE   = (0xD4, 1)  # 0xD81B bit 1 — Marsh Badge received
    EVENT_BEAT_BLAINE       = (0xCB, 0)  # 0xD812 bit 0 — Blaine defeated
    EVENT_GOT_VOLCANO_BADGE = (0xCB, 1)  # 0xD812 bit 1 — Volcano Badge received
    EVENT_BEAT_GIOVANNI     = (0x25, 0)  # 0xD76C bit 0 — Giovanni defeated
    EVENT_GOT_EARTH_BADGE   = (0x25, 1)  # 0xD76C bit 1 — Earth Badge received
    EVENT_ENTERED_HOF       = (0x10, 0)  # 0xD757 bit 0 — Hall of Fame entered

    # =========================================================================
    # PARTY / BATTLE STATE
    # =========================================================================
    PARTY_COUNT      = 0xD163  # wPartyCount — number of Pokémon in party (0–6)
    PARTY_HP_BASE    = 0xD16C  # wPartyMon1HP — first party member current HP (2 bytes)
    PARTY_MAX_HP_BASE= 0xD18D  # wPartyMon1MaxHP (2 bytes)
    ENEMY_SPECIES    = 0xCFCA  # wEnemyMon1Species — opponent Pokémon species ID

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    @classmethod
    def read_badge_count(cls, wram_bytes: bytes) -> int:
        """Count number of badges from wObtainedBadges byte."""
        badge_byte = wram_bytes[cls.OBTAINED_BADGES - 0xC000]
        return bin(badge_byte).count('1')

    @classmethod
    def read_event_flag(cls, wram_bytes: bytes, byte_off: int, bit_idx: int) -> bool:
        """Read a single event flag bit from wEventFlags."""
        addr_off = (cls.EVENT_FLAGS_BASE - 0xC000) + byte_off
        return bool(wram_bytes[addr_off] & (1 << bit_idx))

    @classmethod
    def read_safari_steps(cls, wram_bytes: bytes) -> int:
        """Read 16-bit Safari Zone step counter."""
        lo = wram_bytes[cls.SAFARI_STEPS_LO - 0xC000]
        hi = wram_bytes[cls.SAFARI_STEPS_HI - 0xC000]
        return lo + (hi << 8)

    @classmethod
    def read_joy_ignore(cls, wram_bytes: bytes) -> int:
        """Read hardware joypad suppression mask."""
        return wram_bytes[cls.JOY_IGNORE - 0xC000]
