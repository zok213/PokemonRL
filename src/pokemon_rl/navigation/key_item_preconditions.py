"""
key_item_preconditions.py — Generation 1 Key Item & HM Precondition Manager
=============================================================================
Enforces authentic Nintendo Game Boy LR35902 hardware constraints on environmental
barriers across the Kanto region:
  - HM01 Cut: Requires Cascade Badge (bit 1) + Move Cut (15) or HM01 (0xC4)
  - HM02 Fly: Requires Thunder Badge (bit 2) + Move Fly (19) or HM02 (0xC5)
  - HM03 Surf: Requires Soul Badge (bit 4) + Move Surf (57) or HM03 (0xC6)
  - HM04 Strength: Requires Rainbow Badge (bit 3) + Move Strength (70) or HM04 (0xC7)
  - HM05 Flash: Requires Boulder Badge (bit 0) + Move Flash (148) or HM05 (0xC8)
  - Silph Scope (0x48): Essential to identify Ghost Marowak in Pokémon Tower
  - Poké Flute (0x49): Essential to awaken sleeping Snorlax on Routes 12 & 16
  - Card Key (0x30): Unlocks electronic sliding doors in Silph Co. (1F–11F)
  - Secret Key (0x2B): Unlocks Cinnabar Island Gym entrance door
  - S.S. Ticket (0x3F): Authorizes boarding the S.S. Anne in Vermilion Harbor
  - Gold Teeth (0x40): Exchanged with Safari Warden for HM04 Strength
  - Lift Key (0x4A): Operates Team Rocket Hideout elevator on B1F/B2F/B4F
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Union
from pokemon_rl.env.wram_map import RAMMap


class KeyItemPreconditionChecker:
    """
    Evaluates authentic Game Boy WRAM inventory and badge bitfields to determine
    whether physical and topological preconditions are satisfied for overworld exploration.
    """

    # Move IDs in Generation 1 matching HMs
    MOVE_CUT      = 15   # Cut
    MOVE_FLY      = 19   # Fly
    MOVE_SURF     = 57   # Surf
    MOVE_STRENGTH = 70   # Strength
    MOVE_FLASH    = 148  # Flash

    @staticmethod
    def _extract_state(
        source: Union[Dict[str, Any], bytes]
    ) -> Tuple[int, List[int], List[int]]:
        """
        Extracts (badges, bag_item_ids, party_move_ids) from either
        a telemetry snapshot dict or raw WRAM bytes.
        """
        if isinstance(source, dict):
            badges = int(source.get("badges", 0))
            bag_raw = source.get("bag", [])
            bag_items = [item[0] if isinstance(item, (tuple, list)) else item for item in bag_raw]
            party_moves: List[int] = []
            if "party_moves" in source and isinstance(source["party_moves"], list):
                party_moves.extend(source["party_moves"])
            active_mon = source.get("active_mon", {})
            if isinstance(active_mon, dict):
                party_moves.extend(active_mon.get("moves", []))
            for mon in source.get("party", []):
                if isinstance(mon, dict):
                    party_moves.extend(mon.get("moves", []))
            return badges, bag_items, party_moves
        elif isinstance(source, (bytes, bytearray)):
            badges = int(source[RAMMap.OBTAINED_BADGES - 0xC000])
            bag_pairs = RAMMap.read_bag_items(source)
            bag_items = [pair[0] for pair in bag_pairs]
            party_moves = []
            return badges, bag_items, party_moves
        return 0, [], []

    @classmethod
    def can_cut_tree(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """
        Cut requires Cascade Badge (bit 1 of wObtainedBadges) AND
        either a Pokémon knowing Cut or HM01 in bag.
        """
        badges, bag, moves = cls._extract_state(source)
        has_cascade = bool(badges & RAMMap.BADGE_CASCADE)
        has_cut_move = cls.MOVE_CUT in moves
        has_hm01 = RAMMap.ITEM_HM01_CUT in bag
        return has_cascade and (has_cut_move or has_hm01)

    @classmethod
    def can_surf_water(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """
        Surf requires Soul Badge (bit 4 of wObtainedBadges) AND
        either a Pokémon knowing Surf or HM03 in bag.
        """
        badges, bag, moves = cls._extract_state(source)
        has_soul = bool(badges & RAMMap.BADGE_SOUL)
        has_surf_move = cls.MOVE_SURF in moves
        has_hm03 = RAMMap.ITEM_HM03_SURF in bag
        return has_soul and (has_surf_move or has_hm03)

    @classmethod
    def can_push_boulder(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """
        Strength requires Rainbow Badge (bit 3 of wObtainedBadges) AND
        either a Pokémon knowing Strength or HM04 in bag.
        """
        badges, bag, moves = cls._extract_state(source)
        has_rainbow = bool(badges & RAMMap.BADGE_RAINBOW)
        has_strength_move = cls.MOVE_STRENGTH in moves
        has_hm04 = RAMMap.ITEM_HM04_STRENGTH in bag
        return has_rainbow and (has_strength_move or has_hm04)

    @classmethod
    def can_flash_cave(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """
        Flash requires Boulder Badge (bit 0 of wObtainedBadges) AND
        either a Pokémon knowing Flash or HM05 in bag.
        """
        badges, bag, moves = cls._extract_state(source)
        has_boulder = bool(badges & RAMMap.BADGE_BOULDER)
        has_flash_move = cls.MOVE_FLASH in moves
        has_hm05 = RAMMap.ITEM_HM05_FLASH in bag
        return has_boulder and (has_flash_move or has_hm05)

    @classmethod
    def can_see_ghosts(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Identifies Ghost Pokémon in Pokémon Tower (bypasses unidentifiable ghost curse)."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_SILPH_SCOPE in bag

    @classmethod
    def can_wake_snorlax(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Awakens sleeping Snorlax blocking Route 12 and Route 16."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_POKE_FLUTE in bag

    @classmethod
    def can_unlock_silph_doors(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Unlocks electronic sliding security doors in Silph Co."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_CARD_KEY in bag

    @classmethod
    def can_unlock_cinnabar_gym(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Unlocks Cinnabar Island Gym entrance door."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_SECRET_KEY in bag

    @classmethod
    def can_board_ss_anne(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Authorizes boarding the S.S. Anne in Vermilion Harbor."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_S_S_TICKET in bag

    @classmethod
    def can_operate_hideout_elevator(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Operates the Rocket Hideout elevator."""
        _, bag, _ = cls._extract_state(source)
        return RAMMap.ITEM_LIFT_KEY in bag

    @classmethod
    def can_challenge_pokemon_league(cls, source: Union[Dict[str, Any], bytes]) -> bool:
        """Requires all 8 Kanto Gym Badges (0xFF = 255)."""
        badges, _, _ = cls._extract_state(source)
        return badges == 0xFF

    @classmethod
    def get_barrier_status(cls, source: Union[Dict[str, Any], bytes]) -> Dict[str, bool]:
        """
        Returns full boolean status of all environmental barrier preconditions.
        """
        return {
            "can_cut_tree": cls.can_cut_tree(source),
            "can_surf_water": cls.can_surf_water(source),
            "can_push_boulder": cls.can_push_boulder(source),
            "can_flash_cave": cls.can_flash_cave(source),
            "can_see_ghosts": cls.can_see_ghosts(source),
            "can_wake_snorlax": cls.can_wake_snorlax(source),
            "can_unlock_silph_doors": cls.can_unlock_silph_doors(source),
            "can_unlock_cinnabar_gym": cls.can_unlock_cinnabar_gym(source),
            "can_board_ss_anne": cls.can_board_ss_anne(source),
            "can_operate_hideout_elevator": cls.can_operate_hideout_elevator(source),
            "can_challenge_league": cls.can_challenge_pokemon_league(source),
        }
