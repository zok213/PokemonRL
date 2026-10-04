"""
macro_action_compiler.py — Game Boy Hardware Joypad Macro Compiler
===================================================================
Translates high-level affordance selections (Semi-Markov Options) into exact,
hardware-accurate sequences of joypad button presses.

In Game Boy Gen 1, menus remember their cursor positions across frames:
  - Battle root menu (FIGHT / PKMN / ITEM / RUN) restores `wBattleAndStartSavedMenuItem`
  - Move selection list starts at `wPlayerMoveListIndex`
  - Party list restores `wPartyAndBillsPCSavedMenuItem`

Therefore, button presses MUST be computed as dynamic deltas from current RAM cursor
coordinates, never static paths from an assumed top-left corner.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any


ROOT_CELLS: Dict[str, Tuple[int, int]] = {
    "fight": (0, 0),
    "item":  (0, 1),
    "pkmn":  (1, 0),
    "run":   (1, 1),
}


class MacroActionCompiler:
    """
    Compiles high-level affordances into verified Game Boy joypad button sequences.
    """

    @staticmethod
    def walk_grid(current: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
        """Generate direction presses to navigate a 2D menu grid."""
        (c0, r0), (c1, r1) = current, target
        across = ["right"] * (c1 - c0) if c1 > c0 else ["left"] * (c0 - c1)
        down = ["down"] * (r1 - r0) if r1 > r0 else ["up"] * (r0 - r1)
        return across + down

    @staticmethod
    def walk_list(current_idx: int, target_idx: int) -> List[str]:
        """Generate vertical presses to navigate a 1D vertical list menu."""
        if target_idx > current_idx:
            return ["down"] * (target_idx - current_idx)
        return ["up"] * (current_idx - target_idx)

    @classmethod
    def root_path(cls, saved_battle_menu_item: int, entry: str) -> List[str]:
        """Navigate from current battle root cursor to FIGHT, ITEM, PKMN, or RUN."""
        saved = saved_battle_menu_item & 0b11
        current_cell = (saved // 2, saved % 2)
        target_cell = ROOT_CELLS.get(entry, (0, 0))
        return cls.walk_grid(current_cell, target_cell)

    @classmethod
    def compile_option(
        cls,
        option_id: str,
        wram_cursor_state: Dict[str, Any],
        party_species: Optional[List[str]] = None,
        move_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Compile an affordance ID into button presses.

        Args:
            option_id: e.g. 'use_move_scratch', 'switch_to_squirtle', 'use_potion', 'step_up', 'run_away'
            wram_cursor_state: Dict containing cursor positions:
                - 'battle_menu_item': int
                - 'move_list_index': int
                - 'party_menu_item': int
                - 'menu_item': int

        Returns:
            List of button names ('up', 'down', 'left', 'right', 'a', 'b', 'start')
        """
        battle_item = wram_cursor_state.get("battle_menu_item", 0)
        move_idx = wram_cursor_state.get("move_list_index", 0)
        party_item = wram_cursor_state.get("party_menu_item", 0)

        # Overworld step
        if option_id.startswith("step_"):
            return [option_id[len("step_"):]]

        # In-battle: Run
        if option_id == "run_away":
            return cls.root_path(battle_item, "run") + ["a"]

        # In-battle: Item (Potion)
        if option_id in ("use_item_potion", "use_potion"):
            return cls.root_path(battle_item, "item") + ["a", "a", "a"]

        # In-battle: Move attack
        if option_id.startswith("use_move_"):
            move_slug = option_id[len("use_move_"):]
            target_slot = 0
            if move_names:
                for i, m in enumerate(move_names):
                    if m.lower().replace(" ", "_") == move_slug:
                        target_slot = i
                        break
            return (
                cls.root_path(battle_item, "fight")
                + ["a"]
                + cls.walk_list(move_idx, target_slot)
                + ["a"]
            )

        # In-battle: Switch Pokemon
        if option_id.startswith("switch_to_"):
            species_slug = option_id[len("switch_to_"):]
            target_party_slot = 1  # 0-indexed bench
            if party_species:
                for i, s in enumerate(party_species):
                    if s.lower().replace(" ", "_") == species_slug:
                        target_party_slot = i
                        break
            return (
                cls.root_path(battle_item, "pkmn")
                + ["a"]
                + cls.walk_list(party_item, target_party_slot)
                + ["a", "a"]
            )

        # Dialogue answer
        if option_id.startswith("answer_"):
            return ["a"]

        # Default fallback: advance dialog / acknowledge
        return ["a"]
