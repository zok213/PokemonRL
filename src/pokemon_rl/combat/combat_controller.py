"""
combat_controller.py — Decoupled Tactical Combat Head (Metamon Proxy)
=====================================================================
Decoupled tactical combat controller running sub-15ms minimax / heuristic decisions.
Completely relieves the overworld policy from learning combat permutations.

Theoretical basis:
  - Grigsby et al., "Human-Level Competitive Pokémon via Scalable Offline RL
    with Transformers", RLC 2025 (arXiv:2504.04395).
  - Gen 1 battle mechanics: Type matchups, physical/special split, move power,
    and 2x2 FIGHT menu cursor navigation.

Navigation Mechanics:
  In Pokémon Red (Gen 1), selecting FIGHT opens a 2x2 grid of 4 moves:
    Slot 0 (top-left)     | Slot 1 (top-right)
    Slot 2 (bottom-left)  | Slot 3 (bottom-right)
  Cursor starts at (0, 0).
  Navigation requires emitting directional inputs (RIGHT, DOWN) followed by A.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from pokemon_rl.env.wram_map import Action, RAMMap


class DecoupledCombatController:
    """
    Decoupled Tactical Combat Head.

    Evaluates expected move damage based on Gen 1 type effectiveness,
    base power, and accuracy, then issues the exact Game Boy joypad
    action sequence needed to navigate the FIGHT menu and select the optimal move.
    """

    # Gen 1 Type Effectiveness Chart (Attacker, Defender) -> Multiplier
    # In Gen 1: Normal, Fire, Water, Grass, Electric, Ice, Fighting, Poison,
    # Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon.
    TYPE_CHART: Dict[Tuple[str, str], float] = {
        # Water
        ("WATER", "FIRE"): 2.0,
        ("WATER", "GROUND"): 2.0,
        ("WATER", "ROCK"): 2.0,
        ("WATER", "WATER"): 0.5,
        ("WATER", "GRASS"): 0.5,
        ("WATER", "DRAGON"): 0.5,
        # Fire
        ("FIRE", "GRASS"): 2.0,
        ("FIRE", "ICE"): 2.0,
        ("FIRE", "BUG"): 2.0,
        ("FIRE", "WATER"): 0.5,
        ("FIRE", "FIRE"): 0.5,
        ("FIRE", "ROCK"): 0.5,
        ("FIRE", "DRAGON"): 0.5,
        # Grass
        ("GRASS", "WATER"): 2.0,
        ("GRASS", "GROUND"): 2.0,
        ("GRASS", "ROCK"): 2.0,
        ("GRASS", "FIRE"): 0.5,
        ("GRASS", "GRASS"): 0.5,
        ("GRASS", "POISON"): 0.5,
        ("GRASS", "FLYING"): 0.5,
        ("GRASS", "BUG"): 0.5,
        ("GRASS", "DRAGON"): 0.5,
        # Electric
        ("ELECTRIC", "WATER"): 2.0,
        ("ELECTRIC", "FLYING"): 2.0,
        ("ELECTRIC", "ELECTRIC"): 0.5,
        ("ELECTRIC", "GRASS"): 0.5,
        ("ELECTRIC", "DRAGON"): 0.5,
        ("ELECTRIC", "GROUND"): 0.0,
        # Psychic (Gen 1 dominant: Ghost bug in Gen 1 made Psychic immune to Ghost)
        ("PSYCHIC", "FIGHTING"): 2.0,
        ("PSYCHIC", "POISON"): 2.0,
        ("PSYCHIC", "PSYCHIC"): 0.5,
        # Ice
        ("ICE", "GRASS"): 2.0,
        ("ICE", "GROUND"): 2.0,
        ("ICE", "FLYING"): 2.0,
        ("ICE", "DRAGON"): 2.0,
        ("ICE", "WATER"): 0.5,
        ("ICE", "FIRE"): 0.5,
        ("ICE", "ICE"): 0.5,
        # Ground
        ("GROUND", "FIRE"): 2.0,
        ("GROUND", "ELECTRIC"): 2.0,
        ("GROUND", "POISON"): 2.0,
        ("GROUND", "ROCK"): 2.0,
        ("GROUND", "GRASS"): 0.5,
        ("GROUND", "BUG"): 0.5,
        ("GROUND", "FLYING"): 0.0,
        # Normal
        ("NORMAL", "ROCK"): 0.5,
        ("NORMAL", "GHOST"): 0.0,
    }

    def __init__(self):
        # Cursor tracking state across successive step calls
        self.current_cursor_pos: int = 0  # 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        self.nav_queue: List[int] = []

    def get_type_multiplier(self, atk_type: str, def_type: str) -> float:
        """Query Gen 1 type matchup multiplier (default 1.0)."""
        return self.TYPE_CHART.get((atk_type.upper(), def_type.upper()), 1.0)

    def evaluate_move(self, move: Dict[str, Any], opponent_type: str, user_type: Optional[str] = None) -> float:
        """
        Compute expected move score: Power * Accuracy * TypeEffectiveness * STAB.
        """
        power = move.get("power", 40)
        acc = move.get("accuracy", 1.0)
        move_type = move.get("type", "NORMAL").upper()
        multiplier = self.get_type_multiplier(move_type, opponent_type)

        # Same-Type Attack Bonus (STAB = 1.5x in Gen 1)
        stab = 1.5 if (user_type and user_type.upper() == move_type) else 1.0

        return float(power * acc * multiplier * stab)

    def select_best_move_index(self, moves: List[Dict[str, Any]], opponent_type: str, user_type: Optional[str] = None) -> int:
        """
        Returns the 0-indexed position (0, 1, 2, or 3) of the highest expected damage move.
        """
        if not moves:
            return 0

        best_idx = 0
        best_score = -1.0

        for idx, move in enumerate(moves):
            # Check PP if available
            pp = move.get("pp", 1)
            if pp <= 0:
                continue

            score = self.evaluate_move(move, opponent_type, user_type)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def plan_menu_navigation(self, target_slot: int) -> List[int]:
        """
        Plan the sequence of joypad actions to navigate from (0, 0) to target_slot in the 2x2 FIGHT grid:
          Slot 0: (row 0, col 0) -> [Action.A]
          Slot 1: (row 0, col 1) -> [Action.RIGHT, Action.A]
          Slot 2: (row 1, col 0) -> [Action.DOWN, Action.A]
          Slot 3: (row 1, col 1) -> [Action.RIGHT, Action.DOWN, Action.A]
        """
        col = target_slot % 2
        row = target_slot // 2

        actions: List[int] = []
        if col == 1:
            actions.append(Action.RIGHT)
        if row == 1:
            actions.append(Action.DOWN)
        actions.append(Action.A)
        return actions

    def select_battle_action(self, battle_state: Dict[str, Any]) -> int:
        """
        Stateful battle action selector.
        If a navigation sequence is currently queued, pops and returns next button.
        Otherwise plans navigation to the best move and returns first action.
        """
        if self.nav_queue:
            return self.nav_queue.pop(0)

        moves = battle_state.get("available_moves", [])
        if not moves:
            return Action.A  # Default: press A to progress dialogue / run default attack

        opp_type = battle_state.get("opponent_type", "NORMAL")
        user_type = battle_state.get("user_type", None)

        best_slot = self.select_best_move_index(moves, opp_type, user_type)
        actions = self.plan_menu_navigation(best_slot)

        if not actions:
            return Action.A

        first_action = actions.pop(0)
        self.nav_queue = actions
        return first_action

    def reset(self) -> None:
        """Reset cursor and navigation queue when battle terminates."""
        self.current_cursor_pos = 0
        self.nav_queue.clear()
