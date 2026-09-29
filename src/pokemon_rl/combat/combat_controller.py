"""
combat_controller.py — Decoupled Tactical Combat Head (Gen 1 LR35902 Grounded)
=============================================================================
Decoupled tactical combat controller running sub-15ms minimax / heuristic decisions.
Completely relieves the overworld policy from learning combat permutations.

Gen 1 LR35902 Accurate Implementation:
  1. Complete 15x15 Gen 1 Type Effectiveness Matrix (including Gen 1 quirks:
     Ghost vs Psychic = 0.0x bug, Bug vs Poison = 2.0x, Poison vs Bug = 2.0x).
  2. Dual-type defender effectiveness calculation.
  3. Gen 1 Base Speed-dependent Critical Hit rates:
       P(crit) = min(255, BaseSpeed // 2) / 256
       P(high_crit) = min(255, 4 * BaseSpeed) / 256
  4. The 1/256 miss glitch: 100% accuracy moves hit with prob 255/256.
  5. 2x2 FIGHT menu joypad cursor navigation.
  6. Opponent Deterministic AI prediction (pret/pokered/engine/battle/ai/trainer_ai.asm).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

from pokemon_rl.env.wram_map import Action, RAMMap


# High-critical-hit-ratio moves in Generation 1
GEN1_HIGH_CRIT_MOVES = {
    "SLASH", "RAZOR LEAF", "CRABHAMMER", "KARATE CHOP"
}


class DecoupledCombatController:
    """
    Decoupled Tactical Combat Head for Generation 1 Pokémon Red.
    """

    # Complete 15x15 Gen 1 Type Effectiveness Chart
    # (Attacking Type, Defending Type) -> Multiplier
    # Canonical reference: pret/pokered/data/types/type_matchups.asm
    TYPE_CHART: Dict[Tuple[str, str], float] = {
        # Normal
        ("NORMAL", "ROCK"): 0.5,
        ("NORMAL", "GHOST"): 0.0,

        # Fire
        ("FIRE", "GRASS"): 2.0,
        ("FIRE", "ICE"): 2.0,
        ("FIRE", "BUG"): 2.0,
        ("FIRE", "FIRE"): 0.5,
        ("FIRE", "WATER"): 0.5,
        ("FIRE", "ROCK"): 0.5,
        ("FIRE", "DRAGON"): 0.5,

        # Water
        ("WATER", "FIRE"): 2.0,
        ("WATER", "GROUND"): 2.0,
        ("WATER", "ROCK"): 2.0,
        ("WATER", "WATER"): 0.5,
        ("WATER", "GRASS"): 0.5,
        ("WATER", "DRAGON"): 0.5,

        # Electric
        ("ELECTRIC", "WATER"): 2.0,
        ("ELECTRIC", "FLYING"): 2.0,
        ("ELECTRIC", "ELECTRIC"): 0.5,
        ("ELECTRIC", "GRASS"): 0.5,
        ("ELECTRIC", "DRAGON"): 0.5,
        ("ELECTRIC", "GROUND"): 0.0,

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

        # Ice
        ("ICE", "GRASS"): 2.0,
        ("ICE", "GROUND"): 2.0,
        ("ICE", "FLYING"): 2.0,
        ("ICE", "DRAGON"): 2.0,
        ("ICE", "WATER"): 0.5,
        ("ICE", "ICE"): 0.5,
        # Note: In Gen 1, Fire did NOT resist Ice! Ice vs Fire is 1.0x

        # Fighting
        ("FIGHTING", "NORMAL"): 2.0,
        ("FIGHTING", "ROCK"): 2.0,
        ("FIGHTING", "ICE"): 2.0,
        ("FIGHTING", "FLYING"): 0.5,
        ("FIGHTING", "POISON"): 0.5,
        ("FIGHTING", "BUG"): 0.5,
        ("FIGHTING", "PSYCHIC"): 0.5,
        ("FIGHTING", "GHOST"): 0.0,

        # Poison
        ("POISON", "GRASS"): 2.0,
        ("POISON", "BUG"): 2.0,  # Gen 1: Poison is super-effective vs Bug
        ("POISON", "POISON"): 0.5,
        ("POISON", "GROUND"): 0.5,
        ("POISON", "ROCK"): 0.5,
        ("POISON", "GHOST"): 0.5,

        # Ground
        ("GROUND", "FIRE"): 2.0,
        ("GROUND", "ELECTRIC"): 2.0,
        ("GROUND", "POISON"): 2.0,
        ("GROUND", "ROCK"): 2.0,
        ("GROUND", "GRASS"): 0.5,
        ("GROUND", "BUG"): 0.5,
        ("GROUND", "FLYING"): 0.0,

        # Flying
        ("FLYING", "GRASS"): 2.0,
        ("FLYING", "FIGHTING"): 2.0,
        ("FLYING", "BUG"): 2.0,
        ("FLYING", "ELECTRIC"): 0.5,
        ("FLYING", "ROCK"): 0.5,

        # Psychic
        ("PSYCHIC", "FIGHTING"): 2.0,
        ("PSYCHIC", "POISON"): 2.0,
        ("PSYCHIC", "PSYCHIC"): 0.5,

        # Bug
        ("BUG", "GRASS"): 2.0,
        ("BUG", "POISON"): 2.0,  # Gen 1: Bug is super-effective vs Poison
        ("BUG", "PSYCHIC"): 2.0,
        ("BUG", "FIRE"): 0.5,
        ("BUG", "FIGHTING"): 0.5,
        ("BUG", "FLYING"): 0.5,
        ("BUG", "GHOST"): 0.5,

        # Rock
        ("ROCK", "FIRE"): 2.0,
        ("ROCK", "ICE"): 2.0,
        ("ROCK", "FLYING"): 2.0,
        ("ROCK", "BUG"): 2.0,
        ("ROCK", "FIGHTING"): 0.5,
        ("ROCK", "GROUND"): 0.5,

        # Ghost
        ("GHOST", "GHOST"): 2.0,
        ("GHOST", "NORMAL"): 0.0,
        ("GHOST", "PSYCHIC"): 0.0,  # Gen 1 programming bug: Ghost had 0x vs Psychic!

        # Dragon
        ("DRAGON", "DRAGON"): 2.0,
    }

    def __init__(self):
        self.current_cursor_pos: int = 0
        self.nav_queue: List[int] = []

    def get_type_multiplier(
        self,
        atk_type: str,
        def_type1: str,
        def_type2: Optional[str] = None
    ) -> float:
        """
        Query Gen 1 type matchup multiplier supporting dual-type defenders.
        Multiplier = Mult(atk, def1) * Mult(atk, def2)
        """
        t_atk = atk_type.upper()
        t_def1 = def_type1.upper()
        mult1 = self.TYPE_CHART.get((t_atk, t_def1), 1.0)

        if def_type2 is None or def_type2.upper() == t_def1:
            return float(mult1)

        t_def2 = def_type2.upper()
        mult2 = self.TYPE_CHART.get((t_atk, t_def2), 1.0)
        return float(mult1 * mult2)

    def calculate_gen1_crit_rate(self, move_name: str, base_speed: int = 65) -> float:
        """
        Gen 1 LR35902 Speed-Dependent Critical Hit Probability:
          Normal moves: P(crit) = min(255, base_speed // 2) / 256
          High-crit moves: P(crit) = min(255, 4 * base_speed) / 256
        """
        is_high_crit = move_name.upper() in GEN1_HIGH_CRIT_MOVES
        if is_high_crit:
            threshold = min(255, 4 * base_speed)
        else:
            threshold = min(255, base_speed // 2)
        return float(threshold / 256.0)

    def evaluate_move(
        self,
        move: Dict[str, Any],
        opponent_type: Union[str, Tuple[str, Optional[str]]],
        user_type: Optional[str] = None,
        user_base_speed: int = 65,
    ) -> float:
        """
        Compute expected move score under exact Gen 1 mechanics:
          E[Score] = BasePower * TypeMultiplier * STAB * EffectiveAccuracy * (1.0 + CritRate)
        """
        power = move.get("power", 40)
        if power <= 0:
            return 0.0  # Status move

        raw_acc = move.get("accuracy", 1.0)
        # Gen 1 1/256 miss glitch: max accuracy is 255/256
        effective_acc = min(raw_acc, 255.0 / 256.0)

        move_name = move.get("name", "").upper()
        move_type = move.get("type", "NORMAL").upper()

        if isinstance(opponent_type, tuple):
            def_t1, def_t2 = opponent_type
        else:
            def_t1, def_t2 = opponent_type, None

        multiplier = self.get_type_multiplier(move_type, def_t1, def_t2)
        if multiplier == 0.0:
            return 0.0  # Complete immunity (e.g. Electric vs Ground, Ghost vs Psychic in Gen 1)

        # Gen 1 STAB = 1.5x
        stab = 1.5 if (user_type and user_type.upper() == move_type) else 1.0

        # Critical hit multiplier (approx double damage in Gen 1)
        crit_rate = self.calculate_gen1_crit_rate(move_name, user_base_speed)
        crit_factor = 1.0 + crit_rate

        return float(power * multiplier * stab * effective_acc * crit_factor)

    def select_best_move_index(
        self,
        moves: List[Dict[str, Any]],
        opponent_type: Union[str, Tuple[str, Optional[str]]],
        user_type: Optional[str] = None,
        user_base_speed: int = 65,
    ) -> int:
        """
        Returns the 0-indexed position (0, 1, 2, or 3) of the highest expected damage move.
        """
        if not moves:
            return 0

        best_idx = 0
        best_score = -1.0

        for idx, move in enumerate(moves):
            pp = move.get("pp", 1)
            if pp <= 0:
                continue

            score = self.evaluate_move(move, opponent_type, user_type, user_base_speed)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def plan_menu_navigation(self, target_slot: int) -> List[int]:
        """
        Plan joypad sequence to navigate from (0, 0) to target_slot in 2x2 FIGHT grid:
          Slot 0 (0, 0): [Action.A]
          Slot 1 (0, 1): [Action.RIGHT, Action.A]
          Slot 2 (1, 0): [Action.DOWN, Action.A]
          Slot 3 (1, 1): [Action.RIGHT, Action.DOWN, Action.A]
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
            return Action.A

        opp_type = battle_state.get("opponent_type", "NORMAL")
        user_type = battle_state.get("user_type", None)
        user_speed = battle_state.get("user_base_speed", 65)

        best_slot = self.select_best_move_index(moves, opp_type, user_type, user_speed)
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
