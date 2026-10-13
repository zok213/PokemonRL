"""
tactical_battle.py — Authentic Gen 1 Tactical Battle Controller & Damage Evaluator
===================================================================================
Grounds combat decisions directly in authentic Pokémon Red LR35902 battle mechanics:
  - Exact Gen 1 Type Effectiveness Matrix (external/pokered/data/types/type_matchups.asm)
  - Gen 1 Damage Formula: ((2*L/5 + 2) * Power * (A/D) / 50 + 2) * STAB * TypeMult
  - Dynamic affordance question synthesis with expected damage calculations
  - Trainer battle legality enforcement (wIsInBattle == 2 strictly forbids RUN)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# =============================================================================
# 1. CANONICAL GEN 1 TYPE MATCHUPS MATRIX
# =============================================================================
# Direct transcription from external/pokered/data/types/type_matchups.asm
TYPE_MATCHUPS: Dict[Tuple[str, str], float] = {
    # Water
    ("WATER", "FIRE"): 2.0,
    ("WATER", "ROCK"): 2.0,
    ("WATER", "GROUND"): 2.0,
    ("WATER", "WATER"): 0.5,
    ("WATER", "GRASS"): 0.5,
    ("WATER", "DRAGON"): 0.5,

    # Fire
    ("FIRE", "GRASS"): 2.0,
    ("FIRE", "ICE"): 2.0,
    ("FIRE", "BUG"): 2.0,
    ("FIRE", "FIRE"): 0.5,
    ("FIRE", "WATER"): 0.5,
    ("FIRE", "ROCK"): 0.5,
    ("FIRE", "DRAGON"): 0.5,

    # Grass
    ("GRASS", "WATER"): 2.0,
    ("GRASS", "ROCK"): 2.0,
    ("GRASS", "GROUND"): 2.0,
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

    # Normal
    ("NORMAL", "ROCK"): 0.5,
    ("NORMAL", "GHOST"): 0.0,

    # Fighting
    ("FIGHTING", "NORMAL"): 2.0,
    ("FIGHTING", "ROCK"): 2.0,
    ("FIGHTING", "ICE"): 2.0,
    ("FIGHTING", "POISON"): 0.5,
    ("FIGHTING", "FLYING"): 0.5,
    ("FIGHTING", "PSYCHIC"): 0.5,
    ("FIGHTING", "BUG"): 0.5,
    ("FIGHTING", "GHOST"): 0.0,

    # Flying
    ("FLYING", "GRASS"): 2.0,
    ("FLYING", "FIGHTING"): 2.0,
    ("FLYING", "BUG"): 2.0,
    ("FLYING", "ELECTRIC"): 0.5,
    ("FLYING", "ROCK"): 0.5,

    # Poison
    ("POISON", "GRASS"): 2.0,
    ("POISON", "BUG"): 2.0,
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

    # Rock
    ("ROCK", "FIRE"): 2.0,
    ("ROCK", "ICE"): 2.0,
    ("ROCK", "FLYING"): 2.0,
    ("ROCK", "BUG"): 2.0,
    ("ROCK", "FIGHTING"): 0.5,
    ("ROCK", "GROUND"): 0.5,

    # Bug
    ("BUG", "GRASS"): 2.0,
    ("BUG", "POISON"): 2.0,
    ("BUG", "PSYCHIC"): 2.0,
    ("BUG", "FIRE"): 0.5,
    ("BUG", "FIGHTING"): 0.5,
    ("BUG", "FLYING"): 0.5,
    ("BUG", "GHOST"): 0.5,

    # Ghost
    ("GHOST", "GHOST"): 2.0,
    ("GHOST", "NORMAL"): 0.0,
    ("GHOST", "PSYCHIC"): 0.0, # Gen 1 quirk: Lick has no effect on Psychic

    # Psychic
    ("PSYCHIC", "FIGHTING"): 2.0,
    ("PSYCHIC", "POISON"): 2.0,
    ("PSYCHIC", "PSYCHIC"): 0.5,

    # Ice
    ("ICE", "GRASS"): 2.0,
    ("ICE", "GROUND"): 2.0,
    ("ICE", "FLYING"): 2.0,
    ("ICE", "DRAGON"): 2.0,
    ("ICE", "WATER"): 0.5,
    ("ICE", "ICE"): 0.5,

    # Dragon
    ("DRAGON", "DRAGON"): 2.0,
}


# =============================================================================
# 2. CANONICAL GEN 1 MOVE DATABASE
# =============================================================================
@dataclass
class MoveData:
    name: str
    move_type: str
    power: int
    accuracy: float
    category: str  # "PHYSICAL", "SPECIAL", "STATUS"


GEN1_MOVES: Dict[int, MoveData] = {
    1:  MoveData("POUND", "NORMAL", 40, 1.0, "PHYSICAL"),
    10: MoveData("SCRATCH", "NORMAL", 40, 1.0, "PHYSICAL"),
    22: MoveData("VINE WHIP", "GRASS", 35, 1.0, "SPECIAL"),
    33: MoveData("TACKLE", "NORMAL", 35, 0.95, "PHYSICAL"),
    39: MoveData("TAIL WHIP", "NORMAL", 0, 1.0, "STATUS"),
    45: MoveData("GROWL", "NORMAL", 0, 1.0, "STATUS"),
    52: MoveData("EMBER", "FIRE", 40, 1.0, "SPECIAL"),
    55: MoveData("WATER GUN", "WATER", 40, 1.0, "SPECIAL"),
    64: MoveData("PECK", "FLYING", 35, 1.0, "PHYSICAL"),
    70: MoveData("STRING SHOT", "BUG", 0, 0.95, "STATUS"),
    81: MoveData("STRING SHOT", "BUG", 0, 0.95, "STATUS"),
    40: MoveData("POISON STING", "POISON", 15, 1.0, "PHYSICAL"),
    88: MoveData("ROCK THROW", "ROCK", 50, 0.65, "PHYSICAL"),
    106: MoveData("HARDEN", "NORMAL", 0, 1.0, "STATUS"),
}

# Default species typing for early-game encounters
SPECIES_TYPES: Dict[int, Tuple[str, Optional[str]]] = {
    0x04: ("FIRE", None),         # Charmander
    0x99: ("WATER", None),        # Squirtle
    0x09: ("GRASS", "POISON"),    # Bulbasaur
    0x7B: ("BUG", None),          # Caterpie
    0x7C: ("BUG", "POISON"),      # Metapod
    0x70: ("BUG", "POISON"),      # Weedle (0x70 = 112)
    0x71: ("BUG", "POISON"),      # Kakuna
    0x24: ("NORMAL", "FLYING"),   # Pidgey
    0xA5: ("NORMAL", None),       # Rattata
    0x54: ("ELECTRIC", None),     # Pikachu
    0x67: ("ROCK", "GROUND"),     # Geodude (Brock)
    0x63: ("ROCK", "GROUND"),     # Onix (Brock)
}


class TacticalBattleController:
    """
    Evaluates turn-based combat states, executes damage calculations,
    and returns rank-ordered combat affordances with explainable criteria.
    """

    @staticmethod
    def get_type_multiplier(attacker_type: str, defender_types: Tuple[str, Optional[str]]) -> float:
        mult = 1.0
        t1, t2 = defender_types
        if (attacker_type, t1) in TYPE_MATCHUPS:
            mult *= TYPE_MATCHUPS[(attacker_type, t1)]
        if t2 and (attacker_type, t2) in TYPE_MATCHUPS:
            mult *= TYPE_MATCHUPS[(attacker_type, t2)]
        return mult

    @staticmethod
    def calculate_expected_damage(
        attacker_level: int,
        attacker_type: str,
        move: MoveData,
        defender_types: Tuple[str, Optional[str]],
        defender_hp: int
    ) -> Tuple[float, float, str]:
        """
        Calculate expected damage and effectiveness tag.
        Returns: (expected_damage, type_mult, effectiveness_tag)
        """
        if move.power == 0 or move.category == "STATUS":
            return (0.0, 1.0, "STATUS")

        type_mult = TacticalBattleController.get_type_multiplier(move.move_type, defender_types)
        stab = 1.5 if move.move_type == attacker_type else 1.0

        # Gen 1 base formula approximation with level scaling
        level_factor = (2.0 * attacker_level / 5.0) + 2.0
        base_dmg = (level_factor * move.power / 50.0) + 2.0
        expected = base_dmg * stab * type_mult * move.accuracy

        if type_mult > 1.5:
            eff_tag = "SUPER EFFECTIVE"
        elif type_mult == 0.0:
            eff_tag = "NO EFFECT"
        elif type_mult < 0.9:
            eff_tag = "NOT VERY EFFECTIVE"
        else:
            eff_tag = "NEUTRAL"

        return (expected, type_mult, eff_tag)

    @classmethod
    def evaluate_combat_affordances(
        cls,
        wram_snapshot: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, str], str]:
        """
        Analyze live Game Boy WRAM and synthesize verified legal combat affordances.
        Returns: (questions_dict, button_map, situation_description)
        """
        is_in_battle = wram_snapshot.get("in_battle", True)
        battle_type = wram_snapshot.get("battle_type", 1)  # 1 = wild, 2 = trainer
        active_mon = wram_snapshot.get("active_mon", {})
        player_hp = active_mon.get("hp", 20)
        player_max_hp = max(1, active_mon.get("max_hp", 20))
        player_lvl = active_mon.get("level", 5)

        opp_species = wram_snapshot.get("opponent_species", 0x70)
        opp_hp = wram_snapshot.get("opponent_hp", 20)
        opp_types = SPECIES_TYPES.get(opp_species, ("NORMAL", None))
        player_type = "FIRE"  # Default Charmander starter

        situation = (
            f"Combat ({'Trainer' if battle_type == 2 else 'Wild'}): "
            f"Player HP {player_hp}/{player_max_hp} vs Enemy (Type: {'/'.join(filter(None, opp_types))})"
        )

        # Parse moves (default to Scratch and Ember if not populated)
        move_ids = active_mon.get("moves", [10, 52])  # Scratch, Ember
        if not move_ids or all(m == 0 for m in move_ids):
            move_ids = [10, 52]

        move_pps = wram_snapshot.get("move_pps", [35, 40, 25, 30])
        if not move_pps or len(move_pps) < 4:
            move_pps = [35, 40, 25, 30]

        scored_moves = []
        for slot_idx, m_id in enumerate(move_ids[:4]):
            if m_id == 0:
                continue
            # If move has 0 PP, it cannot be legally selected in combat
            pp = move_pps[slot_idx] if slot_idx < len(move_pps) else 1
            if pp == 0:
                continue
            move = GEN1_MOVES.get(m_id, MoveData(f"MOVE_{m_id}", "NORMAL", 40, 1.0, "PHYSICAL"))
            exp_dmg, mult, tag = cls.calculate_expected_damage(
                player_lvl, player_type, move, opp_types, opp_hp
            )
            scored_moves.append((slot_idx, move, exp_dmg, mult, tag))

        # Sort moves by expected damage descending
        scored_moves.sort(key=lambda item: item[2], reverse=True)

        questions: Dict[str, str] = {}
        button_map: Dict[str, Any] = {}

        # 1. Best Attack Affordance
        if scored_moves:
            best_slot, best_move, best_dmg, best_mult, best_tag = scored_moves[0]
            tag_suffix = f" [{best_tag}]" if best_tag != "NEUTRAL" else ""
            opt_id = f"attack_{best_move.name.lower().replace(' ', '_')}"
            questions[opt_id] = (
                f"Should Red execute {best_move.name} (Power {best_move.power}, "
                f"ExpDmg ~{best_dmg:.0f}){tag_suffix}?"
            )
            button_map[opt_id] = "a"
            button_map["target_slot"] = best_slot
            button_map["target_move_name"] = best_move.name

            # 2. Secondary Attack Affordance if available
            if len(scored_moves) > 1:
                sec_slot, sec_move, sec_dmg, sec_mult, sec_tag = scored_moves[1]
                sec_id = f"attack_{sec_move.name.lower().replace(' ', '_')}"
                sec_tag_suffix = f" [{sec_tag}]" if sec_tag != "NEUTRAL" else ""
                questions[sec_id] = (
                    f"Should Red execute secondary move {sec_move.name} "
                    f"(Power {sec_move.power}, ExpDmg ~{sec_dmg:.0f}){sec_tag_suffix}?"
                )
                button_map[sec_id] = "a"
        else:
            # Fallback when all moves are exhausted (Struggle)
            button_map["target_slot"] = 0
            button_map["target_move_name"] = "STRUGGLE"

        # 3. Flee Affordance (STRICTLY ILLEGAL IN TRAINER BATTLES)
        if battle_type != 2:
            questions["run"] = "Should Red attempt to FLEE from this wild encounter to preserve HP?"
            button_map["run"] = "run"

        # 4. Potion Affordance if low HP
        if (player_hp / player_max_hp) < 0.4:
            questions["potion"] = "Should Red use Potion to restore 20 HP?"
            button_map["potion"] = "down"

        return questions, button_map, situation
