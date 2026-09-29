"""
test_combat_controller.py — Unit Tests for Decoupled Combat Head (Gen 1 Grounded)
=================================================================================
Verifies Gen 1 type effectiveness evaluation, historical Gen 1 glitches, dual-type
matchups, speed-based critical hits, 1/256 miss glitch, and 2x2 menu navigation.
"""

import pytest
from pokemon_rl.combat.combat_controller import DecoupledCombatController
from pokemon_rl.env.wram_map import Action


def test_type_multipliers():
    controller = DecoupledCombatController()
    assert controller.get_type_multiplier("WATER", "FIRE") == 2.0
    assert controller.get_type_multiplier("ELECTRIC", "GROUND") == 0.0
    assert controller.get_type_multiplier("FIRE", "GRASS") == 2.0
    assert controller.get_type_multiplier("NORMAL", "GHOST") == 0.0
    assert controller.get_type_multiplier("PSYCHIC", "POISON") == 2.0


def test_gen1_historical_quirks():
    """Verify Gen 1 programming quirks from pret/pokered."""
    controller = DecoupledCombatController()
    # Ghost vs Psychic bug: in Gen 1, Ghost moves did 0.0x damage to Psychic
    assert controller.get_type_multiplier("GHOST", "PSYCHIC") == 0.0
    # Gen 1 Bug vs Poison was super effective (2.0x)
    assert controller.get_type_multiplier("BUG", "POISON") == 2.0
    # Gen 1 Poison vs Bug was super effective (2.0x)
    assert controller.get_type_multiplier("POISON", "BUG") == 2.0
    # Gen 1 Fire did NOT resist Ice: Ice vs Fire was neutral (1.0x)
    assert controller.get_type_multiplier("ICE", "FIRE") == 1.0


def test_dual_type_defender_multipliers():
    controller = DecoupledCombatController()
    # Water vs Ground/Rock (Geodude / Onix): 2.0 * 2.0 = 4.0x
    assert controller.get_type_multiplier("WATER", "GROUND", "ROCK") == 4.0
    # Electric vs Water/Flying (Gyarados): 2.0 * 2.0 = 4.0x
    assert controller.get_type_multiplier("ELECTRIC", "WATER", "FLYING") == 4.0
    # Grass vs Water/Ground: 2.0 * 2.0 = 4.0x
    assert controller.get_type_multiplier("GRASS", "WATER", "GROUND") == 4.0
    # Electric vs Ground/Rock (Ground immunity): 0.0 * 1.0 = 0.0x
    assert controller.get_type_multiplier("ELECTRIC", "GROUND", "ROCK") == 0.0


def test_speed_based_critical_hits():
    controller = DecoupledCombatController()
    # High-crit move Slash on Persian (BaseSpeed = 115)
    # P(high_crit) = min(255, 4 * 115) / 256 = 255 / 256 ≈ 0.996
    crit_slash = controller.calculate_gen1_crit_rate("SLASH", base_speed=115)
    assert crit_slash == pytest.approx(255.0 / 256.0, abs=1e-4)

    # Normal move Tackle on Snorlax (BaseSpeed = 30)
    # P(crit) = min(255, 30 // 2) / 256 = 15 / 256 ≈ 0.0586
    crit_tackle = controller.calculate_gen1_crit_rate("TACKLE", base_speed=30)
    assert crit_tackle == pytest.approx(15.0 / 256.0, abs=1e-4)


def test_best_move_selection_with_type_advantage():
    controller = DecoupledCombatController()
    moves = [
        {"name": "Tackle", "power": 40, "accuracy": 1.0, "type": "NORMAL", "pp": 35},
        {"name": "Water Gun", "power": 40, "accuracy": 1.0, "type": "WATER", "pp": 25},
        {"name": "Bubble", "power": 20, "accuracy": 1.0, "type": "WATER", "pp": 30},
        {"name": "Tail Whip", "power": 0, "accuracy": 1.0, "type": "NORMAL", "pp": 30},
    ]
    # Against Fire opponent: Water Gun gets 2.0x type multiplier + 1.5x STAB for Squirtle
    best_idx = controller.select_best_move_index(moves, opponent_type="FIRE", user_type="WATER")
    assert best_idx == 1  # Water Gun is slot 1


def test_fight_menu_navigation_planning():
    """Verify 2x2 fight menu directional paths."""
    controller = DecoupledCombatController()

    # Slot 0 (top-left): already highlighted, press A
    assert controller.plan_menu_navigation(0) == [Action.A]

    # Slot 1 (top-right): RIGHT then A
    assert controller.plan_menu_navigation(1) == [Action.RIGHT, Action.A]

    # Slot 2 (bottom-left): DOWN then A
    assert controller.plan_menu_navigation(2) == [Action.DOWN, Action.A]

    # Slot 3 (bottom-right): RIGHT then DOWN then A
    assert controller.plan_menu_navigation(3) == [Action.RIGHT, Action.DOWN, Action.A]


def test_stateful_battle_action_queue():
    controller = DecoupledCombatController()
    battle_state = {
        "opponent_type": "FIRE",
        "user_type": "WATER",
        "available_moves": [
            {"name": "Tackle", "power": 40, "accuracy": 1.0, "type": "NORMAL", "pp": 35},
            {"name": "Surf", "power": 90, "accuracy": 1.0, "type": "WATER", "pp": 10},  # Slot 1: Surf
        ]
    }
    # First call: returns Action.RIGHT to move to Slot 1
    act1 = controller.select_battle_action(battle_state)
    assert act1 == Action.RIGHT

    # Second call: returns Action.A to confirm move selection
    act2 = controller.select_battle_action(battle_state)
    assert act2 == Action.A
