"""
test_world_graph_and_tactical_battle.py
=======================================
Unit and regression tests for:
  1. TacticalBattleController (Gen 1 battle math, type effectiveness, trainer battle legality)
  2. WorldGraphNavigator (Topological Kanto world graph, milestone progression, labyrinth navigation)
  3. LocalAffordancePolicy persistent weight loading & inference reproducibility
"""

import os
import pytest
import torch
from pathlib import Path

from pokemon_rl.agent.tactical_battle import TacticalBattleController, TYPE_MATCHUPS, GEN1_MOVES
from pokemon_rl.navigation.world_graph import WorldGraphNavigator, MapSubgoal
from pokemon_rl.agent.local_decision_policy import LocalAffordancePolicy, SMDPDecisionHarness


# =============================================================================
# 1. TACTICAL BATTLE CONTROLLER TESTS
# =============================================================================

def test_gen1_type_matchups():
    """Verify authentic Generation 1 type interactions according to pret/pokered."""
    # Water vs Fire: 2.0x
    assert TacticalBattleController.get_type_multiplier("WATER", ("FIRE", None)) == 2.0
    # Electric vs Ground: 0.0x (Immunity)
    assert TacticalBattleController.get_type_multiplier("ELECTRIC", ("GROUND", None)) == 0.0
    # Ghost vs Normal: 0.0x (Gen 1 immunity)
    assert TacticalBattleController.get_type_multiplier("GHOST", ("NORMAL", None)) == 0.0
    # Fire vs Grass/Poison (Dual type Bulbasaur): 2.0 * 1.0 = 2.0x
    assert TacticalBattleController.get_type_multiplier("FIRE", ("GRASS", "POISON")) == 2.0
    # Fire vs Bug/Poison (Dual type Weedle): 2.0 * 1.0 = 2.0x
    assert TacticalBattleController.get_type_multiplier("FIRE", ("BUG", "POISON")) == 2.0


def test_tactical_combat_affordances_super_effective_ranking():
    """Verify Ember is ranked above Scratch when fighting a Bug-type opponent."""
    snapshot = {
        "in_battle": True,
        "battle_type": 1,  # Wild
        "opponent_species": 0x70,  # Weedle (Bug/Poison)
        "opponent_hp": 15,
        "active_mon": {
            "hp": 20,
            "max_hp": 20,
            "level": 7,
            "moves": [10, 52],  # Scratch (Normal), Ember (Fire)
        }
    }
    questions, button_map, situation = TacticalBattleController.evaluate_combat_affordances(snapshot)

    assert "attack_ember" in questions
    assert "attack_scratch" in questions
    assert "SUPER EFFECTIVE" in questions["attack_ember"]
    assert "run" in questions  # Wild battle permits flee


def test_tactical_combat_trainer_battle_legality():
    """Verify Gen 1 rule: RUN is strictly prohibited in trainer battles (wIsInBattle == 2)."""
    snapshot = {
        "in_battle": True,
        "battle_type": 2,  # Trainer battle!
        "opponent_species": 0x7B,  # Caterpie
        "opponent_hp": 12,
        "active_mon": {
            "hp": 8,
            "max_hp": 20,
            "level": 6,
            "moves": [10, 52]
        }
    }
    questions, button_map, situation = TacticalBattleController.evaluate_combat_affordances(snapshot)

    # RUN must NOT exist in legal affordances for trainer battles
    assert "run" not in questions
    assert "run" not in button_map
    assert "Trainer" in situation


def test_tactical_combat_low_hp_potion_offer():
    """Verify that Potion affordance is synthesized when player HP drops below 40%."""
    snapshot = {
        "in_battle": True,
        "battle_type": 1,
        "opponent_species": 0x24,  # Pidgey
        "opponent_hp": 10,
        "active_mon": {
            "hp": 5,
            "max_hp": 20,  # 25% HP remaining
            "level": 5,
            "moves": [10]
        }
    }
    questions, button_map, situation = TacticalBattleController.evaluate_combat_affordances(snapshot)
    assert "potion" in questions
    assert "run" in questions


# =============================================================================
# 2. TOPOLOGICAL WORLD GRAPH NAVIGATOR TESTS
# =============================================================================

def test_world_graph_oak_lab():
    nav = WorldGraphNavigator()
    # Inside Oak's Lab at north desk
    sg1 = nav.get_subgoal(40, 5, 3)
    assert sg1.goal_name == "exit_lab"
    assert sg1.target_coord == (4, 11)

    # At exit mat
    sg2 = nav.get_subgoal(40, 4, 11)
    assert sg2.goal_name == "step_outside"
    assert sg2.is_warp is True


def test_world_graph_pallet_town():
    nav = WorldGraphNavigator()
    # In Pallet Town square
    sg1 = nav.get_subgoal(0, 10, 8)
    assert sg1.goal_name == "north_to_route1"
    assert sg1.target_coord == (10, 0)

    # At border tile
    sg2 = nav.get_subgoal(0, 10, 1)
    assert sg2.goal_name == "enter_route1"
    assert sg2.is_warp is True

    # East of road in Pallet Town
    sg3 = nav.get_subgoal(0, 13, 5)
    assert sg3.goal_name == "align_west_to_road"
    assert sg3.target_coord == (10, 6)


def test_world_graph_blues_and_reds_house():
    nav = WorldGraphNavigator()
    # Inside Blue's House at Daisy's table
    sg1 = nav.get_subgoal(39, 2, 4)
    assert sg1.goal_name == "exit_blues_house"
    assert sg1.target_coord == (2, 7)
    assert sg1.is_warp is False

    # At door mat of Blue's house
    sg2 = nav.get_subgoal(39, 2, 7)
    assert sg2.goal_name == "step_outside_blues_house"
    assert sg2.target_coord == (2, 8)
    assert sg2.is_warp is True

    # Inside Red's House 1F
    sg3 = nav.get_subgoal(37, 2, 4)
    assert sg3.goal_name == "exit_reds_house"
    assert sg3.target_coord == (2, 7)

    # Inside Red's House 2F
    sg4 = nav.get_subgoal(38, 3, 5)
    assert sg4.goal_name == "descend_stairs"
    assert sg4.target_coord == (7, 1)


def test_world_graph_route1_waypoints():
    nav = WorldGraphNavigator()
    # Route 1 southern fence
    sg1 = nav.get_subgoal(12, 11, 29)
    assert sg1.goal_name == "bypass_fence_west"
    assert sg1.target_coord == (8, 28)

    # Route 1 jumping ledge at y=24
    sg2 = nav.get_subgoal(12, 8, 24)
    assert sg2.goal_name == "bypass_ledge_east"
    assert sg2.target_coord == (12, 24)

    # Approaching Viridian gate
    sg3 = nav.get_subgoal(12, 10, 1)
    assert sg3.goal_name == "enter_viridian"
    assert sg3.is_warp is True


def test_world_graph_viridian_forest_labyrinth():
    nav = WorldGraphNavigator()
    # Viridian Forest entrance corridor
    sg1 = nav.get_subgoal(51, 17, 47)
    assert sg1.goal_name == "forest_corridor_entry"
    assert sg1.target_coord == (17, 43)

    # Eastern crossway
    sg2 = nav.get_subgoal(51, 20, 41)
    assert sg2.goal_name == "forest_eastern_crossway"
    assert sg2.target_coord == (25, 41)

    # Approaching exit door
    sg3 = nav.get_subgoal(51, 1, 1)
    assert sg3.goal_name == "forest_exit_door"
    assert sg3.target_coord == (1, 0)


def test_world_graph_pewter_city_and_gym():
    nav = WorldGraphNavigator()
    # Pewter City without badge -> Gym target
    sg_gym = nav.get_subgoal(2, 16, 25, badges=0)
    assert sg_gym.goal_name == "advance_pewter_gym_entrance"
    assert sg_gym.target_coord == (16, 18)

    # Pewter City with Boulder Badge exiting gym -> walk west out of enclosure
    sg_r3_west = nav.get_subgoal(2, 16, 18, badges=1)
    assert sg_r3_west.goal_name == "exit_gym_enclosure_west"
    assert sg_r3_west.target_coord == (7, 18)

    # Pewter City on western avenue -> walk north to crossway
    sg_r3_north = nav.get_subgoal(2, 7, 14, badges=1)
    assert sg_r3_north.goal_name == "pewter_avenue_north"
    assert sg_r3_north.target_coord == (7, 8)

    # Pewter City on northern crossway -> walk east to column 32
    sg_r3_east = nav.get_subgoal(2, 20, 8, badges=1)
    assert sg_r3_east.goal_name == "pewter_crossway_east"
    assert sg_r3_east.target_coord == (32, 8)

    # Pewter City on column 32 -> walk south to Y=17
    sg_r3_south = nav.get_subgoal(2, 32, 10, badges=1)
    assert sg_r3_south.goal_name == "pewter_exit_channel_south"
    assert sg_r3_south.target_coord == (32, 17)

    # Pewter City approaching border -> Route 3 target
    sg_r3 = nav.get_subgoal(2, 32, 17, badges=1)
    assert sg_r3.goal_name == "east_to_route3"
    assert sg_r3.target_coord == (40, 17)

    # Inside Pewter Gym -> Brock target
    sg_brock = nav.get_subgoal(54, 4, 10)
    assert sg_brock.goal_name == "challenge_brock"
    assert sg_brock.target_coord == (4, 3)

    # Mt. Moon 1F entry and ladder subgoals
    sg_mm1f = nav.get_subgoal(59, 14, 35)
    assert sg_mm1f.goal_name == "mt_moon_1f_corridor_north"
    assert sg_mm1f.target_coord == (14, 25)

    # Mt. Moon B1F corridor south to ladder
    sg_b1f = nav.get_subgoal(60, 25, 15)
    assert sg_b1f.goal_name == "b1f_corridor_south"
    assert sg_b1f.target_coord == (25, 27)

    # Mt. Moon B2F cavern east
    sg_b2f = nav.get_subgoal(61, 15, 27)
    assert sg_b2f.goal_name == "b2f_bottom_cavern_east"
    assert sg_b2f.target_coord == (29, 27)


def test_world_graph_cerulean_city_and_gym():
    nav = WorldGraphNavigator()

    # 1. Cerulean City entry with fainted lead mon -> route to Pokecenter entrance
    sg_cc_heal = nav.get_subgoal(3, 0, 18, badges=1, hp=0, max_hp=55)
    assert sg_cc_heal.goal_name in ("approach_pokecenter", "cerulean_road_east")

    # 2. In front of Pokecenter door -> warp enter
    sg_cc_enter = nav.get_subgoal(3, 19, 18, badges=1, hp=0, max_hp=55)
    assert sg_cc_enter.goal_name == "enter_cerulean_pokecenter"
    assert sg_cc_enter.target_coord == (19, 17)
    assert sg_cc_enter.is_warp is True

    # 3. Inside Cerulean Pokecenter with low HP -> approach & speak to Nurse Joy
    sg_pk_joy = nav.get_subgoal(64, 3, 5, badges=1, hp=0, max_hp=55)
    assert sg_pk_joy.goal_name == "approach_nurse_joy"
    assert sg_pk_joy.target_coord == (3, 3)

    sg_pk_talk = nav.get_subgoal(64, 3, 3, badges=1, hp=0, max_hp=55)
    assert sg_pk_talk.goal_name == "interact_nurse_joy"
    assert sg_pk_talk.target_coord == (3, 2)

    # 4. Inside Cerulean Pokecenter fully healed -> exit south
    sg_pk_exit = nav.get_subgoal(64, 3, 3, badges=1, hp=55, max_hp=55)
    assert sg_pk_exit.goal_name == "exit_pokecenter"
    assert sg_pk_exit.target_coord == (3, 7)

    # 5. Outside Pokecenter fully healed -> route to Cerulean Gym
    sg_to_gym = nav.get_subgoal(3, 19, 18, badges=1, hp=55, max_hp=55)
    assert sg_to_gym.goal_name == "approach_cerulean_gym"

    # 6. In front of Cerulean Gym door -> enter Gym
    sg_enter_gym = nav.get_subgoal(3, 30, 20, badges=1, hp=55, max_hp=55)
    assert sg_enter_gym.goal_name == "enter_cerulean_gym"
    assert sg_enter_gym.target_coord == (30, 19)
    assert sg_enter_gym.is_warp is True

    # 7. Inside Cerulean Gym -> approach Misty avoiding pool
    sg_gym_approach = nav.get_subgoal(65, 4, 13, badges=1, hp=55, max_hp=55)
    assert sg_gym_approach.goal_name == "approach_misty"

    # 8. In front of Misty -> challenge Misty
    sg_misty = nav.get_subgoal(65, 4, 3, badges=1, hp=55, max_hp=55)
    assert sg_misty.goal_name == "challenge_misty"
    assert sg_misty.target_coord == (4, 2)

    # 9. After defeating Misty (Cascade Badge acquired, badges=3) -> exit Gym
    sg_exit_gym = nav.get_subgoal(65, 4, 5, badges=3, hp=55, max_hp=55)
    assert sg_exit_gym.goal_name == "exit_cerulean_gym"



# =============================================================================
# 3. LOCAL AFFORDANCE POLICY PERSISTENCE & INFERENCE REPRODUCIBILITY
# =============================================================================

def test_persistent_model_weights_loaded():
    """Verify that models/pokemonrl_affordance_model.pt loads and powers zero-shot decisions."""
    weights_path = Path("models/pokemonrl_affordance_model.pt")
    assert weights_path.exists(), "Trained model weights file models/pokemonrl_affordance_model.pt must exist"

    policy = LocalAffordancePolicy(weights_path=str(weights_path))
    assert policy.is_trained is True

    # Test inference latency & confidence
    harness = SMDPDecisionHarness(local_policy=policy)
    snapshot = {
        "map_id": 12,
        "x": 10,
        "y": 20,
        "in_battle": False,
        "badges": 0,
        "text_box_id": 0,
        "active_mon": {"hp": 20, "max_hp": 20, "level": 6}
    }
    options = {
        "advance": "Should Red advance north along Route 1 towards destination?",
        "retreat": "Should Red return south to Pallet Town to rest?"
    }

    state_t = harness.extract_state_tensor(snapshot, subgoal=(10, 2))
    decision = policy.decide(state_t, options)

    assert decision.selected_option == "advance"
    assert decision.confidence > 0.90
    assert decision.latency_ms < 15.0  # CPU latency < 15 ms
