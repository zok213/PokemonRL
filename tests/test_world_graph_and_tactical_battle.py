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
    # Pewter City without badge -> Gym target via BFS waypoint
    sg_gym = nav.get_subgoal(2, 16, 25, badges=0)
    assert sg_gym.goal_name == "advance_pewter_gym_entrance"
    assert sg_gym.target_coord == (16, 24)

    # In front of Gym door -> warp enter
    sg_door = nav.get_subgoal(2, 16, 18, badges=0)
    assert sg_door.goal_name == "enter_pewter_gym"
    assert sg_door.target_coord == (16, 17)
    assert sg_door.is_warp is True

    # Pewter City with Boulder Badge exiting gym -> walk west out of enclosure
    # Pewter City post-Brock: route to Route 3 via authentic ROM BFS
    sg_r3_start = nav.get_subgoal(2, 16, 18, badges=1)
    assert sg_r3_start.goal_name == "east_to_route3"
    assert sg_r3_start.target_coord == (15, 18)

    # Pewter City approaching border -> Route 3 warp target
    sg_r3_border = nav.get_subgoal(2, 39, 17, badges=1)
    assert sg_r3_border.goal_name == "east_to_route3"
    assert sg_r3_border.target_coord == (40, 17)
    assert sg_r3_border.is_warp is True

    # Inside Pewter Gym -> Brock target
    sg_brock = nav.get_subgoal(54, 4, 10)
    assert sg_brock.goal_name == "challenge_brock"
    assert sg_brock.target_coord == (4, 9)

    sg_brock_at_2 = nav.get_subgoal(54, 4, 2)
    assert sg_brock_at_2.goal_name == "interact_brock"
    assert sg_brock_at_2.target_coord == (4, 1)
    assert sg_brock_at_2.is_interaction is True

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

    # 10. In Cerulean City post-Misty with low HP (5/63) -> route to Pokecenter
    sg_post_misty_heal = nav.get_subgoal(3, 30, 19, badges=3, hp=5, max_hp=63)
    assert sg_post_misty_heal.goal_name == "approach_pokecenter"

    # 11. In Cerulean City fully healed without SS Ticket -> route north to Route 24
    sg_to_r24 = nav.get_subgoal(3, 19, 18, badges=3, hp=63, max_hp=63, has_ss_ticket=False)
    assert sg_to_r24.goal_name == "north_to_route24"

    # 12. At Cerulean northern border tile -> warp to Route 24
    sg_r24_warp = nav.get_subgoal(3, 20, 0, badges=3, hp=63, max_hp=63, has_ss_ticket=False)
    assert sg_r24_warp.goal_name == "enter_route24"
    assert sg_r24_warp.is_warp is True

    # 13. On Route 24 Nugget Bridge -> advance north
    sg_bridge = nav.get_subgoal(35, 10, 30, badges=3, hp=63, max_hp=63, has_ss_ticket=False)
    assert sg_bridge.goal_name == "route24_nugget_bridge"

    # 14. On Route 25 towards Bill's House -> advance east
    sg_r25 = nav.get_subgoal(36, 10, 8, badges=3, hp=63, max_hp=63, has_ss_ticket=False)
    assert sg_r25.goal_name == "route25_advance_east"

    # 15. In Bill's House -> Clefairy interaction, PC interaction, human Bill interaction
    sg_bill_mon = nav.get_subgoal(88, 3, 5, badges=3, hp=63, max_hp=63, bill_said_use_pc=False)
    assert sg_bill_mon.goal_name == "approach_bill_pokemon"

    sg_bill_pc = nav.get_subgoal(88, 1, 5, badges=3, hp=63, max_hp=63, bill_said_use_pc=True, used_cell_separator=False)
    assert sg_bill_pc.goal_name == "interact_bills_pc"
    assert sg_bill_pc.is_interaction is True

    sg_bill_human = nav.get_subgoal(88, 4, 5, badges=3, hp=63, max_hp=63, bill_said_use_pc=True, used_cell_separator=True, has_ss_ticket=False)
    assert sg_bill_human.goal_name == "interact_bill_human"
    assert sg_bill_human.is_interaction is True

    sg_bill_exit = nav.get_subgoal(88, 3, 5, badges=3, hp=63, max_hp=63, has_ss_ticket=True)
    assert sg_bill_exit.goal_name == "approach_exit_bills_house"

    # 16. Returning through Trashed House (Map 62) -> advance to wall hole
    sg_trashed = nav.get_subgoal(62, 3, 5, badges=3, hp=63, max_hp=63, has_ss_ticket=True)
    assert sg_trashed.goal_name == "advance_trashed_house_hole"

    # 17. Backyard south towards Route 5
    sg_backyard = nav.get_subgoal(3, 27, 9, badges=3, hp=63, max_hp=63, has_ss_ticket=True)
    assert sg_backyard.goal_name == "south_to_route5"


def test_world_graph_vermilion_city_and_gym():
    """Verify complete topological navigation across Route 6, Vermilion City, and Vermilion Gym."""
    nav = WorldGraphNavigator()

    # 1. Route 6 south into Vermilion gatehouse
    sg1 = nav.get_subgoal(17, 9, 35, badges=3, party_moves=[15])
    assert sg1.goal_name == "enter_vermilion_from_route6"
    assert sg1.target_coord == (9, 36)
    assert sg1.is_warp is True

    # 2. Vermilion City entry: route to Pokecenter entrance
    sg2 = nav.get_subgoal(5, 18, 0, badges=3, party_moves=[15], visited_vermilion_pokecenter=False)
    assert sg2.goal_name == "approach_vermilion_pokecenter"

    # 3. In front of Vermilion Pokecenter door: enter
    sg3 = nav.get_subgoal(5, 11, 4, badges=3, party_moves=[15], visited_vermilion_pokecenter=False)
    assert sg3.goal_name == "enter_vermilion_pokecenter"
    assert sg3.target_coord == (11, 3)
    assert sg3.is_warp is True

    # 4. Outside Pokecenter with Cut learned: route to Cut tree blocking Gym
    sg4 = nav.get_subgoal(5, 15, 17, badges=3, party_moves=[15], visited_vermilion_pokecenter=True)
    assert sg4.goal_name == "cut_vermilion_gym_tree"
    assert sg4.is_interaction is True

    # 5. In front of Vermilion Gym door: enter Gym
    sg5 = nav.get_subgoal(5, 12, 19, badges=3, party_moves=[15], visited_vermilion_pokecenter=True)
    assert sg5.goal_name == "enter_vermilion_gym"
    assert sg5.target_coord == (12, 18)
    assert sg5.is_warp is True

    # 6. Inside Vermilion Gym: approach Lt. Surge
    sg6 = nav.get_subgoal(92, 4, 17, badges=3, party_moves=[15])
    assert sg6.goal_name == "approach_lt_surge"

    # 7. In front of Lt. Surge: challenge for Thunder Badge
    sg7 = nav.get_subgoal(92, 5, 2, badges=3, party_moves=[15])
    assert sg7.goal_name == "challenge_lt_surge"
    assert sg7.target_coord == (5, 1)
    assert sg7.is_interaction is True

    # 8. Post-battle in Gym with Thunder Badge (badges=7): approach exit
    sg8 = nav.get_subgoal(92, 5, 2, badges=7, party_moves=[15])
    assert sg8.goal_name == "approach_gym_exit"

    # 9. At Vermilion Gym door mat: exit to city
    sg9 = nav.get_subgoal(92, 4, 17, badges=7, party_moves=[15])
    assert sg9.goal_name == "exit_vermilion_gym"
    assert sg9.target_coord == (4, 18)
    assert sg9.is_warp is True

    # 10. In Vermilion City with Thunder Badge: advance east to Route 11 / Diglett's Cave
    sg10 = nav.get_subgoal(5, 12, 19, badges=7, party_moves=[15])
    assert sg10.goal_name == "east_to_route11"
    assert sg10.target_coord == (12, 20) # Step down from gym door mat towards (39, 14)

    sg10_border = nav.get_subgoal(5, 39, 14, badges=7, party_moves=[15])
    assert sg10_border.goal_name == "enter_route11"
    assert sg10_border.is_warp is True


def test_world_graph_digletts_cave_and_route2():
    """Verify topological navigation across Route 11, Diglett's Cave, and Route 2."""
    nav = WorldGraphNavigator()

    # 1. On Route 11 towards Diglett's Cave entrance building at (4, 5)
    sg1 = nav.get_subgoal(22, 0, 6, badges=7, party_moves=[15])
    assert sg1.goal_name == "approach_digletts_cave_route11"

    sg1_door = nav.get_subgoal(22, 4, 5, badges=7, party_moves=[15])
    assert sg1_door.goal_name == "enter_digletts_cave_route11"
    assert sg1_door.target_coord == (4, 4)
    assert sg1_door.is_warp is True

    # 2. Inside Route 11 entrance building (Map 85): descend stairs at (4, 4)
    sg2 = nav.get_subgoal(85, 2, 7, badges=7, party_moves=[15])
    assert sg2.goal_name == "approach_cave_stairs"

    sg2_stairs = nav.get_subgoal(85, 4, 4, badges=7, party_moves=[15])
    assert sg2_stairs.goal_name == "descend_digletts_cave"
    assert sg2_stairs.is_warp is True

    # 3. Inside Diglett's Cave (Map 197): traverse cavern to northern ladder at (5, 5)
    sg3 = nav.get_subgoal(197, 37, 31, badges=7, party_moves=[15])
    assert sg3.goal_name == "traverse_digletts_cave"

    sg3_ladder = nav.get_subgoal(197, 5, 5, badges=7, party_moves=[15])
    assert sg3_ladder.goal_name == "ascend_to_route2"
    assert sg3_ladder.target_coord == (5, 4)
    assert sg3_ladder.is_warp is True

    # 4. Inside Route 2 entrance building (Map 46): exit to Route 2 at (2, 7)
    sg4 = nav.get_subgoal(46, 4, 4, badges=7, party_moves=[15])
    assert sg4.goal_name == "approach_route2_exit"

    sg4_door = nav.get_subgoal(46, 2, 7, badges=7, party_moves=[15])
    assert sg4_door.goal_name == "exit_digletts_cave_route2"
    assert sg4_door.target_coord == (2, 8)
    assert sg4_door.is_warp is True

    # 5. On Route 2: cut tree at (5, 10)
    sg5 = nav.get_subgoal(13, 12, 10, badges=7, party_moves=[15])
    assert sg5.goal_name == "approach_route2_tree"

    sg5_cut = nav.get_subgoal(13, 5, 11, badges=7, party_moves=[15])
    assert sg5_cut.goal_name == "cut_route2_tree"
    assert sg5_cut.target_coord == (5, 10)
    assert sg5_cut.is_interaction is True

    # 6. Past cut tree on Route 2: advance north to Pewter City at (8, 0)
    sg6 = nav.get_subgoal(13, 5, 8, badges=7, party_moves=[15])
    assert sg6.goal_name == "route2_north_to_pewter"

    sg6_pewter = nav.get_subgoal(13, 8, 0, badges=7, party_moves=[15])
    assert sg6_pewter.goal_name == "enter_pewter_from_route2"
    assert sg6_pewter.target_coord == (8, -1)
    assert sg6_pewter.is_warp is True


def test_world_graph_cerulean_bike_shop_and_route9():
    """Verify topological navigation across Cerulean Bike Shop and Route 9."""
    nav = WorldGraphNavigator()

    # 1. In Cerulean City with Thunder Badge but no Bike: route to Bike Shop (13, 25)
    sg1 = nav.get_subgoal(3, 0, 18, badges=7, bag=[45], party_moves=[15])
    assert sg1.goal_name == "approach_bike_shop"

    sg1_door = nav.get_subgoal(3, 13, 25, badges=7, bag=[45], party_moves=[15])
    assert sg1_door.goal_name == "enter_bike_shop"
    assert sg1_door.is_warp is True

    # 2. Inside Bike Shop (Map 66) without Bike: route to Clerk at (6, 4)
    sg2 = nav.get_subgoal(66, 2, 7, badges=7, bag=[45], party_moves=[15])
    assert sg2.goal_name == "approach_clerk"

    sg2_clerk = nav.get_subgoal(66, 6, 4, badges=7, bag=[45], party_moves=[15])
    assert sg2_clerk.goal_name == "redeem_bicycle"
    assert sg2_clerk.is_interaction is True

    # 3. Inside Bike Shop with Bike: route to exit door at (2, 7)
    sg3 = nav.get_subgoal(66, 6, 4, badges=7, bag=[6], party_moves=[15])
    assert sg3.goal_name == "approach_bike_shop_exit"

    sg3_door = nav.get_subgoal(66, 2, 7, badges=7, bag=[6], party_moves=[15])
    assert sg3_door.goal_name == "exit_bike_shop"
    assert sg3_door.is_warp is True

    # 4. In Cerulean City with Bike: route to South Cut tree at (19, 27)
    sg4 = nav.get_subgoal(3, 13, 25, badges=7, bag=[6], party_moves=[15])
    assert sg4.goal_name == "approach_cerulean_cut_tree"

    sg4_cut = nav.get_subgoal(3, 19, 27, badges=7, bag=[6], party_moves=[15])
    assert sg4_cut.goal_name == "cut_cerulean_tree"
    assert sg4_cut.target_coord == (19, 28)
    assert sg4_cut.is_interaction is True

    # 5. Past cut tree on Cerulean south/east corridor: advance to Route 9
    sg5 = nav.get_subgoal(3, 19, 29, badges=7, bag=[6], party_moves=[15])
    assert sg5.goal_name == "advance_to_route9"

    sg5_border = nav.get_subgoal(3, 39, 17, badges=7, bag=[6], party_moves=[15])
    assert sg5_border.goal_name == "enter_route9"
    assert sg5_border.is_warp is True

    # 6. On Route 9: cut tree at (5, 8) and advance east to Route 10
    sg6 = nav.get_subgoal(20, 0, 9, badges=7, bag=[6], party_moves=[15])
    assert sg6.goal_name == "approach_route9_tree"

    sg6_cut = nav.get_subgoal(20, 4, 8, badges=7, bag=[6], party_moves=[15])
    assert sg6_cut.goal_name == "cut_route9_tree"
    assert sg6_cut.target_coord == (5, 8)
    assert sg6_cut.is_interaction is True

    sg7 = nav.get_subgoal(20, 6, 8, badges=7, bag=[6], party_moves=[15])
    assert sg7.goal_name == "advance_route9_east"

    sg7_r10 = nav.get_subgoal(20, 59, 8, badges=7, bag=[6], party_moves=[15])
    assert sg7_r10.goal_name == "enter_route10"
    assert sg7_r10.is_warp is True


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
    # Warm-up pass to initialize PyTorch JIT / CPU dispatcher
    _ = policy.decide(state_t, options)
    decision = policy.decide(state_t, options)

    assert decision.selected_option == "advance"
    assert decision.confidence > 0.90
    assert decision.latency_ms < 15.0  # CPU latency < 15 ms
