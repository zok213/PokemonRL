"""
test_key_item_preconditions_and_world_graph.py
==============================================
Exhaustive verification of:
  1. KeyItemPreconditionChecker: authentic WRAM inventory & badge bitfield enforcement
     (HMs: Cut, Fly, Surf, Strength, Flash; Key Items: Silph Scope, Poké Flute, Card Key,
      Secret Key, S.S. Ticket, Lift Key, Gold Teeth).
  2. WorldGraphNavigator: full Kanto progression across Badges 3 through 8, intermediate
     dungeons, and Indigo Plateau Elite Four & Champion gauntlet.
"""

import pytest
from pokemon_rl.env.wram_map import RAMMap
from pokemon_rl.navigation.key_item_preconditions import KeyItemPreconditionChecker
from pokemon_rl.navigation.world_graph import WorldGraphNavigator, MapSubgoal


# =============================================================================
# 1. KEY ITEM & HM PRECONDITION TESTS
# =============================================================================

def test_key_item_precondition_cut_tree():
    checker = KeyItemPreconditionChecker

    # Case 1: No Cascade Badge -> False even with HM01 in bag
    state_no_badge = {"badges": 0, "bag": [RAMMap.ITEM_HM01_CUT], "party_moves": []}
    assert checker.can_cut_tree(state_no_badge) is False

    # Case 2: Has Cascade Badge but neither Cut move nor HM01 -> False
    state_no_item = {"badges": RAMMap.BADGE_CASCADE, "bag": [], "party_moves": []}
    assert checker.can_cut_tree(state_no_item) is False

    # Case 3: Has Cascade Badge + HM01 in bag -> True
    state_valid_hm = {"badges": RAMMap.BADGE_CASCADE, "bag": [RAMMap.ITEM_HM01_CUT], "party_moves": []}
    assert checker.can_cut_tree(state_valid_hm) is True

    # Case 4: Has Cascade Badge + Pokémon knows Cut (move 15) -> True
    state_valid_move = {"badges": RAMMap.BADGE_CASCADE, "bag": [], "party_moves": [15]}
    assert checker.can_cut_tree(state_valid_move) is True


def test_key_item_precondition_surf_and_strength():
    checker = KeyItemPreconditionChecker

    # Surf requires Soul Badge (bit 4) + HM03/Surf move
    assert checker.can_surf_water({"badges": RAMMap.BADGE_CASCADE, "bag": [RAMMap.ITEM_HM03_SURF]}) is False
    assert checker.can_surf_water({"badges": RAMMap.BADGE_SOUL, "bag": [RAMMap.ITEM_HM03_SURF]}) is True
    assert checker.can_surf_water({"badges": RAMMap.BADGE_SOUL, "party_moves": [57]}) is True

    # Strength requires Rainbow Badge (bit 3) + HM04/Strength move
    assert checker.can_push_boulder({"badges": RAMMap.BADGE_BOULDER, "bag": [RAMMap.ITEM_HM04_STRENGTH]}) is False
    assert checker.can_push_boulder({"badges": RAMMap.BADGE_RAINBOW, "bag": [RAMMap.ITEM_HM04_STRENGTH]}) is True
    assert checker.can_push_boulder({"badges": RAMMap.BADGE_RAINBOW, "party_moves": [70]}) is True


def test_key_item_preconditions_dungeon_and_quest_items():
    checker = KeyItemPreconditionChecker

    bag_all = [
        RAMMap.ITEM_SILPH_SCOPE,
        RAMMap.ITEM_POKE_FLUTE,
        RAMMap.ITEM_CARD_KEY,
        RAMMap.ITEM_SECRET_KEY,
        RAMMap.ITEM_S_S_TICKET,
        RAMMap.ITEM_LIFT_KEY,
    ]
    state = {"badges": 0, "bag": bag_all}

    assert checker.can_see_ghosts(state) is True
    assert checker.can_wake_snorlax(state) is True
    assert checker.can_unlock_silph_doors(state) is True
    assert checker.can_unlock_cinnabar_gym(state) is True
    assert checker.can_board_ss_anne(state) is True
    assert checker.can_operate_hideout_elevator(state) is True

    # Empty bag -> all False
    empty = {"badges": 0, "bag": []}
    assert checker.can_see_ghosts(empty) is False
    assert checker.can_wake_snorlax(empty) is False
    assert checker.can_unlock_silph_doors(empty) is False
    assert checker.can_unlock_cinnabar_gym(empty) is False
    assert checker.can_board_ss_anne(empty) is False
    assert checker.can_operate_hideout_elevator(empty) is False


def test_key_item_precondition_league_challenge():
    checker = KeyItemPreconditionChecker
    # Only all 8 badges (0xFF = 255) authorizes League Challenge
    assert checker.can_challenge_pokemon_league({"badges": 0x7F}) is False
    assert checker.can_challenge_pokemon_league({"badges": 0xFF}) is True

    status = checker.get_barrier_status({"badges": 0xFF, "bag": [RAMMap.ITEM_CARD_KEY]})
    assert status["can_challenge_league"] is True
    assert status["can_unlock_silph_doors"] is True
    assert status["can_see_ghosts"] is False


# =============================================================================
# 2. WORLD GRAPH PROGRESSION: BADGES 3 THROUGH 8 & ELITE FOUR
# =============================================================================

@pytest.fixture
def nav():
    return WorldGraphNavigator()


def test_world_graph_badge3_vermilion_and_ss_anne(nav):
    # In Vermilion City without Cut -> route to S.S. Anne dock
    sg_dock = nav.get_subgoal(5, 10, 10, badges=3, bag=[])
    assert "dock" in sg_dock.goal_name

    # In S.S. Anne Captain's room without HM01 -> interact with Captain
    sg_capt = nav.get_subgoal(101, 2, 3, badges=3, bag=[])
    assert sg_capt.goal_name == "interact_captain"
    assert sg_capt.is_interaction is True

    # In S.S. Anne Captain's room with HM01 -> exit
    sg_capt_exit = nav.get_subgoal(101, 2, 3, badges=3, bag=[RAMMap.ITEM_HM01_CUT])
    assert sg_capt_exit.goal_name == "exit_captains_room"
    assert sg_capt_exit.is_warp is True

    # In Vermilion City with Cut -> cut tree and enter gym
    sg_tree = nav.get_subgoal(5, 16, 17, badges=3, bag=[RAMMap.ITEM_HM01_CUT])
    assert sg_tree.goal_name == "cut_vermilion_gym_tree"
    assert sg_tree.is_interaction is True

    # Inside Vermilion Gym -> challenge Lt. Surge
    sg_surge = nav.get_subgoal(92, 4, 3, badges=3, bag=[RAMMap.ITEM_HM01_CUT])
    assert sg_surge.goal_name == "challenge_lt_surge"
    assert sg_surge.is_interaction is True

    # Inside Vermilion Gym after Thunder Badge acquired (badges=7) -> exit gym
    sg_exit_surge = nav.get_subgoal(92, 4, 17, badges=7, bag=[RAMMap.ITEM_HM01_CUT])
    assert sg_exit_surge.goal_name in ("approach_gym_exit", "exit_vermilion_gym")


def test_world_graph_badge4_rock_tunnel_and_celadon(nav):
    # Route 10 North -> enter Rock Tunnel
    sg_rt_enter = nav.get_subgoal(21, 11, 15, badges=7)
    assert sg_rt_enter.goal_name == "approach_rock_tunnel"

    # Rock Tunnel 1F -> navigate to B1F ladder
    sg_rt1f = nav.get_subgoal(82, 15, 2, badges=7)
    assert sg_rt1f.goal_name == "rock_tunnel_1f_to_b1f"

    # Rock Tunnel B1F -> navigate to exit ladder
    sg_rt_b1f = nav.get_subgoal(232, 20, 20, badges=7)
    assert sg_rt_b1f.goal_name == "rock_tunnel_b1f_exit_ladder"

    # Lavender Town without Rainbow Badge -> advance west to Celadon
    sg_lav = nav.get_subgoal(4, 10, 10, badges=7)
    assert sg_lav.goal_name == "west_to_route8"

    # Celadon City -> cut gym tree
    sg_celadon = nav.get_subgoal(6, 20, 20, badges=7, bag=[RAMMap.ITEM_HM01_CUT])
    assert sg_celadon.goal_name == "cut_celadon_gym_tree"

    # Celadon Gym -> challenge Erika
    sg_erika = nav.get_subgoal(134, 4, 4, badges=7)
    assert sg_erika.goal_name == "challenge_erika"
    assert sg_erika.is_interaction is True

    # Celadon Rocket Hideout B4F -> claim lift key or challenge Giovanni
    sg_lift = nav.get_subgoal(202, 10, 10, badges=15, bag=[])
    assert sg_lift.goal_name == "claim_lift_key"

    sg_gio_hideout = nav.get_subgoal(202, 10, 10, badges=15, bag=[RAMMap.ITEM_LIFT_KEY])
    assert sg_gio_hideout.goal_name == "challenge_giovanni_hideout"
    assert sg_gio_hideout.is_interaction is True


def test_world_graph_badge5_pokemon_tower_and_fuchsia(nav):
    # Pokémon Tower 7F -> rescue Mr. Fuji
    sg_fuji = nav.get_subgoal(148, 10, 5, badges=15, bag=[RAMMap.ITEM_SILPH_SCOPE])
    assert sg_fuji.goal_name == "rescue_mr_fuji"
    assert sg_fuji.is_interaction is True

    # Route 16 -> wake Snorlax
    sg_snorlax = nav.get_subgoal(27, 26, 10, badges=15, bag=[RAMMap.ITEM_POKE_FLUTE])
    assert sg_snorlax.goal_name == "wake_snorlax_route16"
    assert sg_snorlax.is_interaction is True

    # Safari Zone West -> Gold Teeth
    sg_teeth = nav.get_subgoal(219, 19, 7, badges=15)
    assert sg_teeth.goal_name == "claim_gold_teeth"

    # Safari Zone Secret House -> Surf
    sg_surf = nav.get_subgoal(222, 3, 4, badges=15)
    assert sg_surf.goal_name == "claim_hm03_surf"

    # Fuchsia Gym -> challenge Koga
    sg_koga = nav.get_subgoal(157, 4, 11, badges=15, bag=[RAMMap.ITEM_HM03_SURF, RAMMap.ITEM_HM04_STRENGTH])
    assert sg_koga.goal_name == "challenge_koga"
    assert sg_koga.is_interaction is True


def test_world_graph_badge6_and_7_saffron_and_cinnabar(nav):
    # Silph Co. 5F -> Card Key
    sg_card = nav.get_subgoal(210, 10, 10, badges=31)
    assert sg_card.goal_name == "claim_card_key"

    # Silph Co. 11F -> Giovanni
    sg_gio_silph = nav.get_subgoal(235, 10, 10, badges=31, bag=[RAMMap.ITEM_CARD_KEY])
    assert sg_gio_silph.goal_name == "challenge_giovanni_silph"

    # Saffron Gym -> Sabrina
    sg_sabrina = nav.get_subgoal(178, 9, 7, badges=31)
    assert sg_sabrina.goal_name == "challenge_sabrina"

    # Pokémon Mansion B1F -> Secret Key
    sg_secret = nav.get_subgoal(216, 10, 10, badges=63)
    assert sg_secret.goal_name == "claim_secret_key"

    # Cinnabar Gym -> Blaine
    sg_blaine = nav.get_subgoal(166, 3, 2, badges=63, bag=[RAMMap.ITEM_SECRET_KEY])
    assert sg_blaine.goal_name == "challenge_blaine"


def test_world_graph_badge8_viridian_gym_giovanni(nav):
    # Viridian City with 7 badges -> route to Viridian Gym
    sg_vir_gym = nav.get_subgoal(1, 32, 8, badges=0x7F)
    assert sg_vir_gym.goal_name == "enter_viridian_gym"
    assert sg_vir_gym.is_warp is True

    # Viridian Gym -> Giovanni
    sg_gio_earth = nav.get_subgoal(45, 10, 4, badges=0x7F)
    assert sg_gio_earth.goal_name == "challenge_giovanni_earth"
    assert sg_gio_earth.is_interaction is True

    # After defeating Giovanni (badges=0xFF) -> exit gym
    sg_exit_earth = nav.get_subgoal(45, 10, 18, badges=0xFF)
    assert sg_exit_earth.goal_name == "exit_viridian_gym"


def test_world_graph_pokemon_league_and_indigo_plateau(nav):
    # Route 22 with 8 badges -> west to front gate
    sg_r22 = nav.get_subgoal(33, 10, 9, badges=0xFF)
    assert sg_r22.goal_name == "west_to_route23_gate"

    # Victory Road 1F -> stairs
    sg_vr1 = nav.get_subgoal(108, 5, 5, badges=0xFF)
    assert sg_vr1.goal_name == "victory_road_1f_stairs"

    # Indigo Plateau -> enter lobby
    sg_ind = nav.get_subgoal(9, 10, 6, badges=0xFF)
    assert sg_ind.goal_name == "enter_indigo_plateau_lobby"

    # Indigo Plateau Lobby -> enter Elite Four
    sg_lobby = nav.get_subgoal(174, 9, 5, badges=0xFF)
    assert sg_lobby.goal_name == "enter_elite_four_door"

    # Elite Four Gauntlet:
    # Lorelei
    sg_lor = nav.get_subgoal(245, 4, 3, badges=0xFF)
    assert sg_lor.goal_name == "challenge_lorelei"

    # Bruno
    sg_bru = nav.get_subgoal(246, 4, 3, badges=0xFF)
    assert sg_bru.goal_name == "challenge_bruno"

    # Agatha
    sg_aga = nav.get_subgoal(247, 4, 3, badges=0xFF)
    assert sg_aga.goal_name == "challenge_agatha"

    # Lance
    sg_lan = nav.get_subgoal(113, 6, 6, badges=0xFF)
    assert sg_lan.goal_name == "challenge_lance"

    # Champion Blue
    sg_blue = nav.get_subgoal(120, 4, 5, badges=0xFF)
    assert sg_blue.goal_name == "challenge_champion_blue"

    # Hall of Fame
    sg_hof = nav.get_subgoal(118, 2, 3, badges=0xFF)
    assert sg_hof.goal_name == "champion_hall_of_fame"
    assert sg_hof.is_interaction is True


def test_safari_zone_shortest_path_geodesics(nav):
    # Safari Center -> East pass
    sg_cen = nav.get_subgoal(220, 14, 25, badges=15)
    assert sg_cen.goal_name == "safari_center_to_east"

    # Safari East -> North pass
    sg_east = nav.get_subgoal(217, 0, 10, badges=15)
    assert sg_east.goal_name == "safari_east_to_north"

    # Safari North -> West pass
    sg_north = nav.get_subgoal(218, 8, 25, badges=15)
    assert sg_north.goal_name == "safari_north_to_west"

    # Safari West without Gold Teeth -> approach gold teeth
    sg_west = nav.get_subgoal(219, 29, 10, badges=15, bag=[])
    assert "gold_teeth" in sg_west.goal_name

    # Safari West with Gold Teeth -> approach secret house
    sg_west_surf = nav.get_subgoal(219, 29, 10, badges=15, bag=[RAMMap.ITEM_GOLD_TEETH])
    assert "secret_house" in sg_west_surf.goal_name


def test_macro_action_compiler_advanced_macros():
    from pokemon_rl.agent.macro_action_compiler import MacroActionCompiler

    # Teach HM sequence
    teach_seq = MacroActionCompiler.compile_teach_hm(hm_bag_slot=1, target_party_slot=0)
    assert teach_seq[0] == "start"
    assert "a" in teach_seq
    assert len(teach_seq) > 6

    # Forced switch
    switch_seq = MacroActionCompiler.compile_forced_battle_switch(target_slot=2, cur_party_slot=0)
    assert switch_seq == ["down", "down", "a", "a"]

    # Field moves
    assert MacroActionCompiler.compile_field_cut() == ["a", "a"]
    assert MacroActionCompiler.compile_field_surf() == ["a", "a"]

    # Option compilation
    assert MacroActionCompiler.compile_option("cut_tree", {}) == ["a", "a"]
    assert MacroActionCompiler.compile_option("surf_water", {}) == ["a", "a"]
    assert MacroActionCompiler.compile_option("forced_switch_to_slot_1", {"party_menu_item": 0}) == ["down", "a", "a"]

