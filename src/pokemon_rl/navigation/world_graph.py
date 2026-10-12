"""
world_graph.py — Topological World Graph Navigator for Pokémon Red (Gen 1)
=============================================================================
Provides a unified topological roadmap across the Kanto region.
Decouples high-level quest progression from low-level obstacle avoidance:
  - Directed map graph G = (V, E) connecting towns, routes, gates, and dungeons
  - Dynamic subgoal generation based on current WRAM map ID, coordinates, and quest state
  - Automatic obstacle and waypoint sequencing for Route 1, Viridian, Route 2, and Viridian Forest
  - Eliminates hardcoded coordinate logic across individual server handlers
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MapSubgoal:
    target_coord: Tuple[int, int]
    goal_name: str
    description: str
    is_warp: bool = False


class WorldGraphNavigator:
    """
    Hierarchical topological roadmap and goal generator across all Generation 1 maps.
    """

    def __init__(self):
        # Current active quest milestone
        # Milestones: "oaks_parcel", "deliver_parcel", "boulder_badge", "cascade_badge"
        self.milestone = "boulder_badge"

    def set_milestone(self, milestone: str):
        self.milestone = milestone

    def get_subgoal(
        self,
        map_id: int,
        x: int,
        y: int,
        badges: int = 0,
        has_parcel: bool = True
    ) -> MapSubgoal:
        """
        Dynamically determine the optimal next waypoint/subgoal given current map and coordinates.
        """
        # 1. Oak's Laboratory (Map 40)
        if map_id == 40:
            if y < 11:
                return MapSubgoal((4, 11), "exit_lab", "Walk south towards laboratory exit", is_warp=False)
            return MapSubgoal((4, 12), "step_outside", "Step through door into Pallet Town", is_warp=True)

        # 1b. Blue's House / Daisy's House (Map 39)
        elif map_id == 39:
            if y < 7:
                return MapSubgoal((2, 7), "exit_blues_house", "Walk south towards exit of Blue's house", is_warp=False)
            return MapSubgoal((2, 8), "step_outside_blues_house", "Step through door into Pallet Town", is_warp=True)

        # 1c. Red's House 1F (Map 37) & 2F (Map 38)
        elif map_id == 37:
            if y < 7:
                return MapSubgoal((2, 7), "exit_reds_house", "Walk south towards exit of Red's house", is_warp=False)
            return MapSubgoal((2, 8), "step_outside_reds_house", "Step through door into Pallet Town", is_warp=True)
        elif map_id == 38:
            return MapSubgoal((7, 1), "descend_stairs", "Walk to stairs to descend to 1F", is_warp=True)

        # 1d. Viridian City Interiors (Map 41, 43, 44)
        elif map_id == 41:  # Viridian Pokecenter
            if y < 7:
                return MapSubgoal((3, 7), "exit_pokecenter", "Walk south towards Pokecenter exit", is_warp=False)
            return MapSubgoal((3, 8), "step_outside_pokecenter", "Step through door into Viridian City", is_warp=True)
        elif map_id in (43, 44):  # Viridian School / Houses
            if y < 7:
                return MapSubgoal((2, 7), "exit_house", "Walk south towards house exit", is_warp=False)
            return MapSubgoal((2, 8), "step_outside_house", "Step through door into Viridian City", is_warp=True)

        # 2. Pallet Town (Map 0)
        elif map_id == 0:
            if x > 11:
                # Align west onto main central avenue to bypass Blue's House doorway (13, 5) and northeast trees
                target_y = y if y >= 6 else 6
                return MapSubgoal((10, target_y), "align_west_to_road", "Align west onto central Pallet Town avenue")
            elif x < 9:
                target_y = y if y >= 6 else 6
                return MapSubgoal((10, target_y), "align_east_to_road", "Align east onto central Pallet Town avenue")
            elif y > 1:
                return MapSubgoal((10, 0), "north_to_route1", "Advance north along Pallet Town road to Route 1")
            return MapSubgoal((10, -1), "enter_route1", "Step north through border into Route 1", is_warp=True)

        # 3. Route 1 (Map 12)
        elif map_id == 12:
            if y > 28:
                if x > 8:
                    return MapSubgoal((8, 28), "bypass_fence_west", "Bypass western fence on Route 1")
                return MapSubgoal((8, 24), "advance_corridor", "Advance north through western corridor")
            elif 24 < y <= 28:
                return MapSubgoal((8, 24), "advance_corridor", "Advance north along roadside")
            elif y == 24 and x < 12:
                return MapSubgoal((12, 24), "bypass_ledge_east", "Bypass southern ledge towards eastern highway")
            elif 20 < y <= 24 and x >= 12:
                return MapSubgoal((12, 20), "advance_highway", "Advance north along eastern highway")
            elif y == 20 and x > 9:
                return MapSubgoal((9, 20), "bypass_fence_west", "Bypass northern fence towards central clearing")
            elif 18 < y <= 20 and x <= 9:
                return MapSubgoal((9, 18), "advance_clearing", "Advance north along central clearing")
            elif y == 18 and x < 14:
                return MapSubgoal((14, 18), "bypass_highway_east", "Cut east to reach open highway to Viridian")
            elif 2 < y <= 18 and x >= 14:
                return MapSubgoal((14, 2), "advance_highway", "Advance north along Route 1 highway")
            elif y <= 2:
                if x > 10:
                    return MapSubgoal((10, 2), "align_gate_west", "Align west with Viridian City entrance gate")
                return MapSubgoal((10, -1), "enter_viridian", "Step north through gate into Viridian City", is_warp=True)
            return MapSubgoal((10, -1), "advance_north", "Advance north along Route 1")

        # 4. Viridian City (Map 1)
        elif map_id == 1:
            # Check if parcel quest is active and not yet delivered
            if not has_parcel and self.milestone == "oaks_parcel":
                # Route to Viridian Poke Mart at (29, 19)
                if x < 29 or y > 19:
                    return MapSubgoal((29, 20), "approach_mart", "Navigate east towards Viridian Poké Mart")
                return MapSubgoal((29, 19), "enter_mart", "Enter Viridian Poké Mart to receive Parcel", is_warp=True)

            # Standard overworld progression towards Route 2 / Viridian Forest:
            # Wooden signpost at (19, 1) blocks direct north -> detour west via (18, 2)
            if y > 2:
                if x > 18:
                    return MapSubgoal((18, 2), "detour_signpost_west", "Detour west to bypass Viridian north signpost")
                return MapSubgoal((18, 1), "advance_city_north", "Advance north along open avenue")
            elif y == 2 and x >= 19:
                return MapSubgoal((18, 2), "detour_signpost_west", "Detour west around signpost")
            elif y == 1:
                return MapSubgoal((18, 0), "advance_exit", "Advance north towards Route 2 entrance")
            else:
                return MapSubgoal((18, -1), "enter_route2", "Step north through border into Route 2", is_warp=True)

        # 5. Route 2 South (Map 13)
        elif map_id == 13:
            # Route 2 jumping ledge blocks x=0..7 at y=48.
            # Structured waypoints: (8, 71) -> (3, 57) -> (3, 48) -> (9, 48) -> (9, 44) -> (3, 44) -> (3, 43)
            if y > 57:
                return MapSubgoal((3, 57), "bypass_south_fence", "Walk west to enter Route 2 western corridor")
            elif 48 < y <= 57:
                return MapSubgoal((3, 48), "advance_western_corridor", "Advance north towards jumping ledge")
            elif y == 48 and x < 9:
                return MapSubgoal((9, 48), "bypass_ledge_east", "Cut east along ledge to reach the opening at X=9")
            elif 44 < y <= 48 and x >= 9:
                return MapSubgoal((9, 44), "advance_past_ledge", "Advance north past the ledge opening")
            elif y == 44 and x > 3:
                return MapSubgoal((3, 44), "cut_west_to_gate", "Cut west towards the Viridian Forest South Gate")
            elif y >= 43 and x <= 4:
                return MapSubgoal((3, 43), "enter_south_gate", "Step into Viridian Forest South Gatehouse", is_warp=True)
            elif y <= 33: # Route 2 North (after exiting North Gate)
                return MapSubgoal((3, 0), "north_to_pewter", "Advance north along Route 2 into Pewter City")
            else:
                return MapSubgoal((3, 43), "approach_gate", "Advance towards Viridian Forest Gate")

        # 6. Viridian Forest South Gatehouse (Map 50)
        elif map_id == 50:
            if y > 0:
                return MapSubgoal((4, 0), "advance_gate_corridor", "Walk north through gatehouse towards Forest")
            return MapSubgoal((4, -1), "enter_forest", "Step through doorway into Viridian Forest", is_warp=True)

        # 7. Viridian Forest Labyrinth (Map 51)
        elif map_id == 51:
            # Complete 15-waypoint inverted-U labyrinth path towards Pewter City North Gate
            if y >= 44:
                return MapSubgoal((17, 43), "forest_corridor_entry", "Step north into forest path")
            elif x <= 17 and y == 43:
                return MapSubgoal((18, 43), "forest_eastern_cut", "Turn east to bypass tree barrier")
            elif x == 18 and y > 41:
                return MapSubgoal((18, 41), "forest_clear_corner", "Advance north to clear corner")
            elif y >= 41 and x < 25:
                return MapSubgoal((25, 41), "forest_eastern_crossway", "Advance east along crossway towards highway")
            elif x >= 24 and y > 9:
                return MapSubgoal((25, 9), "forest_eastern_highway", "Advance north along eastern forest highway")
            elif y <= 9 and x > 16:
                return MapSubgoal((16, 9), "forest_northern_crossway", "Cut west across northern highway")
            elif x == 16 and y < 16:
                return MapSubgoal((16, 16), "forest_central_channel", "Navigate south down central channel")
            elif y == 16 and x > 13:
                return MapSubgoal((13, 16), "forest_middle_crossway", "Cut west along middle crossway")
            elif x == 13 and y > 3:
                return MapSubgoal((13, 3), "forest_western_channel", "Navigate north through western channel")
            elif y == 3 and x > 6:
                return MapSubgoal((6, 3), "forest_northern_ridge", "Cut west across northern ridge")
            elif x == 6 and y < 22:
                return MapSubgoal((6, 22), "forest_western_loop", "Follow western loop south towards exit alley")
            elif y >= 20 and x > 1:
                return MapSubgoal((1, 22), "forest_exit_alley", "Turn west into the exit alleyway")
            elif y <= 0:
                return MapSubgoal((1, -1), "forest_warp_north_gate", "Step north through doorway into North Gate", is_warp=True)
            else:
                return MapSubgoal((1, 0), "forest_exit_door", "Advance north towards North Gate doorway")

        # 8. Viridian Forest North Gatehouse (Map 47)
        elif map_id == 47:
            if y > 0:
                return MapSubgoal((4, 0), "north_gate_corridor", "Walk north through gatehouse towards Pewter City")
            return MapSubgoal((4, -1), "exit_to_route2_north", "Step north into Route 2 North / Pewter City", is_warp=True)

        # 9. Pewter City (Map 2)
        elif map_id == 2:
            if badges == 0:
                # Target: Pewter Gym at (16, 17)
                if y > 18:
                    return MapSubgoal((16, 18), "advance_pewter_center", "Advance north through Pewter City streets")
                return MapSubgoal((16, 17), "enter_pewter_gym", "Enter Pewter Gym to challenge Gym Leader Brock", is_warp=True)
            else:
                # Boulder Badge acquired! Target: Route 3 East at (40, 17)
                return MapSubgoal((40, 17), "east_to_route3", "Advance east onto Route 3 towards Mt. Moon")

        # 10. Pewter Gym (Map 55)
        elif map_id == 55:
            # Gym Leader Brock is at (4, 3)
            return MapSubgoal((4, 3), "challenge_brock", "Approach Gym Leader Brock to battle for Boulder Badge")

        # 11. Viridian Poké Mart (Map 42)
        elif map_id == 42:
            if y >= 7:
                return MapSubgoal((3, 7), "step_outside", "Step through door back to Viridian City", is_warp=True)
            return MapSubgoal((3, 5), "approach_counter", "Approach front counter to receive Oak's Parcel")

        # Default fallback
        return MapSubgoal((x, max(0, y - 1)), "explore_ahead", "Explore area")
