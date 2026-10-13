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
import json
from pathlib import Path
from collections import deque
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
    Grounded in authentic ROM collision grids via real-time BFS shortest-path waypoints.
    """

    def __init__(self):
        # Current active quest milestone
        # Milestones: "oaks_parcel", "deliver_parcel", "boulder_badge", "cascade_badge"
        self.milestone = "boulder_badge"
        self._map_grids: Dict[int, List[List[str]]] = {}
        self._load_map_grids()

    def _load_map_grids(self):
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        processed_dir = repo_root / "external" / "continual-harness" / "pokemon_red_env" / "data" / "processed_map"
        map_files = {
            14: "Route3.json",
            15: "Route4.json",
            57: "MtMoon1F.json",
            59: "MtMoon1F.json",
            58: "MtMoonB1F.json",
            60: "MtMoonB1F.json",
            61: "MtMoonB2F.json",
            3: "CeruleanCity.json",
            64: "CeruleanPokecenter.json",
            65: "CeruleanGym.json",
        }
        for mid, fn in map_files.items():
            fpath = processed_dir / fn
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        self._map_grids[mid] = [list(row) for row in json.load(f)["grid"]]
                except Exception:
                    pass

        # Mark static NPC obstacles in Mt Moon 1F so BFS automatically routes around them
        for mid in (57, 59):
            if mid in self._map_grids and len(self._map_grids[mid]) > 16:
                # Youngster 3 at (12, 16)
                if len(self._map_grids[mid][16]) > 12:
                    self._map_grids[mid][16][12] = '#'

        # Mark static NPC obstacles in Mt Moon B2F so BFS routes cleanly around them
        if 61 in self._map_grids:
            # Super Nerd Miguel at (12, 8)
            if len(self._map_grids[61]) > 8 and len(self._map_grids[61][8]) > 12:
                self._map_grids[61][8][12] = '#'
            # Rocket 1 at (11, 16)
            if len(self._map_grids[61]) > 16 and len(self._map_grids[61][16]) > 11:
                self._map_grids[61][16][11] = '#'

    def _bfs_next_waypoint(self, map_id: int, start: Tuple[int, int], goal: Tuple[int, int], lookahead: int = 3) -> Tuple[int, int]:
        if map_id not in self._map_grids or start == goal:
            return goal
        grid = self._map_grids[map_id]
        h, w = len(grid), len(grid[0])
        q = deque([start])
        visited: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        while q:
            c, r = q.popleft()
            if (c, r) == goal:
                break
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < w and 0 <= nr < h and (nc, nr) not in visited:
                    t = grid[nr][nc]
                    if t not in ('#', '!', '~', 'W', 'C', 'P', '=', 'B'):
                        # Authentic Gen 1 ledge jumping constraints (directional one-way hopping)
                        code = ord(t)
                        if code == 0x2193 and dr != 1:  # Down ledge: only stepping down
                            continue
                        if code == 0x2192 and dc != 1:  # Right ledge: only stepping right
                            continue
                        if code == 0x2190 and dc != -1: # Left ledge: only stepping left
                            continue
                        if code == 0x2191 and dr != -1: # Up ledge: only stepping up
                            continue
                        visited[(nc, nr)] = (c, r)
                        q.append((nc, nr))
        if goal not in visited:
            return goal
        curr = goal
        path: List[Tuple[int, int]] = []
        while curr is not None:
            path.append(curr)
            curr = visited[curr]
        path.reverse()
        idx = min(lookahead, len(path) - 1)
        return path[idx]

    def set_milestone(self, milestone: str):
        self.milestone = milestone

    def get_subgoal(
        self,
        map_id: int,
        x: int,
        y: int,
        badges: int = 0,
        has_parcel: bool = True,
        hp: int = 20,
        max_hp: int = 20,
        has_fossil: bool = False
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
            # Keep Red on eastern main avenue (x=21) when advancing north to prevent drifting west onto Route 22
            if y > 8:
                return MapSubgoal((21, 8), "advance_viridian_avenue", "Advance north along Viridian main avenue")
            elif y > 2:
                if x > 18:
                    return MapSubgoal((18, 2), "detour_signpost_west", "Detour west to bypass Viridian north signpost")
                return MapSubgoal((18, 1), "advance_city_north", "Advance north along open avenue")
            elif y == 2 and x >= 19:
                return MapSubgoal((18, 2), "detour_signpost_west", "Detour west around signpost")
            elif y == 1:
                return MapSubgoal((18, 0), "advance_exit", "Advance north towards Route 2 entrance")
            else:
                return MapSubgoal((18, -1), "enter_route2", "Step north through border into Route 2", is_warp=True)

        # 4b. Route 22 (Map 33) — If agent strays west from Viridian City, guide back east
        elif map_id == 33:
            return MapSubgoal((39, 9), "east_to_viridian", "Return east towards Viridian City", is_warp=True)

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
            # Complete 11-waypoint continuous inverted-U labyrinth path to North Gate
            if y > 43 and x <= 18:
                return MapSubgoal((17, 43), "forest_corridor_entry", "Advance north through entry corridor")
            elif x <= 18 and y > 40:
                return MapSubgoal((18, 40), "forest_clear_corner", "Advance north to clear corner")
            elif 39 <= y <= 42 and x < 25:
                return MapSubgoal((25, 41), "forest_eastern_crossway", "Cut east along bottom crossway towards highway")
            elif x >= 23 and y > 9:
                return MapSubgoal((25, 9), "forest_eastern_highway", "Advance north along eastern forest highway")
            elif y <= 11 and x > 16:
                return MapSubgoal((16, 9), "forest_northern_crossway", "Cut west across northern crossway")
            elif 15 <= x <= 18 and y < 16:
                return MapSubgoal((16, 16), "forest_central_channel", "Navigate south down central channel")
            elif x > 13 and 14 <= y <= 18:
                return MapSubgoal((13, 16), "forest_middle_crossway", "Cut west along middle crossway")
            elif 11 <= x <= 14 and y > 3:
                return MapSubgoal((13, 3), "forest_western_channel", "Navigate north through western channel")
            elif x > 6 and y <= 5:
                return MapSubgoal((6, 3), "forest_northern_ridge", "Cut west across northern ridge")
            elif 5 <= x <= 8 and y < 22:
                return MapSubgoal((6, 22), "forest_western_loop", "Follow western loop south towards exit alley")
            elif x > 2 and y >= 20:
                return MapSubgoal((1, 22), "forest_exit_alley", "Turn west into exit alleyway")
            elif y > 0:
                return MapSubgoal((1, 0), "forest_exit_door", "Advance north towards North Gatehouse doorway")
            else:
                return MapSubgoal((1, -1), "forest_warp_north_gate", "Step through doorway into North Gate", is_warp=True)

        # 8. Viridian Forest North Gatehouse (Map 47)
        elif map_id == 47:
            if y > 0:
                return MapSubgoal((4, 0), "north_gate_corridor", "Walk north through gatehouse towards Pewter City")
            return MapSubgoal((4, -1), "exit_to_route2_north", "Step north into Route 2 North / Pewter City", is_warp=True)

        # 9. Pewter City (Map 2)
        elif map_id == 2:
            if (badges & 1) == 0:
                # If HP not full, heal at Pewter Pokecenter at (13, 25) to heal party and set respawn checkpoint
                if hp < max_hp:
                    if (x, y) == (13, 26):
                        return MapSubgoal((13, 25), "enter_pewter_pokecenter", "Step north through door into Pokécenter", is_warp=True)
                    elif y < 24:
                        if x > 14:
                            return MapSubgoal((13, y), "channel_west_to_pokecenter", "Cut west towards Pokécenter avenue")
                        return MapSubgoal((13, 26), "approach_pokecenter_south", "Advance south towards Pokécenter entrance")
                    else:
                        return MapSubgoal((13, 26), "approach_pokecenter", "Approach Pewter Pokecenter entrance at (13, 26)")
                # Target: Pewter Gym at (16, 17), approached via sidewalk at (16, 18)
                if (x, y) == (16, 18) or (x, y) == (16, 17):
                    return MapSubgoal((16, 17), "enter_pewter_gym", "Enter Pewter Gym to challenge Gym Leader Brock", is_warp=True)
                elif y < 14 and x > 18:
                    # In north-eastern sector: channel west to main north-south avenue (x=18) to bypass hedges
                    return MapSubgoal((18, y), "channel_west_to_avenue", "Cut west towards main avenue")
                else:
                    return MapSubgoal((16, 18), "advance_pewter_gym_entrance", "Advance towards Pewter Gym entrance sidewalk at (16, 18)")
            else:
                # Boulder Badge acquired! Route past gym enclosure to Route 3 via northern highway
                # 1. Anywhere in western/central sector south of Y=8: cut west to column 7
                if x < 31 and y > 8:
                    if x > 7:
                        return MapSubgoal((7, y), "exit_gym_enclosure_west", "Walk west out of enclosure")
                    return MapSubgoal((7, 8), "pewter_avenue_north", "Walk north along western avenue to northern crossway")
                # 2. East along clear northern crossway to column 32
                elif y <= 10 and x < 32:
                    return MapSubgoal((32, 8), "pewter_crossway_east", "Walk east along clear northern crossway")
                # 3. South down column 32 to Route 3 pass at Y=17
                elif x >= 31 and y < 17:
                    return MapSubgoal((32, 17), "pewter_exit_channel_south", "Walk south down eastern channel towards Route 3 pass")
                # 4. East into Route 3!
                return MapSubgoal((40, 17), "east_to_route3", "Advance east onto Route 3 towards Mt. Moon")

        # 10. Pewter Gym (Map 54)
        elif map_id == 54:
            if (badges & 1) == 0:
                if y > 3:
                    return MapSubgoal((4, 3), "challenge_brock", "Challenge Brock for Boulder Badge")
                elif y == 3:
                    return MapSubgoal((4, 2), "challenge_brock", "Step up to Gym Leader Brock to battle for Boulder Badge")
                else:
                    return MapSubgoal((4, 2), "interact_brock", "Speak with Brock to initiate battle for Boulder Badge")
            else:
                # Brock defeated, exit south through Gym doorway
                if y < 13:
                    return MapSubgoal((4, 13), "exit_pewter_gym", "Walk south towards Pewter Gym exit")
                return MapSubgoal((4, 14), "step_outside_gym", "Step outside back into Pewter City", is_warp=True)

        # Pewter Nidoran House (Map 55)
        elif map_id == 55:
            if y < 7:
                return MapSubgoal((2, 7), "exit_nidoran_house", "Walk south towards exit")
            return MapSubgoal((2, 8), "step_outside_nidoran_house", "Step outside back into Pewter City", is_warp=True)

        # Pewter Poké Mart (Map 56)
        elif map_id == 56:
            if y < 7:
                return MapSubgoal((3, 7), "exit_pewter_mart", "Walk south towards Poké Mart exit")
            return MapSubgoal((3, 8), "step_outside_pewter_mart", "Step outside back into Pewter City", is_warp=True)

        # Pewter Speech House (Map 57)
        elif map_id == 57:
            if y < 7:
                return MapSubgoal((2, 7), "exit_speech_house", "Walk south towards exit")
            return MapSubgoal((2, 8), "step_outside_speech_house", "Step outside back into Pewter City", is_warp=True)

        # Museum 2F (Map 53)
        elif map_id == 53:
            return MapSubgoal((7, 7), "descend_stairs", "Descend stairs to Museum 1F", is_warp=True)

        # Museum 1F (Map 52)
        elif map_id == 52:
            if x <= 13:
                return MapSubgoal((10, 8), "exit_museum", "Step outside back to Pewter City", is_warp=True)
            else:
                return MapSubgoal((16, 8), "exit_museum", "Step outside back to Pewter City", is_warp=True)

        # Pokécenters: Viridian (41), Pewter (58), Mt. Moon (68), Cerulean (64)
        elif map_id in (41, 58, 64, 68):
            if hp < max_hp or hp == 0:
                if y > 3:
                    return MapSubgoal((3, 3), "approach_nurse_joy", "Approach Nurse Joy counter to heal")
                return MapSubgoal((3, 2), "interact_nurse_joy", "Speak with Nurse Joy to heal party")
            else:
                if y < 7:
                    return MapSubgoal((3, 7), "exit_pokecenter", "Walk south towards Pokécenter exit")
                return MapSubgoal((3, 7), "step_outside_pokecenter", "Step through door back outside", is_warp=True)

        # 11. Route 3 (Map 14)
        elif map_id == 14:
            # Step north through mountain pass into Route 4
            if x >= 58 and y <= 0:
                return MapSubgoal((59, -1), "enter_route4", "Step north through mountain pass onto Route 4", is_warp=True)
            wp = self._bfs_next_waypoint(14, (x, y), (59, 0), lookahead=3)
            return MapSubgoal(wp, "route3_advance_pass", f"Advance east along Route 3 mountain pass towards Route 4 via {wp}")

        # 12. Route 4 (Map 15)
        elif map_id == 15:
            # West side of Mt. Moon: entrance cave is at (18, 5)
            if x < 22:
                # If party damaged on Route 3, heal at Mt. Moon Pokécenter at (11, 5) to set checkpoint
                if hp < max_hp and x < 16:
                    if (x, y) == (11, 6) or (x, y) == (11, 5):
                        return MapSubgoal((11, 5), "enter_mt_moon_pokecenter", "Enter Mt. Moon Pokécenter to heal", is_warp=True)
                    wp = self._bfs_next_waypoint(15, (x, y), (11, 6), lookahead=2)
                    return MapSubgoal(wp, "approach_mt_moon_pokecenter", "Approach Mt. Moon Pokécenter entrance")
                if (x, y) == (18, 6) or (x, y) == (18, 5):
                    return MapSubgoal((18, 5), "enter_mt_moon", "Enter Mt. Moon cave", is_warp=True)
                wp = self._bfs_next_waypoint(15, (x, y), (18, 5), lookahead=3)
                return MapSubgoal(wp, "approach_mt_moon", "Advance towards Mt. Moon entrance")
            # East side of Mt. Moon (after exiting at 24, 5): highway to Cerulean City
            else:
                if x >= 89:
                    return MapSubgoal((90, 10), "enter_cerulean", "Step east into Cerulean City", is_warp=True)
                wp = self._bfs_next_waypoint(15, (x, y), (89, 10), lookahead=1)
                return MapSubgoal(wp, "route4_east_highway", f"Advance east across Route 4 towards Cerulean City via {wp}")

        # 13. Mt. Moon 1F (Map 59 canonical, 57 legacy)
        elif map_id in (57, 59):
            if (x, y) == (14, 35):
                return MapSubgoal((14, 25), "mt_moon_1f_corridor_north", "Advance north along entry corridor")
            if (x, y) == (5, 5):
                return MapSubgoal((5, 5), "descend_ladder_1f", "Descend ladder to B1F", is_warp=True)
            wp = self._bfs_next_waypoint(59, (x, y), (5, 5), lookahead=1)
            return MapSubgoal(wp, "mt_moon_1f_ladder_north", "Advance through Mt. Moon 1F towards ladder", is_warp=(wp == (5, 5)))

        # 14. Mt. Moon B1F (Map 60 canonical, 58 legacy)
        elif map_id == 60 or (map_id == 58 and y > 8):
            # Northern tunnel: ladder from B2F is at (23, 3), exit doorway is right at (27, 3)
            if x > 20 and y < 8:
                if x >= 27:
                    return MapSubgoal((27, 3), "exit_to_route4", "Step through cave exit into Route 4 East", is_warp=True)
                wp = self._bfs_next_waypoint(60, (x, y), (27, 3), lookahead=1)
                return MapSubgoal(wp, "b1f_to_exit_door", "Advance east to cave exit doorway", is_warp=(wp == (27, 3)))
            # Southern tunnel from 1F (5, 5): navigate to B2F ladder at (21, 17)
            else:
                if (x, y) == (25, 15):
                    return MapSubgoal((25, 27), "b1f_corridor_south", "Advance south along tunnel corridor")
                if (x, y) == (21, 17):
                    return MapSubgoal((21, 17), "b1f_ladder_to_b2f", "Descend ladder into B2F cavern", is_warp=True)
                wp = self._bfs_next_waypoint(60, (x, y), (21, 17), lookahead=1)
                return MapSubgoal(wp, "b1f_corridor_south", "Advance south along tunnel corridor towards B2F ladder", is_warp=(wp == (21, 17)))

        # 15. Mt. Moon B2F (Map 61 canonical)
        elif map_id == 61:
            if (x, y) == (15, 27):
                return MapSubgoal((29, 27), "b2f_bottom_cavern_east", "Advance east along southern cavern")
            if not has_fossil and (x, y) == (11, 6):
                return MapSubgoal((12, 6), "claim_dome_fossil", "Claim Dome Fossil with Button A")
            if (x, y) == (5, 7):
                return MapSubgoal((5, 7), "ascend_to_b1f_exit", "Ascend ladder towards cave exit", is_warp=True)
            wp = self._bfs_next_waypoint(61, (x, y), (5, 7), lookahead=1)
            return MapSubgoal(wp, "b2f_approach_exit_ladder", "Advance through B2F cavern towards exit ladder", is_warp=(wp == (5, 7)))

        # 16. Cerulean City (Map 3)
        elif map_id == 3:
            # If lead mon or party needs healing before Cascade Badge, heal at Cerulean Pokécenter at (19, 17)
            if (badges & 2) == 0:  # Cascade Badge not yet acquired
                if hp < max_hp or hp == 0:
                    if (x, y) in ((19, 18), (19, 17)):
                        return MapSubgoal((19, 17), "enter_cerulean_pokecenter", "Step north through door into Cerulean Pokécenter", is_warp=True)
                    else:
                        wp = self._bfs_next_waypoint(3, (x, y), (19, 18), lookahead=2)
                        return MapSubgoal(wp, "approach_pokecenter", f"Advance towards Pokécenter entrance via {wp}")
                else:
                    # Party is fully healed! Target: Cerulean Gym at (30, 19)
                    if (x, y) in ((30, 20), (30, 19)):
                        return MapSubgoal((30, 19), "enter_cerulean_gym", "Enter Cerulean Gym to challenge Gym Leader Misty", is_warp=True)
                    else:
                        wp = self._bfs_next_waypoint(3, (x, y), (30, 20), lookahead=2)
                        return MapSubgoal(wp, "approach_cerulean_gym", f"Advance towards Cerulean Gym entrance via {wp}")
            else:
                # Cascade Badge acquired!
                return MapSubgoal((x, y), "cascade_badge_achieved", "Cascade Badge Acquired!")

        # 18. Cerulean Gym (Map 65)
        elif map_id == 65:
            if (badges & 2) == 0:
                if (x, y) == (4, 3):
                    return MapSubgoal((4, 2), "challenge_misty", "Step up to Gym Leader Misty to battle for Cascade Badge")
                elif (x, y) == (5, 2):
                    return MapSubgoal((4, 2), "challenge_misty_from_right", "Face west towards Gym Leader Misty to battle for Cascade Badge")
                else:
                    wp = self._bfs_next_waypoint(65, (x, y), (4, 3), lookahead=1)
                    return MapSubgoal(wp, "approach_misty", f"Advance through Cerulean Gym towards Misty via {wp}")
            else:
                # Misty defeated, exit south through Gym doorway
                if (x, y) in ((4, 13), (5, 13)):
                    return MapSubgoal((4, 14), "step_outside_gym", "Step outside back into Cerulean City", is_warp=True)
                wp = self._bfs_next_waypoint(65, (x, y), (4, 13), lookahead=1)
                return MapSubgoal(wp, "exit_cerulean_gym", f"Walk south towards Cerulean Gym exit via {wp}")

        # 17. Viridian Poké Mart (Map 42)
        elif map_id == 42:
            if y >= 7:
                return MapSubgoal((3, 8), "step_outside", "Step through door back to Viridian City", is_warp=True)
            return MapSubgoal((3, 5), "approach_counter", "Approach front counter to receive Oak's Parcel")

        # Default fallback
        return MapSubgoal((x, max(0, y - 1)), "explore_ahead", "Explore area")
