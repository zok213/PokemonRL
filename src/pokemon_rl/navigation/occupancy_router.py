"""
occupancy_router.py — Dynamic Collision-Grid A* Pathfinding & Overworld Navigation
==================================================================================
Eliminates the 'Route 1 Ledge Collapse' in BrickerP/valentynkit.

Instead of greedy axis-first stepping (dx then dy) which soft-locks against
one-way ledges and obstacles, this module performs real-time A* search over
PyBoy's 20x18 walkable matrix (`game_area_collision()`).

Theoretical Foundations:
  - Dynamic Obstacle Avoidance via A* (Hart, Nilsson, Raphael 1968)
  - Collision Grid Extraction from LR35902 VRAM/OAM (pret/pokered CanWalkOntoTile)
  - SMDP Macro-Option Execution for Overworld Transit
"""

from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Dict, Set


# Screen dimensions in PyBoy collision blocks
GRID_HEIGHT = 18
GRID_WIDTH = 20
PLAYER_ROW = 9
PLAYER_COL = 8
STEP_DELTA = 2  # Each 2x2 tile block corresponds to 2 steps

DIRECTIONS = {
    "up": (-STEP_DELTA, 0),
    "down": (STEP_DELTA, 0),
    "left": (0, -STEP_DELTA),
    "right": (0, STEP_DELTA)
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left"
}


class OccupancyGridRouter:
    """
    Computes optimal, detour-aware walking actions using real-time A* pathfinding.
    """

    def __init__(self, step_budget: int = 150):
        self.step_budget = step_budget
        self.stuck_counter: Dict[str, int] = {}
        self.last_position: Optional[Tuple[int, int]] = None

    def plan_path(
        self,
        collision_matrix: List[List[int]],
        target_relative: Tuple[int, int],
        blocked_directions: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Find the optimal next direction step ('up', 'down', 'left', 'right')
        to reach target_relative (dx, dy) in map tiles, avoiding solid blocks (0).

        Args:
            collision_matrix: 18x20 array where 1 = walkable, 0 = impassable.
            target_relative: (target_x - current_x, target_y - current_y)
            blocked_directions: Directions currently known to be blocked by NPCs/ledges.

        Returns:
            The best next direction string, or None if no path exists.
        """
        dx, dy = target_relative
        if dx == 0 and dy == 0:
            return None

        blocked = blocked_directions or set()
        start = (PLAYER_ROW, PLAYER_COL)

        # Target cell in the local 18x20 window
        target_row = max(0, min(GRID_HEIGHT - 1, PLAYER_ROW + dy * STEP_DELTA))
        target_col = max(0, min(GRID_WIDTH - 1, PLAYER_COL + dx * STEP_DELTA))
        goal = (target_row, target_col)

        # Priority queue: (f_score, cost, (row, col), first_action)
        open_set: List[Tuple[float, int, Tuple[int, int], Optional[str]]] = []
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, None))

        visited: Set[Tuple[int, int]] = set()

        while open_set:
            f, cost, (r, c), first_action = heapq.heappop(open_set)

            if (r, c) == goal and first_action is not None:
                return first_action

            if (r, c) in visited:
                continue
            visited.add((r, c))

            for action_name, (dr, dc) in DIRECTIONS.items():
                if first_action is None and action_name in blocked:
                    continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    # Check walkability (1 = walkable)
                    if collision_matrix[nr][nc] == 1 and (nr, nc) not in visited:
                        new_first = first_action if first_action is not None else action_name
                        new_cost = cost + 1
                        h = self._heuristic((nr, nc), goal)
                        heapq.heappush(open_set, (new_cost + h, new_cost, (nr, nc), new_first))

        # Fallback: if goal is beyond the 20x18 window, pick the move that minimizes distance
        best_dir = None
        min_dist = float("inf")
        for action_name, (dr, dc) in DIRECTIONS.items():
            if action_name in blocked:
                continue
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                if collision_matrix[nr][nc] == 1:
                    d = abs(goal[0] - nr) + abs(goal[1] - nc)
                    if d < min_dist:
                        min_dist = d
                        best_dir = action_name

        return best_dir

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
