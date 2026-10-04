"""
test_router_and_compiler.py
===========================
Unit tests for OccupancyGridRouter (dynamic A* pathfinding) and MacroActionCompiler.
Verifies obstacle detour around ledges and hardware-accurate menu button sequences.
"""

import pytest
from pokemon_rl.navigation.occupancy_router import OccupancyGridRouter
from pokemon_rl.agent.macro_action_compiler import MacroActionCompiler


def test_occupancy_grid_router_direct_path():
    router = OccupancyGridRouter()
    # 18x20 open walkable grid (all 1s)
    grid = [[1 for _ in range(20)] for _ in range(18)]

    # Target is 2 tiles North (dy = -2, dx = 0)
    action = router.plan_path(grid, (0, -2))
    assert action == "up"

    # Target is 2 tiles East (dx = 2, dy = 0)
    action = router.plan_path(grid, (2, 0))
    assert action == "right"


def test_occupancy_grid_router_obstacle_detour():
    """Verify that A* finds a detour around a solid wall/ledge (0s) directly in front."""
    router = OccupancyGridRouter()
    grid = [[1 for _ in range(20)] for _ in range(18)]

    # Place a solid wall directly North of player (at row 7, cols 7, 8, 9)
    # Player sits at (row=9, col=8)
    grid[7][7] = 0
    grid[7][8] = 0  # Impassable
    grid[7][9] = 0

    # Target is North (dy = -3)
    action = router.plan_path(grid, (0, -3))
    # Direct "up" is blocked by the wall at row 7, so A* must detour "left" or "right"
    assert action in ("left", "right")


def test_macro_action_compiler_battle_moves():
    # Cursor starts at FIGHT (0) with move_idx = 0
    wram = {"battle_menu_item": 0, "move_list_index": 0}
    moves = ["Scratch", "Growl", "Ember", "Leer"]

    # Select Ember (slot 2)
    buttons = MacroActionCompiler.compile_option(
        "use_move_ember",
        wram,
        move_names=moves
    )
    # Should open fight with 'a', step down twice, and confirm with 'a'
    assert buttons == ["a", "down", "down", "a"]


def test_macro_action_compiler_cursor_memory_offset():
    """Verify that button sequences compensate for remembered cursor positions."""
    # Cursor was left on PKMN (2: col 1, row 0)
    wram = {"battle_menu_item": 2, "move_list_index": 1}
    moves = ["Scratch", "Growl"]

    # Select Scratch (slot 0)
    buttons = MacroActionCompiler.compile_option(
        "use_move_scratch",
        wram,
        move_names=moves
    )
    # Must walk from PKMN (col 1, row 0) to FIGHT (col 0, row 0) -> ["left"], then 'a', then step up from slot 1 -> ["up"], then 'a'
    assert buttons == ["left", "a", "up", "a"]


def test_macro_action_compiler_run_away():
    wram = {"battle_menu_item": 0}  # FIGHT
    # RUN is at (col 1, row 1), from FIGHT (col 0, row 0) -> ['right', 'down', 'a']
    buttons = MacroActionCompiler.compile_option("run_away", wram)
    assert buttons == ["right", "down", "a"]
