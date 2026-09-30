"""
interactive.constants — Hardware Enums, Memory Maps & Constants
================================================================
Canonical definitions for Game Boy LR35902 emulation, joypad events,
Gen 1 map IDs, and battle move lookup tables.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
from pyboy.utils import WindowEvent
from pokemon_rl.env.wram_map import Action

# Action Display Names
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

# Whidden 439M Trained Checkpoint Actions (Canonical PokemonRedExperiments ordering)
WHIDDEN_ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START", "PASS"]

# Game Boy Action to PyBoy WindowEvent Mapping
ACTION_TO_PYBOY_EVENTS: Dict[int, Tuple[int, int]] = {
    Action.UP: (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    Action.DOWN: (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    Action.LEFT: (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    Action.RIGHT: (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    Action.A: (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    Action.B: (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    Action.START: (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    Action.SELECT: (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
}

# Whidden 439M Action to PyBoy WindowEvent Mapping
WHIDDEN_ACTION_TO_PYBOY_EVENTS: Dict[int, Tuple[int, int]] = {
    0: (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    1: (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    2: (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    3: (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    4: (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    5: (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    6: (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    7: (WindowEvent.PASS, WindowEvent.PASS),
}

# Canonical Map Names for Game Boy Generation 1
MAP_NAMES: Dict[int, str] = {
    0: "Pallet Town",
    1: "Viridian City",
    2: "Pewter City",
    3: "Cerulean City",
    4: "Lavender Town",
    5: "Vermilion City",
    6: "Celadon City",
    7: "Fuchsia City",
    8: "Cinnabar Island",
    9: "Indigo Plateau",
    10: "Saffron City",
    12: "Route 1",
    13: "Route 2",
    14: "Route 3",
    15: "Route 4",
    37: "Red's House 1F",
    38: "Red's House 2F",
    39: "Blues House",
    40: "Oak's Lab",
    41: "Viridian Pokecenter",
    42: "Viridian Pokemart",
    43: "Viridian School",
    44: "Viridian House",
    45: "Viridian Gym",
    47: "Route 22 Gate",
    51: "Viridian Forest",
    54: "Pewter Pokecenter",
    55: "Pewter Gym",
    57: "Mt Moon 1F",
    58: "Mt Moon B1F",
    59: "Mt Moon B2F",
    60: "Cerulean Pokecenter",
    61: "Cerulean Gym",
}

# Generation 1 Canonical Move Database for Tactical Battle Controller
GEN1_MOVE_DATABASE: Dict[int, Dict[str, Any]] = {
    1: {"name": "POUND", "type": "NORMAL", "power": 40, "accuracy": 1.0},
    10: {"name": "SCRATCH", "type": "NORMAL", "power": 40, "accuracy": 1.0},
    22: {"name": "VINE WHIP", "type": "GRASS", "power": 35, "accuracy": 1.0},
    33: {"name": "TACKLE", "type": "NORMAL", "power": 40, "accuracy": 1.0},
    39: {"name": "TAIL WHIP", "type": "NORMAL", "power": 0, "accuracy": 1.0},
    45: {"name": "GROWL", "type": "NORMAL", "power": 0, "accuracy": 1.0},
    52: {"name": "EMBER", "type": "FIRE", "power": 40, "accuracy": 1.0},
    55: {"name": "WATER GUN", "type": "WATER", "power": 40, "accuracy": 1.0},
    84: {"name": "THUNDERSHOCK", "type": "ELECTRIC", "power": 40, "accuracy": 1.0},
    98: {"name": "QUICK ATTACK", "type": "NORMAL", "power": 40, "accuracy": 1.0},
    145: {"name": "BUBBLE", "type": "WATER", "power": 20, "accuracy": 1.0},
}
