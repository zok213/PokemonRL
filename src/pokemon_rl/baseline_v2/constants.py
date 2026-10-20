"""
constants.py — Exact Constants, Memory Addresses, and Hyperparameters for Baseline V2
=====================================================================================
Directly mapped from:
  Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920 / PWhiddy V2
"""

from __future__ import annotations
from enum import IntEnum
from typing import Dict, Any


class PleinesAction(IntEnum):
    """
    The 7 discrete Game Boy controller actions used in Pleines et al. (2025).
    SELECT is intentionally omitted as non-essential for progression.
    """
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    A     = 4
    B     = 5
    START = 6
    NUM_ACTIONS = 7


class PleinesWRAM:
    """Canonical Work RAM registers tracked in Pleines et al."""
    MAP_N        = 0xD35E  # Current map ID index
    X_POS        = 0xD362  # Player X coordinate (wPlayerXCoord)
    Y_POS        = 0xD361  # Player Y coordinate (wPlayerYCoord)
    BADGES       = 0xD356  # Acquired Gym Badges bitfield (0x01 to 0x80)
    IS_IN_BATTLE = 0xD057  # Battle state (0 = Overworld, >0 = Battle)
    TEXT_BOX_ID  = 0xCF13  # Active dialogue text box
    PARTY_COUNT  = 0xD163  # Number of Pokémon in party (1-6)
    PARTY_HP     = 0xD16C  # Base address for party HP (2 bytes per Pokémon)
    PARTY_MAX_HP = 0xD18D  # Base address for party Max HP (2 bytes per Pokémon)
    PARTY_LEVEL  = 0xD18C  # Base address for party level (wPartyMon1Level)
    EVENT_FLAGS  = 0xD5A6  # Start of story event flags (320 bytes through 0xD85F)


# Exact paper execution parameters
FRAME_HOLD: int = 8          # Joypad button held down for 8 emulation frames
FRAME_RELEASE: int = 16      # Joypad released for 16 emulation frames
DECISION_STRIDE: int = 24    # 1 decision step = 24 Game Boy frames (~392 SPS)

# Dynamic step budget: B_t = BASE_BUDGET + EVENT_BONUS * N_events
BASE_BUDGET: int = 10240
EVENT_BONUS: int = 2048

# Composite linear reward weights (Equation: R_t = R_event + R_nav + R_heal + R_lvl)
WEIGHT_EVENT: float = 2.0    # 2.0 * Delta N_events
WEIGHT_NAV: float = 0.005    # 0.005 * I[coords not in H_visited]
WEIGHT_HEAL: float = 2.5     # 2.5 * sum_i (HP_gain_i / MaxHP_i)
WEIGHT_LVL: float = 0.5      # 0.5 * Delta LevelPotential

# PPO-GAE hyperparameters from Table 2 of Pleines et al. (2025)
GAMMA: float = 0.997         # Discount factor
LAMBDA_GAE: float = 0.95     # GAE smoothing factor
CLIP_RANGE: float = 0.2      # PPO surrogate clipping epsilon
LEARNING_RATE: float = 2.5e-4
VF_COEF: float = 0.5         # Value function loss coefficient
ENT_COEF: float = 0.01       # Entropy bonus coefficient

# Ground-truth empirical results reported in Table 3 of Pleines et al. (2025)
# (Mean +/- Std across 5 independent seeds)
PLEINES_TABLE3_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "human_playthroughs": {
        "milestones": "7.00 +/- 0.0",
        "beat_brock": "1.00 +/- 0.0",
        "mt_moon": "1.00 +/- 0.0",
        "cerulean_city": "1.00 +/- 0.0",
        "beat_misty": "1.00 +/- 0.0",
        "bills_quest": "1.00 +/- 0.0",
        "cerulean_done": "1.00 +/- 0.0",
        "brock_steps": "5403 +/- 3824",
        "cerulean_steps": "11188 +/- 5224",
        "heals_farmed": "22.97 +/- 11.3",
        "species_entropy": "3.53 +/- 0.0",
    },
    "baseline_squirtle": {
        "milestones": "4.19 +/- 0.2",
        "beat_brock": "0.99 +/- 0.0",
        "mt_moon": "0.97 +/- 0.0",
        "cerulean_city": "0.85 +/- 0.1",
        "beat_misty": "0.27 +/- 0.3",
        "bills_quest": "0.11 +/- 0.2",
        "cerulean_done": "0.00 +/- 0.0",
        "brock_steps": "5587 +/- 1235",
        "cerulean_steps": "25299 +/- 4950",
        "heals_farmed": "12.81 +/- 5.9",
        "species_entropy": "2.61 +/- 0.1",
    },
    "baseline_no_lvl": {
        "milestones": "4.60 +/- 0.7",
        "beat_brock": "0.95 +/- 0.0",
        "mt_moon": "0.91 +/- 0.1",
        "cerulean_city": "0.79 +/- 0.1",
        "beat_misty": "0.31 +/- 0.3",
        "bills_quest": "0.32 +/- 0.4",
        "cerulean_done": "0.03 +/- 0.1",
        "brock_steps": "6117 +/- 1758",
        "cerulean_steps": "21352 +/- 3898",
        "heals_farmed": "11.73 +/- 4.0",
        "species_entropy": "1.24 +/- 0.2",
    },
    "baseline_no_heal": {
        "milestones": "3.94 +/- 0.2",
        "beat_brock": "0.99 +/- 0.0",
        "mt_moon": "0.97 +/- 0.0",
        "cerulean_city": "0.91 +/- 0.1",
        "beat_misty": "0.07 +/- 0.1",
        "bills_quest": "0.00 +/- 0.0",
        "cerulean_done": "0.00 +/- 0.0",
        "brock_steps": "5497 +/- 503",
        "cerulean_steps": "28813 +/- 2880",
        "heals_farmed": "1.45 +/- 0.7",
        "species_entropy": "2.27 +/- 0.2",
    },
    "gru_memory_body": {
        "milestones": "3.93 +/- 2.2",
        "beat_brock": "0.76 +/- 0.4",
        "mt_moon": "0.71 +/- 0.4",
        "cerulean_city": "0.49 +/- 0.4",
        "beat_misty": "0.21 +/- 0.3",
        "bills_quest": "0.48 +/- 0.4",
        "cerulean_done": "0.21 +/- 0.3",
        "brock_steps": "5624 +/- 780",
        "cerulean_steps": "27118 +/- 4236",
        "heals_farmed": "23.09 +/- 23.3",
        "species_entropy": "1.87 +/- 1.0",
    },
    "choose_starter_pallet": {
        "milestones": "3.43 +/- 1.7",
        "beat_brock": "0.79 +/- 0.4",
        "mt_moon": "0.79 +/- 0.4",
        "cerulean_city": "0.75 +/- 0.4",
        "beat_misty": "0.29 +/- 0.4",
        "bills_quest": "0.00 +/- 0.0",
        "cerulean_done": "0.00 +/- 0.0",
        "brock_steps": "6485 +/- 578",
        "cerulean_steps": "30593 +/- 1143",
        "heals_farmed": "33.75 +/- 34.1",
        "species_entropy": "1.25 +/- 0.6",
    },
    "bulbasaur_starter": {
        "milestones": "3.00 +/- 0.0",
        "beat_brock": "0.94 +/- 0.0",
        "mt_moon": "0.94 +/- 0.0",
        "cerulean_city": "0.00 +/- 0.0",
        "beat_misty": "0.00 +/- 0.0",
        "bills_quest": "0.00 +/- 0.0",
        "cerulean_done": "0.00 +/- 0.0",
        "brock_steps": "6231 +/- 1282",
        "cerulean_steps": "N/A (Trapped at Mt. Moon)",
        "heals_farmed": "399.33 +/- 177.1",  # Pathological Leech Life trap!
        "species_entropy": "2.34 +/- 0.2",
    },
}
