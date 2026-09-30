"""
interactive/wram/addresses.py — SINGLE SOURCE OF TRUTH for all WRAM addresses
==============================================================================
Sourced from:
  - pret/pokered disassembly (canonical)
  - PokemonRedExperiments/baselines/memory_addresses.py (Whidden baseline)
  - DataCrystal ROM map wiki

Import ONLY from here. Never hardcode addresses inline elsewhere.
"""

# ---------------------------------------------------------------------------
# Spatial Navigation
# ---------------------------------------------------------------------------
MAP_N_ADDR  = 0xD35E   # wCurMap  — current map ID (0-247)
X_POS_ADDR  = 0xD362   # wXCoord  — player X tile
Y_POS_ADDR  = 0xD361   # wYCoord  — player Y tile

# ---------------------------------------------------------------------------
# Game State Flags
# ---------------------------------------------------------------------------
IS_IN_BATTLE_ADDR = 0xD057   # wIsInBattle — 0=overworld, >0=battle
TEXT_BOX_ID_ADDR  = 0xCF13   # wTextBoxID  — non-zero during active text box
MENU_TYPE_ADDR    = 0xCC24   # wMenuType   — non-zero when any menu is open
JOY_IGNORE_ADDR   = 0xCD6B   # wJoyIgnore  — hardware joypad suppression mask
                              #   bit 0=A, 1=B, 2=Sel, 3=Start
                              #   bit 4=Right, 5=Left, 6=Up, 7=Down

# ---------------------------------------------------------------------------
# Party / Battle
# ---------------------------------------------------------------------------
PARTY_SIZE_ADDR   = 0xD163   # wPartyCount — number of Pokémon (0-6)
PARTY_SPECIES_ADDRS = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]

# All 6 party Pokémon levels (Whidden baseline: LEVELS_ADDRESSES)
LEVELS_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

# All 6 HP / MaxHP — each 2-byte big-endian
HP_ADDRS     = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDRS = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]

# Lead Pokémon move IDs (4 moves) and PP
MOVE_IDS_BASE = 0xD173   # 0xD173..0xD176
MOVE_PPS_BASE = 0xD188   # 0xD188..0xD18B

# Opponent Pokémon levels (Whidden: OPPONENT_LEVELS_ADDRESSES)
OPP_LVL_ADDRS = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]

# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------
BADGE_ADDR = 0xD356   # wObtainedBadges — 8-bit bitfield

# ---------------------------------------------------------------------------
# Event Flags (quest progression)
# ---------------------------------------------------------------------------
EVENT_FLAGS_START = 0xD747   # wEventFlags start
EVENT_FLAGS_END   = 0xD886   # wEventFlags end (exclusive in range())

# Museum ticket — excluded from event scoring (same as Whidden baseline)
MUSEUM_TICKET_ADDR = 0xD754
MUSEUM_TICKET_BIT  = 0

# Starting event flags already set at game boot (baseline offset)
BASE_EVENT_FLAGS = 13

# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_2 = 0xD348
MONEY_ADDR_3 = 0xD349
