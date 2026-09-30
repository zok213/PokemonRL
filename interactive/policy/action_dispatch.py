"""
interactive/policy/action_dispatch.py — Action Selection, Masking & Navigation Dispatch
========================================================================================
Handles all decision logic between the neural policy, symbolic heuristics, and Game Boy joypad:

  1. Mode Detection:
       - BATTLE: Tactical combat controller (type-advantage math)
       - DIALOGUE / MENU LOCK: Multi-phase dialogue advance & menu escape
       - OVERWORLD: Whidden 439M PPO policy with neuro-symbolic action guidance

  2. Neuro-Symbolic Action Guidance (Overworld):
       - START suppressed (prevents agent from self-trapping in bag/menus)
       - Oak's Lab dialogue suppression (suppresses A when Y <= 4 in Oak's Lab)
       - Multi-Directional Wall-Bump Memory: Remembers all impassable directions
         from the current tile until the agent successfully moves to a new tile
       - Anti-Oscillation Cycle Breaker: Detects 2-cycle and 3-cycle ping-pong limit
         cycles and suppresses immediate step reversals
       - Novelty-Guided Stagnation Evasion: When stalled without novel tiles for >20 steps,
         suppresses moves back into over-trafficked (>15 visits) neighbor tiles

  3. Hardware Joypad Dispatch:
       - 8-tick press + 16-tick release pulse for joypad register latching
       - PASS action uses full 24-tick no-op
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, Set
from collections import deque
import numpy as np
import torch
from pyboy.utils import WindowEvent

from interactive.wram.reader import (
    is_in_battle, is_menu_or_text_active, read_joy_ignore, read_xy, read_map_id
)
from interactive.constants import (
    WHIDDEN_ACTION_TO_PYBOY_EVENTS, WHIDDEN_ACTION_NAMES, GEN1_MOVE_DATABASE
)
from pokemon_rl.env.wram_map import Action

PRESS_TICKS   = 8
RELEASE_TICKS = 16
ACTION_FREQ   = 24

# Whidden action indices
IDX_DOWN  = 0
IDX_LEFT  = 1
IDX_RIGHT = 2
IDX_UP    = 3
IDX_A     = 4
IDX_B     = 5
IDX_START = 6
IDX_PASS  = 7

REVERSE_MAP = {
    IDX_DOWN:  IDX_UP,
    IDX_UP:    IDX_DOWN,
    IDX_LEFT:  IDX_RIGHT,
    IDX_RIGHT: IDX_LEFT,
}

DELTA_MAP = {
    IDX_DOWN:  (0, 1),
    IDX_LEFT:  (-1, 0),
    IDX_RIGHT: (1, 0),
    IDX_UP:    (0, -1),
}


class ActionDispatcher:
    """
    Selects, guides, masks, and dispatches actions to the Game Boy emulator.
    Combines deep reinforcement learning with neuro-symbolic navigation constraints.
    """

    def __init__(self, use_action_masking: bool = True):
        self.use_action_masking = use_action_masking
        self.last_pos:           Optional[Tuple[int, int, int]] = None
        self.last_action:        Optional[int] = None
        self.blocked_dirs:       Set[int] = set()
        self.pos_history:        deque = deque(maxlen=24)
        self.tile_visits:        Dict[Tuple[int, int, int], int] = {}
        self.steps_since_novel:  int = 0
        self._escape_step_count: int = 0
        self._post_dialogue_cooldown: int = 0

    def select_action(
        self,
        mem,
        policy,
        obs_t: torch.Tensor,
        reward_tracker=None,
    ) -> Tuple[int, np.ndarray, str]:
        """
        Select next action from live WRAM state.

        Returns:
            (action_idx, probs_array, subsystem_label)
        """
        in_battle = is_in_battle(mem)
        in_menu   = is_menu_or_text_active(mem)

        # --- BATTLE MODE ---
        if in_battle:
            return self._battle_action(mem)

        # --- MENU / DIALOGUE LOCK ---
        if in_menu:
            return self._menu_escape_action()

        # If transitioning out of a dialogue, set cooldown to step away from NPC/sign
        if self._escape_step_count > 0:
            self._post_dialogue_cooldown = 8   # was 4 — give agent more time to walk away
            self._escape_step_count = 0

        # --- OVERWORLD: PPO + Neuro-Symbolic Guidance ---
        return self._overworld_action(mem, policy, obs_t, reward_tracker)

    def dispatch(self, pyboy, action_idx: int) -> bool:
        """
        Send press+release pulse to PyBoy for the given action index.
        Returns False if PyBoy window was closed.
        """
        press_ev, release_ev = WHIDDEN_ACTION_TO_PYBOY_EVENTS[action_idx]

        if press_ev == WindowEvent.PASS:
            return pyboy.tick(ACTION_FREQ, True)

        pyboy.send_input(press_ev)
        for _ in range(PRESS_TICKS):
            if not pyboy.tick(1, True):
                return False

        pyboy.send_input(release_ev)
        for _ in range(RELEASE_TICKS):
            if not pyboy.tick(1, True):
                return False

        return True

    # ------------------------------------------------------------------
    # Subsystems
    # ------------------------------------------------------------------

    def _overworld_action(
        self, mem, policy, obs_t: torch.Tensor, reward_tracker=None
    ) -> Tuple[int, np.ndarray, str]:
        """Whidden 439M PPO policy with neuro-symbolic action guidance."""
        cur_map = read_map_id(mem)
        cur_x, cur_y = read_xy(mem)
        cur_pos = (cur_map, cur_x, cur_y)

        # 1. Update visitation frequency & novelty
        is_novel = cur_pos not in self.tile_visits
        self.tile_visits[cur_pos] = self.tile_visits.get(cur_pos, 0) + 1
        if is_novel:
            self.steps_since_novel = 0
        else:
            self.steps_since_novel += 1

        # 2. Multi-Directional Wall-Bump Memory
        if self.last_pos is not None:
            if cur_pos == self.last_pos:
                # Agent attempted an action but stayed on the exact same tile
                if self.last_action in (IDX_DOWN, IDX_LEFT, IDX_RIGHT, IDX_UP):
                    self.blocked_dirs.add(self.last_action)
            else:
                # Successfully moved to a new tile: clear wall blocks
                self.blocked_dirs.clear()

        self.last_pos = cur_pos
        self.pos_history.append(cur_pos)

        # 3. Anti-Oscillation Cycle Detection (2-cycle or 3-cycle limit cycles)
        oscillating = False
        if len(self.pos_history) >= 4:
            if cur_pos == self.pos_history[-3] or cur_pos == self.pos_history[-4]:
                oscillating = True

        # 4. Construct Action Mask
        mask = np.ones(8, dtype=bool)

        # Always suppress START (prevents agent from self-trapping in menu/bag)
        mask[IDX_START] = False

        # Oak's Lab dialogue suppression (adjacent to Oak at Y <= 4)
        if cur_map == 40 and cur_y <= 4:
            mask[IDX_A] = False

        # Post-dialogue cooldown: suppress A for 4 steps to step away from NPC/sign
        if self._post_dialogue_cooldown > 0:
            mask[IDX_A] = False
            self._post_dialogue_cooldown -= 1

        if self.use_action_masking:
            # Hardware joypad_ignore passthrough
            joy = read_joy_ignore(mem)
            if joy > 0:
                if joy & 0x80: mask[IDX_DOWN]  = False
                if joy & 0x20: mask[IDX_LEFT]  = False
                if joy & 0x10: mask[IDX_RIGHT] = False
                if joy & 0x40: mask[IDX_UP]    = False
                if joy & 0x01: mask[IDX_A]     = False
                if joy & 0x02: mask[IDX_B]     = False
                if joy & 0x08: mask[IDX_START] = False

            # Suppress all known blocked directions from this tile
            for b in self.blocked_dirs:
                mask[b] = False

            # Anti-oscillation: suppress immediate step reversal
            if oscillating and self.last_action in REVERSE_MAP:
                mask[REVERSE_MAP[self.last_action]] = False

            # Novelty-guided stagnation evasion:
            # If no novel tile discovered for >20 steps, suppress moves back into
            # over-trafficked (>15 visits) neighbor tiles
            if self.steps_since_novel > 20:
                for act, (dx, dy) in DELTA_MAP.items():
                    target_tile = (cur_map, cur_x + dx, cur_y + dy)
                    if self.tile_visits.get(target_tile, 0) > 15 and mask[act]:
                        other_moves = [a for a in (IDX_DOWN, IDX_LEFT, IDX_RIGHT, IDX_UP) if a != act and mask[a]]
                        if other_moves:
                            mask[act] = False

            # Safety fallback: ensure at least one action is valid
            if not np.any(mask):
                self.blocked_dirs.clear()
                mask = np.ones(8, dtype=bool)
                mask[IDX_START] = False
                if cur_map == 40 and cur_y <= 4:
                    mask[IDX_A] = False

        # 5. Forward Pass through 439M Policy
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            probs_t, _ = policy(obs_t, action_mask=mask_t)

        probs      = probs_t[0].numpy()
        action_idx = int(np.random.choice(len(probs), p=probs))
        self.last_action = action_idx

        # Subsystem description for HUD
        subsystem = "AUTONOMOUS PPO (439M WHIDDEN)"
        tags = []
        if self.blocked_dirs:
            tags.append(f"Blk:{[WHIDDEN_ACTION_NAMES[b] for b in self.blocked_dirs]}")
        if oscillating:
            tags.append("Anti-Osc")
        if self.steps_since_novel > 20:
            tags.append(f"Stag:{self.steps_since_novel}")
        if tags:
            subsystem += f" [{', '.join(tags)}]"

        return action_idx, probs, subsystem

    def _menu_escape_action(self) -> Tuple[int, np.ndarray, str]:
        """
        Smart dialogue / menu escape strategy:
          - Steps  1-20: press A to advance multi-line NPC dialogues (Oak etc.)
            Extended from 8 to 20 because Oak's Lab cutscene is ~10-30 lines.
            Previously at step 8 we switched to B too early, causing an infinite
            A→B→A loop where the text box would immediately reopen.
          - Steps 21+:   press B to close and exit text box / menu
        """
        self._escape_step_count += 1

        if self._escape_step_count <= 20:   # was 8
            action_idx = IDX_A
            label = "A (advance text)"
        else:
            action_idx = IDX_B
            label = "B (close / exit)"

        probs = np.zeros(8, dtype=np.float32)
        probs[action_idx] = 1.0
        return action_idx, probs, f"DIALOGUE/MENU ESCAPE [{label}] step={self._escape_step_count}"

    def _battle_action(self, mem) -> Tuple[int, np.ndarray, str]:
        """Select A or B based on type-advantage tactical heuristic."""
        from pokemon_rl.combat.combat_controller import DecoupledCombatController
        if not hasattr(self, "_combat"):
            self._combat = DecoupledCombatController()

        party_size = mem[0xD163]
        move_ids   = [mem[0xD173 + i] for i in range(4)] if party_size > 0 else [33]
        move_pps   = [mem[0xD188 + i] for i in range(4)] if party_size > 0 else [35]

        available = []
        for m_id, pp in zip(move_ids, move_pps):
            if m_id > 0 and pp > 0:
                info = GEN1_MOVE_DATABASE.get(m_id, {"name": f"MOVE_{m_id}", "type": "NORMAL", "power": 40, "accuracy": 1.0})
                available.append({**info, "pp": pp})
        if not available:
            available = [{"name": "TACKLE", "power": 40, "type": "NORMAL", "accuracy": 1.0, "pp": 35}]

        battle_result = self._combat.select_battle_action({
            "opponent_type": "NORMAL",
            "available_moves": available,
        })
        action_idx = IDX_A if int(battle_result) == Action.A else IDX_B
        probs = np.zeros(8, dtype=np.float32)
        probs[action_idx] = 1.0
        return action_idx, probs, "TACTICAL COMBAT HEAD (GEN 1 TYPE MATH)"
