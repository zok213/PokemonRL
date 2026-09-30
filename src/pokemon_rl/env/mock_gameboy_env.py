"""
mock_gameboy_env.py — High-Throughput Vectorized Game Boy Environment Simulator
================================================================================
Vectorized simulation harness emulating Game Boy LR35902 CPU Work RAM registers,
joypad suppression (wJoyIgnore), 2D tile navigation, battle encounters, and dialogue.

Used for unit testing, continuous integration, and high-throughput benchmarking
without requiring a proprietary Game Boy ROM file.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np

from pokemon_rl.env.wram_map import Action, RAMMap


class MockVectorizedGameBoyEnv:
    """
    Simulates G parallel Game Boy instances in memory.
    Each instance maintains an 8KB/32KB WRAM buffer with authentic register offsets.
    """

    def __init__(self, num_envs: int = 8, wram_size: int = 8192):
        self.num_envs = num_envs
        self.wram_size = wram_size
        self.states = [bytearray(b"\x00" * wram_size) for _ in range(num_envs)]
        self.wram_dicts: List[Dict[int, int]] = []

        for i in range(num_envs):
            wdict = {
                RAMMap.CUR_MAP: 0,            # Pallet Town (Map 0)
                RAMMap.X_POS: 5,              # Spawn X
                RAMMap.Y_POS: 4,              # Spawn Y
                RAMMap.IS_IN_BATTLE: 0,
                RAMMap.TEXT_BOX_ID: 0,
                RAMMap.MENU_ACTIVE: 0,
                RAMMap.OBTAINED_BADGES: 0,
                RAMMap.SAFARI_STEPS_LO: 246,  # 500 steps initially (500 = 0x01F4)
                RAMMap.SAFARI_STEPS_HI: 1,
                RAMMap.JOY_IGNORE: 0,
                RAMMap.CUR_MENU_ITEM: 0,
            }
            self.wram_dicts.append(wdict)
            self._sync_dict_to_bytes(i)

    def _sync_dict_to_bytes(self, env_idx: int) -> None:
        """Write register dict values to bytearray at proper (addr - 0xC000) offsets."""
        buf = self.states[env_idx]
        wdict = self.wram_dicts[env_idx]
        for addr, val in wdict.items():
            offset = addr - 0xC000
            if 0 <= offset < len(buf):
                buf[offset] = val & 0xFF

    def read_wram(self, env_idx: int, addr: int) -> int:
        """Read 1 byte from environment instance at address addr."""
        offset = addr - 0xC000
        if 0 <= offset < len(self.states[env_idx]):
            return self.states[env_idx][offset]
        return self.wram_dicts[env_idx].get(addr, 0)

    def write_wram(self, env_idx: int, addr: int, val: int) -> None:
        """Write 1 byte to environment instance."""
        self.wram_dicts[env_idx][addr] = val & 0xFF
        offset = addr - 0xC000
        if 0 <= offset < len(self.states[env_idx]):
            self.states[env_idx][offset] = val & 0xFF

    def step(self, actions: List[int]) -> List[Tuple[Dict[str, Any], float, bool, bytes]]:
        """
        Step all environments synchronously.

        Returns list of (wram_info_dict, reward, done, wram_bytes)
        """
        results = []
        for i in range(self.num_envs):
            act = actions[i]

            # Simulate CPU Work RAM write activity
            mut_idx = (act * 107 + i * 43) % self.wram_size
            self.states[i][mut_idx] = (self.states[i][mut_idx] + 1) & 0xFF

            # Movement physics on 2D map
            old_x = self.wram_dicts[i][RAMMap.X_POS]
            old_y = self.wram_dicts[i][RAMMap.Y_POS]

            if act == Action.UP:
                self.wram_dicts[i][RAMMap.Y_POS] = max(0, old_y - 1)
            elif act == Action.DOWN:
                self.wram_dicts[i][RAMMap.Y_POS] = min(50, old_y + 1)
            elif act == Action.LEFT:
                self.wram_dicts[i][RAMMap.X_POS] = max(0, old_x - 1)
            elif act == Action.RIGHT:
                self.wram_dicts[i][RAMMap.X_POS] = min(50, old_x + 1)

            # Decrement Safari steps if active
            steps_lo = self.wram_dicts[i][RAMMap.SAFARI_STEPS_LO]
            steps_hi = self.wram_dicts[i][RAMMap.SAFARI_STEPS_HI]
            total_steps = steps_lo + (steps_hi << 8)
            if total_steps > 0:
                total_steps -= 1
                self.wram_dicts[i][RAMMap.SAFARI_STEPS_LO] = total_steps & 0xFF
                self.wram_dicts[i][RAMMap.SAFARI_STEPS_HI] = (total_steps >> 8) & 0xFF

            self._sync_dict_to_bytes(i)

            reward = 0.05
            done = False
            wram_info = dict(self.wram_dicts[i])
            state_bytes = bytes(self.states[i])
            results.append((wram_info, reward, done, state_bytes))

        return results

    def reset_env(self, env_idx: int) -> bytes:
        """Reset a specific environment to initial state."""
        self.states[env_idx] = bytearray(b"\x00" * self.wram_size)
        self.wram_dicts[env_idx] = {
            RAMMap.CUR_MAP: 0,
            RAMMap.X_POS: 5,
            RAMMap.Y_POS: 4,
            RAMMap.IS_IN_BATTLE: 0,
            RAMMap.TEXT_BOX_ID: 0,
            RAMMap.MENU_ACTIVE: 0,
            RAMMap.OBTAINED_BADGES: 0,
            RAMMap.SAFARI_STEPS_LO: 246,
            RAMMap.SAFARI_STEPS_HI: 1,
            RAMMap.JOY_IGNORE: 0,
            RAMMap.CUR_MENU_ITEM: 0,
        }
        self._sync_dict_to_bytes(env_idx)
        return bytes(self.states[env_idx])
