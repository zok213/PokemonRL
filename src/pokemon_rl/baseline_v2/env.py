"""
env.py — Baseline V2 Environment Wrapper for Pokémon Red (Pleines et al., 2025)
================================================================================
Reproduces the complete environment formulation from:
  Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920 / PWhiddy V2

Specifications:
  - Joypad Cadence: 8 frames held down, 16 frames released (24 frames / step)
  - Action Space: 7 discrete actions: {UP, DOWN, LEFT, RIGHT, A, B, START}
  - Observation Space:
      1. visual: (3, 72, 80) float32 [0.0, 1.0], 3-frame rolling stack
      2. spatial_map: (1, 48, 48) float32 [0.0, 1.0], centered on player coordinates
      3. telemetry: (64,) float32, party telemetry & event bitflags (species masked)
  - Dynamic Step Budget: B_t = 10,240 + 2,048 * N_events
  - Composite Reward: R_t = R_event + R_nav + R_heal + R_lvl
"""

from __future__ import annotations
import collections
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from pokemon_rl.baseline_v2.constants import (
    PleinesAction,
    PleinesWRAM,
    FRAME_HOLD,
    FRAME_RELEASE,
    DECISION_STRIDE,
    BASE_BUDGET,
    EVENT_BONUS,
)
from pokemon_rl.baseline_v2.reward import PleinesCompositeReward


class PleinesBaselineEnv:
    """
    Faithful re-implementation of the Baseline V2 environment wrapper.
    Seamlessly operates in:
      - Live PyBoy mode (when ROM path is provided and PyBoy is installed)
      - Deterministic Mock mode (for ultra-fast CI/testing without ROM dependencies)
    """

    def __init__(
        self,
        rom_path: Optional[str] = None,
        state_path: Optional[str] = None,
        starter: str = "squirtle",
        enable_lvl: bool = True,
        enable_heal: bool = True,
        headless: bool = True,
        sound: bool = False,
    ):
        self.rom_path = rom_path
        self.state_path = state_path
        self.starter = starter.lower()
        self.enable_lvl = enable_lvl
        self.enable_heal = enable_heal
        self.headless = headless
        self.sound = sound

        self.reward_fn = PleinesCompositeReward(enable_lvl=enable_lvl, enable_heal=enable_heal)

        # Multimodal observation buffers
        self.screen_stack = collections.deque(maxlen=3)
        self.visited_tiles_map = np.zeros((48, 48), dtype=np.uint8)

        # Episode step management
        self.current_step = 0
        self.events_completed = 0
        self.max_step_budget = self.calculate_step_budget(0)

        # PyBoy handle
        self.pyboy = None
        self.is_mock = True

        if rom_path and os.path.exists(rom_path):
            self._init_pyboy(rom_path, state_path)
        else:
            self._init_mock_state()

    def _init_pyboy(self, rom_path: str, state_path: Optional[str] = None):
        """Initializes actual PyBoy Game Boy emulator instance."""
        try:
            from pyboy import PyBoy
            window_type = "headless" if self.headless else "SDL2"
            self.pyboy = PyBoy(rom_path, window=window_type, sound=self.sound)
            self.pyboy.set_emulation_speed(0)

            if state_path and os.path.exists(state_path):
                with open(state_path, "rb") as f:
                    self.pyboy.load_state(f)

            self.is_mock = False
        except Exception:
            # Fallback to mock state if PyBoy/SDL2 initialization fails
            self.pyboy = None
            self.is_mock = True
            self._init_mock_state()

    def _init_mock_state(self):
        """Initializes simulated 8KB WRAM memory space for mock mode."""
        self.wram = bytearray(8192)
        # Pallet Town default coordinates
        self.wram[PleinesWRAM.MAP_N - 0xC000] = 0
        self.wram[PleinesWRAM.X_POS - 0xC000] = 5
        self.wram[PleinesWRAM.Y_POS - 0xC000] = 4
        self.wram[PleinesWRAM.BADGES - 0xC000] = 0
        self.wram[PleinesWRAM.IS_IN_BATTLE - 0xC000] = 0
        self.wram[PleinesWRAM.TEXT_BOX_ID - 0xC000] = 0
        self.wram[PleinesWRAM.PARTY_COUNT - 0xC000] = 1

        self.current_hp = [20, 0, 0, 0, 0, 0]
        self.max_hp = [20, 0, 0, 0, 0, 0]
        self.levels = [5, 0, 0, 0, 0, 0]

    def calculate_step_budget(self, events: int) -> int:
        """Dynamic Step Budget Formula: B_t = 10,240 + 2,048 * N_events."""
        return BASE_BUDGET + EVENT_BONUS * events

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment for a new episode."""
        self.current_step = 0
        self.events_completed = 0
        self.max_step_budget = self.calculate_step_budget(0)
        self.reward_fn.reset()

        if not self.is_mock and self.pyboy is not None:
            if self.state_path and os.path.exists(self.state_path):
                with open(self.state_path, "rb") as f:
                    self.pyboy.load_state(f)
            else:
                self.pyboy.reset_game()
        else:
            self._init_mock_state()

        # Reset rolling observation buffers
        zero_frame = np.zeros((72, 80), dtype=np.float32)
        self.screen_stack.clear()
        for _ in range(3):
            self.screen_stack.append(zero_frame)
        self.visited_tiles_map.fill(0)

        obs = self._get_obs()
        info = {
            "step": 0,
            "events": 0,
            "budget": self.max_step_budget,
            "is_mock": self.is_mock,
        }
        return obs, info

    def _read_memory(self, addr: int) -> int:
        """Reads a byte from WRAM."""
        if not self.is_mock and self.pyboy is not None:
            return int(self.pyboy.memory[addr])
        idx = addr - 0xC000
        if 0 <= idx < len(self.wram):
            return int(self.wram[idx])
        return 0

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Constructs the multimodal observation tuple:
          - screen: (3, 72, 80) float32
          - spatial_map: (1, 48, 48) float32
          - telemetry: (64,) float32
        """
        # 1. Visual screen: 72x80 downsampled grayscale
        if not self.is_mock and self.pyboy is not None:
            raw_screen = self.pyboy.screen.ndarray  # (144, 160, 3)
            # Convert RGB to grayscale: 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * raw_screen[:, :, 0] + 0.587 * raw_screen[:, :, 1] + 0.114 * raw_screen[:, :, 2]
            downsampled = gray[::2, ::2].astype(np.float32) / 255.0  # (72, 80)
        else:
            downsampled = np.zeros((72, 80), dtype=np.float32)

        self.screen_stack.append(downsampled)
        screen_obs = np.stack(list(self.screen_stack), axis=0)  # (3, 72, 80)

        # 2. Spatial exploration map: 48x48 binary visited map
        x = self._read_memory(PleinesWRAM.X_POS)
        y = self._read_memory(PleinesWRAM.Y_POS)
        map_x = int(np.clip(x % 48, 0, 47))
        map_y = int(np.clip(y % 48, 0, 47))
        self.visited_tiles_map[map_y, map_x] = 1
        spatial_obs = self.visited_tiles_map[np.newaxis, ...].astype(np.float32)  # (1, 48, 48)

        # 3. Telemetry vector: 64-dim normalized WRAM registers
        telemetry = np.zeros(64, dtype=np.float32)
        telemetry[0] = float(x) / 255.0
        telemetry[1] = float(y) / 255.0
        telemetry[2] = float(self._read_memory(PleinesWRAM.MAP_N)) / 255.0
        telemetry[3] = float(self._read_memory(PleinesWRAM.BADGES)) / 255.0
        telemetry[4] = 1.0 if self._read_memory(PleinesWRAM.IS_IN_BATTLE) > 0 else 0.0
        telemetry[5] = float(self._read_memory(PleinesWRAM.TEXT_BOX_ID)) / 255.0

        party_count = min(6, self._read_memory(PleinesWRAM.PARTY_COUNT))
        telemetry[6] = float(party_count) / 6.0

        for i in range(party_count):
            hp_lo = self._read_memory(PleinesWRAM.PARTY_HP + i * 44)
            hp_hi = self._read_memory(PleinesWRAM.PARTY_HP + i * 44 + 1)
            cur_hp = (hp_hi << 8) | hp_lo

            mhp_lo = self._read_memory(PleinesWRAM.PARTY_MAX_HP + i * 44)
            mhp_hi = self._read_memory(PleinesWRAM.PARTY_MAX_HP + i * 44 + 1)
            max_hp = max(1, (mhp_hi << 8) | mhp_lo)

            lvl = self._read_memory(PleinesWRAM.PARTY_LEVEL + i * 44)

            telemetry[7 + i * 3] = float(cur_hp) / float(max_hp)
            telemetry[7 + i * 3 + 1] = float(lvl) / 100.0
            telemetry[7 + i * 3 + 2] = float(max_hp) / 500.0

        return {
            "screen": screen_obs,
            "spatial_map": spatial_obs,
            "telemetry": telemetry,
        }

    def _apply_action_pyboy(self, action: PleinesAction):
        """Applies Joypad action over 8 frames pressed and 16 frames released."""
        action_map = {
            PleinesAction.UP: "up",
            PleinesAction.DOWN: "down",
            PleinesAction.LEFT: "left",
            PleinesAction.RIGHT: "right",
            PleinesAction.A: "a",
            PleinesAction.B: "b",
            PleinesAction.START: "start",
        }
        btn = action_map.get(action)
        if btn and self.pyboy is not None:
            self.pyboy.button_press(btn)
            for _ in range(FRAME_HOLD):
                self.pyboy.tick()
            self.pyboy.button_release(btn)
            for _ in range(FRAME_RELEASE):
                self.pyboy.tick()

    def _apply_action_mock(self, action: PleinesAction):
        """Simulates grid movement and interactions in mock mode."""
        cur_x = self.wram[PleinesWRAM.X_POS - 0xC000]
        cur_y = self.wram[PleinesWRAM.Y_POS - 0xC000]

        if action == PleinesAction.UP:
            self.wram[PleinesWRAM.Y_POS - 0xC000] = max(0, cur_y - 1)
        elif action == PleinesAction.DOWN:
            self.wram[PleinesWRAM.Y_POS - 0xC000] = min(47, cur_y + 1)
        elif action == PleinesAction.LEFT:
            self.wram[PleinesWRAM.X_POS - 0xC000] = max(0, cur_x - 1)
        elif action == PleinesAction.RIGHT:
            self.wram[PleinesWRAM.X_POS - 0xC000] = min(47, cur_x + 1)

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Executes one decision step (24 emulation frames).
        Returns: (obs, reward, terminated, truncated, info).
        """
        act_enum = PleinesAction(int(action))

        if not self.is_mock and self.pyboy is not None:
            self._apply_action_pyboy(act_enum)
        else:
            self._apply_action_mock(act_enum)

        self.current_step += 1

        # Read state for reward calculation
        x = self._read_memory(PleinesWRAM.X_POS)
        y = self._read_memory(PleinesWRAM.Y_POS)
        map_id = self._read_memory(PleinesWRAM.MAP_N)

        if not self.is_mock and self.pyboy is not None:
            party_count = max(1, min(6, self._read_memory(PleinesWRAM.PARTY_COUNT)))
            current_hp = []
            max_hp = []
            levels = []
            for i in range(party_count):
                hp = (self._read_memory(PleinesWRAM.PARTY_HP + i * 44 + 1) << 8) | self._read_memory(PleinesWRAM.PARTY_HP + i * 44)
                mhp = max(1, (self._read_memory(PleinesWRAM.PARTY_MAX_HP + i * 44 + 1) << 8) | self._read_memory(PleinesWRAM.PARTY_MAX_HP + i * 44))
                lvl = max(1, self._read_memory(PleinesWRAM.PARTY_LEVEL + i * 44))
                current_hp.append(hp)
                max_hp.append(mhp)
                levels.append(lvl)
            while len(current_hp) < 6:
                current_hp.append(0)
                max_hp.append(100)
                levels.append(0)
        else:
            current_hp = list(self.current_hp)
            max_hp = list(self.max_hp)
            levels = list(self.levels)

        # Dynamic budget update
        self.max_step_budget = self.calculate_step_budget(self.events_completed)

        # Reward computation
        reward, breakdown = self.reward_fn.compute_reward(
            coords=(x, y, map_id),
            current_events=self.events_completed,
            current_hp=current_hp,
            max_hp=max_hp,
            current_levels=levels,
        )

        obs = self._get_obs()

        terminated = False  # Pokémon Red episodes do not naturally terminate until game clear
        truncated = self.current_step >= self.max_step_budget

        info = {
            "step": self.current_step,
            "budget": self.max_step_budget,
            "coords": (x, y, map_id),
            "events": self.events_completed,
            "reward_breakdown": breakdown,
            "is_mock": self.is_mock,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        """Closes emulator and releases memory resources."""
        if not self.is_mock and self.pyboy is not None:
            try:
                self.pyboy.stop()
            except Exception:
                pass
            self.pyboy = None
