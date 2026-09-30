"""
interactive.emulator — Game Boy Hardware & PyBoy Lifecycle Wrapper
==================================================================
Encapsulates LR35902 CPU, SDL2 window rendering, joypad input pulse timings,
save state serialization, and raw screen framebuffer downsampling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np
import pyboy
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from pokemon_rl.env.wram_map import Action
from interactive.constants import ACTION_TO_PYBOY_EVENTS


class GameBoyEmulator:
    """
    Manages the PyBoy Game Boy emulation environment, display window,
    input pulse scheduling, and save state loading.
    """

    def __init__(
        self,
        rom_path: str | Path,
        state_path: Optional[str | Path] = None,
        headless: bool = False,
        emulation_speed: int = 16,
        scale: int = 3,
        sound: bool = False,
    ):
        self.rom_path = Path(rom_path)
        self.state_path = Path(state_path) if state_path else None
        self.headless = headless
        self.emulation_speed = emulation_speed
        self.scale = scale

        if not self.rom_path.exists():
            raise FileNotFoundError(f"Game Boy ROM not found at: {self.rom_path}")

        window_type = "null" if headless else "SDL2"
        print(f"[*] Initializing PyBoy (ROM: {self.rom_path.name}, Window: {window_type}, Scale: {scale}x, Speed: {emulation_speed}x)...")

        self.pyboy = PyBoy(
            str(self.rom_path),
            window=window_type,
            scale=scale,
            sound=sound,
            sound_emulated=sound,
        )

        if hasattr(self.pyboy, "set_emulation_speed"):
            self.pyboy.set_emulation_speed(emulation_speed)

        # Restore initial save state if provided
        if self.state_path and self.state_path.exists():
            print(f"[*] Loading Initial Save State: {self.state_path.name}...")
            with open(self.state_path, "rb") as f:
                self.pyboy.load_state(f)
            print("[+] Save state restored successfully.")
        else:
            print("[*] Starting from cold cartridge boot.")

    @property
    def memory(self):
        """Direct access to PyBoy Game Boy WRAM / HRAM memory bus."""
        return self.pyboy.memory

    def read_byte(self, address: int) -> int:
        """Read a single byte from LR35902 memory bus."""
        return self.pyboy.memory[address]

    def read_word_be(self, address: int) -> int:
        """Read a 16-bit big-endian unsigned integer from memory."""
        return (self.pyboy.memory[address] << 8) | self.pyboy.memory[address + 1]

    def get_screen_ndarray(self) -> np.ndarray:
        """
        Extract the current LCD screen buffer.
        Returns uint8 array of shape (144, 160, 4) in RGBA format.
        """
        return self.pyboy.screen.ndarray

    def get_downsampled_screen(self) -> np.ndarray:
        """
        Downsample 160x144 RGBA LCD screen to 80x72 RGB normalized float32 array.
        Returns shape (72, 80, 3) with values in [0.0, 1.0].
        """
        raw = self.pyboy.screen.ndarray
        return raw[::2, ::2, :3].astype(np.float32) / 255.0

    def tick(self, count: int = 1, render: bool = True) -> bool:
        """
        Step emulation forward by count frames.
        Returns False if the user closed the window or sent quit signal.
        """
        return self.pyboy.tick(count, render)

    def execute_action(
        self,
        action: int,
        press_ticks: int = 8,
        release_ticks: int = 16,
    ) -> bool:
        """
        Dispatch a hardware joypad press followed by release.
        Game Boy games require a finite pulse duration (typically >= 8 ticks)
        for the internal V-Blank Joypad interrupt handler to register the button.
        """
        if action not in ACTION_TO_PYBOY_EVENTS:
            # No-op action
            return self.tick(press_ticks + release_ticks, True)

        press_ev, release_ev = ACTION_TO_PYBOY_EVENTS[action]
        self.pyboy.send_input(press_ev)
        for _ in range(press_ticks):
            if not self.tick(1, True):
                return False

        self.pyboy.send_input(release_ev)
        for _ in range(release_ticks):
            if not self.tick(1, True):
                return False

        return True

    def close(self):
        """Cleanly terminate PyBoy SDL2 window and emulation thread."""
        try:
            self.pyboy.stop()
        except Exception:
            pass
