"""
interactive/policy/observation.py — Exact Whidden Observation Tensor Builder
=============================================================================
Constructs the exact (1, 3, 128, 40) float32 observation tensor that Peter
Whidden's 439M pretrained policy network expects, mirroring red_gym_env.render().

Tensor Specification (128 rows × 40 columns × 3 channels):
  - Rows  0..7:   Exploration Progress Memory Bar (level, HP, explore, badges)
  - Rows  8..9:   Padding (zeros, 2 rows)
  - Rows 10..17:  Recent Step Reward Memory Bar (rolling short-term reward history)
  - Rows 18..19:  Padding (zeros, 2 rows)
  - Rows 20..55:  Game Boy Frame t   (Current / NEWEST frame, 36 rows)
  - Rows 56..91:  Game Boy Frame t-1 (Previous frame, 36 rows)
  - Rows 92..127: Game Boy Frame t-2 (Oldest frame, 36 rows)

Total Dimensions: (128, 40, 3) -> PyTorch (1, 3, 128, 40) normalized in [0.0, 1.0].
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from math import floor
import numpy as np
import torch

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class WhiddenObservationBuilder:
    """
    Exact observation builder replicating Peter Whidden's RedGymEnv.render()
    with frame_stacks=3, memory_height=8, mem_padding=2, output_shape=(36, 40, 3).
    """

    def __init__(self):
        self.output_shape   = (36, 40, 3)
        self.mem_padding    = 2
        self.memory_height  = 8
        self.col_steps      = 16
        self.recent_memory  = np.zeros((40 * 8, 3), dtype=np.uint8)
        self.recent_frames  = np.zeros((3, 36, 40, 3), dtype=np.uint8)

    def build(
        self,
        raw_screen_rgba: np.ndarray,
        telemetry: Dict[str, Any],
        visited_count: int,
        whidden_tracker: Optional[Any] = None,
        recent_prog_delta: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Build the normalized (1, 3, 128, 40) float32 tensor from current emulator state.

        Args:
            raw_screen_rgba:   uint8 array (144, 160, 4) from pyboy.screen.ndarray
            telemetry:         snapshot dict from RewardTracker.snapshot()
            visited_count:     unique coordinate tile count
            whidden_tracker:   WhiddenRewardState instance (or tracker)
            recent_prog_delta: optional (d_level, d_hp, d_explore) for recent memory

        Returns:
            torch.Tensor of shape (1, 3, 128, 40), float32 in range [0.0, 1.0]
        """
        # Downsample screen: (144, 160, 4) RGBA -> (36, 40, 3) RGB
        # Use cv2.INTER_AREA (anti-alias area interpolation) to match
        # Whidden's original training distribution. Falls back to numpy
        # stride slicing if cv2 is unavailable (slight activation mismatch).
        rgb = raw_screen_rgba[:, :, :3]  # drop alpha channel
        if _HAS_CV2:
            small = _cv2.resize(rgb, (40, 36), interpolation=_cv2.INTER_AREA)
        else:
            # Fallback: nearest-neighbor via stride (minor distribution shift)
            small = rgb[::4, ::4, :]

        # Roll recent frames forward and insert newest at index 0
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        self.recent_frames[0] = small

        # Update recent reward memory if delta provided
        if recent_prog_delta is not None:
            self.recent_memory = np.roll(self.recent_memory, 3)
            self.recent_memory[0, 0] = min(int(recent_prog_delta[0] * 64), 255)
            self.recent_memory[0, 1] = min(int(recent_prog_delta[1] * 64), 255)
            self.recent_memory[0, 2] = min(int(recent_prog_delta[2] * 128), 255)

        # Exploration memory channels: level, HP, exploration count
        # Scaled exactly to match red_gym_env.py group_rewards()
        lvl_rew = getattr(whidden_tracker, "max_level_rew", 0.0) if whidden_tracker else 0.0
        r_level = lvl_rew * 1000.0
        max_hp  = max(telemetry.get("max_hp", 1), 1)
        r_hp    = (telemetry.get("hp", 0) / max_hp) * 2000.0
        r_exp   = visited_count * 0.005 * 1500.0

        w = 40
        h = 8
        col_steps = self.col_steps

        def make_reward_channel(r_val: float) -> np.ndarray:
            max_r_val = (w - 1) * h * col_steps
            r_val = min(max(r_val, 0.0), max_r_val)
            row = floor(r_val / (h * col_steps))
            mem = np.zeros(shape=(h, w), dtype=np.uint8)
            mem[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            mem[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            mem[col, row] = last_pixel * (255 // col_steps)
            return mem

        full_exp_memory = np.stack([
            make_reward_channel(r_level),
            make_reward_channel(r_hp),
            make_reward_channel(r_exp),
        ], axis=-1)  # (8, 40, 3)

        if telemetry.get("badges", 0) > 0:
            full_exp_memory[:, -1, :] = 255

        pad = np.zeros((self.mem_padding, w, 3), dtype=np.uint8)  # (2, 40, 3)
        recent_mem = self.recent_memory.reshape(h, w, 3)           # (8, 40, 3)
        frames_stacked = self.recent_frames.reshape(108, w, 3)     # (108, 40, 3)

        # Full concatenated observation: 8 + 2 + 8 + 2 + 108 = 128 rows
        obs_128 = np.concatenate([
            full_exp_memory,
            pad,
            recent_mem,
            pad,
            frames_stacked,
        ], axis=0)  # (128, 40, 3) uint8

        # Normalize to float32 [0.0, 1.0] and permute to (1, 3, 128, 40)
        obs_float = obs_128.astype(np.float32) / 255.0
        obs_t     = torch.from_numpy(obs_float).permute(2, 0, 1).unsqueeze(0)
        return obs_t

    def reset(self):
        """Reset internal frame buffers upon environment reset."""
        self.recent_memory.fill(0)
        self.recent_frames.fill(0)
