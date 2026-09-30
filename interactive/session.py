"""
interactive/session.py — Game Boy RL Session Orchestrator
=========================================================
Thin coordinator that wires together:
  - GameBoyEmulator (PyBoy hardware loop + SDL2 window)
  - RewardTracker   (Whidden-aligned reward + anti-stagnation)
  - WhiddenObservationBuilder (frame stack + memory bar tensor)
  - ActionDispatcher (mode detection, masking, dispatch)
  - TerminalHUD (live ANSI terminal display)
  - WhiddenPretrainedPolicy (439M pretrained neural network)

All logic lives in the submodules. This file is ~100 lines of wiring.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import numpy as np

from pokemon_rl.agent.torch_policy import WhiddenPretrainedPolicy

from interactive.emulator import GameBoyEmulator
from interactive.reward.tracker import RewardTracker
from interactive.policy.observation import WhiddenObservationBuilder
from interactive.policy.action_dispatch import ActionDispatcher
from interactive.hud import TerminalHUD


class InteractiveGameBoyRLSession:
    """
    Wires emulator + policy + reward + display into a real-time loop.
    Runs pure neural RL policy with hardware action masking and live telemetry.
    """

    def __init__(
        self,
        rom_path:             str | Path,
        state_path:           Optional[str | Path] = None,
        checkpoint_path:      Optional[str | Path] = None,
        headless:             bool = False,
        action_freq:          int  = 24,
        emulation_speed:      int  = 16,
        scale:                int  = 3,
        use_action_masking:   bool = True,
        use_pbrs:             bool = True,
        initial_agent_enabled:bool = True,
        mode:                 str  = "ppo",
    ):
        self.rom_path       = Path(rom_path)
        self.state_path     = Path(state_path) if state_path else None
        self.checkpoint_path= Path(checkpoint_path) if checkpoint_path else None
        self.headless       = headless
        self.emulation_speed= emulation_speed
        self.agent_enabled  = initial_agent_enabled
        self.step_count     = 0
        self.mode           = mode.lower()

        # --- Emulator ---
        self.emulator = GameBoyEmulator(
            rom_path=self.rom_path,
            state_path=self.state_path,
            headless=self.headless,
            emulation_speed=emulation_speed,
            scale=scale,
        )

        # --- Reward Tracker ---
        self.tracker = RewardTracker()

        # --- Observation Builder ---
        self.obs_builder = WhiddenObservationBuilder()

        # --- Action Dispatcher ---
        self.dispatcher = ActionDispatcher(use_action_masking=use_action_masking)

        # --- Terminal HUD ---
        self.hud = TerminalHUD(emulation_speed=emulation_speed)

        # --- Policy / Controller ---
        print(f"[*] Mode: NEURAL PPO (Policy ckpt: {self.checkpoint_path})")
        self.policy = WhiddenPretrainedPolicy(checkpoint_path=self.checkpoint_path)
        self.policy.eval()

        # Initial snapshot
        snap = self.tracker.snapshot(self.emulator.memory)
        print(f"[+] Initial State Ready: {snap['map_name']} (X={snap['x']}, Y={snap['y']})\n")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(self, max_steps: int = 0):
        """Run until max_steps reached (0 = indefinitely / window closed)."""
        print(f"[*] Running {'indefinitely' if max_steps == 0 else f'{max_steps:,} steps'} at {self.emulation_speed}x...")
        try:
            while max_steps <= 0 or self.step_count < max_steps:
                self._check_agent_toggle()

                if self.agent_enabled:
                    alive = self._agent_step()
                    if alive is False:
                        break
                else:
                    # Manual human control — just tick emulator
                    if not self.emulator.tick(1, True):
                        break
                    time.sleep(0.016)

        except KeyboardInterrupt:
            print("\n[*] Session interrupted by user.")
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Single decision step
    # ------------------------------------------------------------------

    def _agent_step(self) -> bool:
        mem = self.emulator.memory
        snap = self.tracker.snapshot(mem)
        raw_screen = self.emulator.get_screen_ndarray()
        obs_t = self.obs_builder.build(
            raw_screen, snap, self.tracker.visited_count, self.tracker._whidden
        )
        action_idx, probs, subsystem_str = self.dispatcher.select_action(
            mem, self.policy, obs_t, self.tracker
        )
        alive = self.dispatcher.dispatch(self.emulator.pyboy, action_idx)
        step_reward = self.tracker.step(self.emulator.memory)
        self.step_count += 1

        if not alive:
            return False

        if self.step_count % 2 == 0:
            post_snap = self.tracker.snapshot(self.emulator.memory)
            self.hud.render(
                step_count              = self.step_count,
                telemetry               = post_snap,
                action_idx              = action_idx,
                probs                   = probs,
                step_reward             = step_reward,
                cumulative_reward       = self.tracker.cumulative_reward,
                visited_count           = self.tracker.visited_count,
                subsystem_str           = subsystem_str,
                agent_enabled           = self.agent_enabled,
                reward_matrix           = self.tracker.reward_matrix,
                reward_matrix_cumulative= self.tracker.reward_matrix_cumulative,
                rm_state                = self.tracker.reward_machine.current_state,
                objective_desc          = self.tracker.active_objective_desc,
                dist_to_goal            = self.tracker.dist_to_goal,
            )
        return True

    def _check_agent_toggle(self):
        """Check agent_enabled.txt for runtime toggle."""
        toggle_file = Path("agent_enabled.txt")
        if toggle_file.exists():
            try:
                with open(toggle_file) as f:
                    self.agent_enabled = f.read().strip().lower().startswith("yes")
            except Exception:
                pass

    # ------------------------------------------------------------------

    def close(self):
        self.emulator.close()
        print("=" * 74)
        print("  SESSION SUMMARY")
        print(f"    Steps:           {self.step_count:,}")
        print(f"    Unique Tiles:    {self.tracker.visited_count:,}")
        print(f"    Total Reward:    {self.tracker.cumulative_reward:+.4f}")
        print("=" * 74)
