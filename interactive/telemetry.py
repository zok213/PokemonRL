"""
interactive/telemetry.py — DEPRECATED SHIM
===========================================
This module is no longer the primary implementation.
Logic has been split into:
  - interactive/wram/addresses.py    — all WRAM addresses
  - interactive/wram/reader.py       — low-level memory read helpers
  - interactive/reward/whidden.py    — Whidden-aligned reward computation
  - interactive/reward/anti_stagnation.py — stagnation & menu penalty
  - interactive/reward/tracker.py    — unified RewardTracker (main API)
  - interactive/policy/observation.py— observation tensor builder
  - interactive/policy/action_dispatch.py — action selection & dispatch

This file is kept for backward compatibility with any external imports.
New code should import from the submodules directly.
"""

# Re-export the main class under the old name for backward compat
from interactive.reward.tracker import RewardTracker as WRAMTelemetryTracker

__all__ = ["WRAMTelemetryTracker"]
