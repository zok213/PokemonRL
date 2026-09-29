"""
pokemon_rl — Autonomous JRPG Agent Research Package
=====================================================
Unified neuro-symbolic blueprint for long-horizon JRPG completion.

Paradigms implemented:
  1. Model-Free Deep RL + Novelty Exploration (Go-Explore)
  2. Hierarchical RL + Action Masking (wJoyIgnore)
  3. Critic-Free GRPO with STAD variance injection
  4. Offline Combat Transformer (Metamon proxy)
  5. 16-State Reward Machine (Healing Trap immune)

Theoretical anchors:
  - Theorem 1: Horizon Collapse (PPO fails at K=25,000 steps)
  - Theorem 2: PBRS Policy Invariance (potential shaping is safe)
  - Lemma: Zero-Variance Black Hole → STAD resolution
"""

__version__ = "0.1.0"
__author__  = "PokemonRL Research"

from pokemon_rl.env.wram_map import RAMMap, Action
from pokemon_rl.agent.reward_machine import RewardMachine, RM_STATES, RM_WIN_STATE

__all__ = [
    "RAMMap",
    "Action",
    "RewardMachine",
    "RM_STATES",
    "RM_WIN_STATE",
]
