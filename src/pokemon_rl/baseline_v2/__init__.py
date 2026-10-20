"""
baseline_v2 — Full Authentic Re-Implementation of Pleines et al. (IEEE CoG 2025)
==================================================================================
Conforming to:
  Marco Pleines, Matthias Addis, David Rubinstein, Frank Zimmer, Mike Preuss, Peter Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920 / PWhiddy V2

Submodules:
  - constants: Hardware WRAM addresses, action space, paper hyperparameters, Table 3 ground-truth.
  - reward: 4-term composite linear reward (R_event + R_nav + R_heal + R_lvl).
  - env: Gym-compatible Joypad cadence environment (8-hold, 16-release) with multimodal observations.
  - models: PyTorch & NumPy Actor-Critic policies (Nature CNN, Spatial CNN, Telemetry MLP, GAE).
  - reproduce: Complete Table 3 ablation replication engine.
  - evaluate: Multi-episode benchmarking and evaluation CLI.
"""

from pokemon_rl.baseline_v2.constants import (
    PleinesAction,
    PleinesWRAM,
    FRAME_HOLD,
    FRAME_RELEASE,
    DECISION_STRIDE,
    BASE_BUDGET,
    EVENT_BONUS,
    WEIGHT_EVENT,
    WEIGHT_NAV,
    WEIGHT_HEAL,
    WEIGHT_LVL,
    GAMMA,
    LAMBDA_GAE,
    CLIP_RANGE,
    PLEINES_TABLE3_GROUND_TRUTH,
)

from pokemon_rl.baseline_v2.reward import (
    PleinesCompositeReward,
    compute_level_potential,
    compute_event_reward,
    compute_nav_reward,
    compute_heal_reward,
    compute_level_reward,
)

from pokemon_rl.baseline_v2.env import PleinesBaselineEnv

from pokemon_rl.baseline_v2.models import (
    PleinesActorCriticNumPy,
    compute_gae,
    HAS_TORCH,
)

if HAS_TORCH:
    from pokemon_rl.baseline_v2.models import (
        PleinesActorCriticTorch,
        PleinesNatureCNN,
        PleinesSpatialCNN,
        PleinesTelemetryMLP,
    )

# Lazy imports for CLI modules to prevent execution-time circular warnings
def __getattr__(name: str):
    if name in ("run_ablation_experiment", "reproduce_all_table3_variants"):
        from pokemon_rl.baseline_v2 import reproduce
        return getattr(reproduce, name)
    elif name == "run_evaluation_episode":
        from pokemon_rl.baseline_v2 import evaluate
        return getattr(evaluate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PleinesAction",
    "PleinesWRAM",
    "FRAME_HOLD",
    "FRAME_RELEASE",
    "DECISION_STRIDE",
    "BASE_BUDGET",
    "EVENT_BONUS",
    "WEIGHT_EVENT",
    "WEIGHT_NAV",
    "WEIGHT_HEAL",
    "WEIGHT_LVL",
    "GAMMA",
    "LAMBDA_GAE",
    "CLIP_RANGE",
    "PLEINES_TABLE3_GROUND_TRUTH",
    "PleinesCompositeReward",
    "compute_level_potential",
    "compute_event_reward",
    "compute_nav_reward",
    "compute_heal_reward",
    "compute_level_reward",
    "PleinesBaselineEnv",
    "PleinesActorCriticNumPy",
    "compute_gae",
    "HAS_TORCH",
    "run_ablation_experiment",
    "reproduce_all_table3_variants",
    "run_evaluation_episode",
]

if HAS_TORCH:
    __all__.extend([
        "PleinesActorCriticTorch",
        "PleinesNatureCNN",
        "PleinesSpatialCNN",
        "PleinesTelemetryMLP",
    ])
