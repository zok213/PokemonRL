# pokemon_rl — Core Production Architecture

The `pokemon_rl` Python package contains the production-grade implementation of the neuro-symbolic JRPG decision system. It is designed to be fully modular, installable, and independent of GUI or emulator windowing.

```
src/pokemon_rl/
├── __init__.py                # Root package exports
├── agent/                     # Decision engines (RewardMachine, MultiModalPolicy, TorchPolicy)
├── combat/                    # Tactical Combat (Gen1CombatController, MetamonBattleAdapter)
├── env/                       # Environment Wrappers (RAMMap, ActionMasker, NativeVectorEngine, PufferBridge)
├── exploration/               # Long-Horizon Archives (GoExploreArchive, DFD Sampling, DeltaCompression)
└── systems/                   # Optimization & Orchestration (AdaptiveTauGRPO, STAD, ProductionPipeline)
```

## Quick Import Guide

```python
from pokemon_rl.agent import RewardMachine, MultiModalPolicyNetwork, WarmStartedPolicyNetwork
from pokemon_rl.combat import Gen1CombatController, MetamonBattleAdapter
from pokemon_rl.env import RAMMap, Action, DynamicActionMasker, NativeVectorEngine
from pokemon_rl.exploration import GoExploreArchive
from pokemon_rl.systems import AdaptiveTauGRPO, ProductionAgentPipeline
```
