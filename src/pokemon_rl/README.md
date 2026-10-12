# pokemon_rl — Core Production Architecture

The `pokemon_rl` Python package contains the production-grade implementation of the neuro-symbolic JRPG decision system. It is designed to be fully modular, installable, and independent of GUI or emulator windowing.

```
src/pokemon_rl/
├── __init__.py                # Root package exports
├── agent/                     # Decision engines (V2JevHybridPolicy, RewardMachine, TacticalBattle, LocalAffordancePolicy)
├── combat/                    # Tactical Combat (Gen1CombatController, MetamonBattleAdapter)
├── env/                       # Environment Wrappers (RealPyBoyEnvironment, RAMMap, ActionMasker, NativeVectorEngine)
├── exploration/               # Long-Horizon Archives (GoExploreArchive, DFD Sampling, DeltaCompression)
├── navigation/                # Topological Routing (WorldGraphNavigator, OccupancyGridRouter)
├── systems/                   # Optimization & Orchestration (AdaptiveTauGRPO, STAD, ProductionPipeline)
└── visualizer/                # Live Emulation Servers (PokemonRLVisualizerServer, JevosServer, Web Dashboards)
```

## Quick Import Guide

```python
# 1. Flagship Hybrid: Baseline V2 Weights + Jev Affordances
from pokemon_rl.agent.v2_jev_hybrid_policy import BaselineV2JevSteeredPolicy

# 2. Tactical Combat & Topological Navigation
from pokemon_rl.agent.tactical_battle import TacticalBattleController
from pokemon_rl.navigation.world_graph import WorldGraphNavigator
from pokemon_rl.navigation.occupancy_router import OccupancyGridRouter

# 3. Local Affordance Policy & SMDP Harness
from pokemon_rl.agent.local_decision_policy import LocalAffordancePolicy, SMDPDecisionHarness

# 4. Authentic Hardware Emulation & Live Server
from pokemon_rl.env.real_pyboy_env import RealPyBoyEnvironment
from pokemon_rl.visualizer.pokemonrl_server import PokemonRLVisualizerServer
```
