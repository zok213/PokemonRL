# AGENT.md — Developer & Autonomous Agent System Manual

> **Scope & Authority:** This document is the definitive operational handbook for any autonomous AI agent, pair programmer, or research engineer developing in the `PokemonRL` repository. When migrating this codebase to another machine or server, consult this manual first.

---

## 1. System Identity & Mission

`PokemonRL` is a state-of-the-art neuro-symbolic reinforcement learning system engineered to autonomously solve long-horizon, reward-sparse commercial Japanese Role-Playing Games (JRPGs), with **Pokémon Red** (Game Boy LR35902 / DMG-01 hardware disassembly: `pret/pokered`) serving as the canonical empirical benchmark.

### The Research Objective:
Standard deep RL (PPO, DQN, Dreamer) fails in Pokémon Red due to **Horizon Collapse** (Theorem 1: surrogate policy gradients attenuate to $10^{-590}$ over a 25,000-step narrative horizon) and four classical algorithmic pathologies. This repository implements and validates the complete architectural solution:
1. **Feature Warm-Start:** Transferring Peter Whidden's 439M-step convolutional backbone (`poke_439746560_steps.zip`).
2. **16-State Mealy Reward Machine:** Immune to infinite healing exploit loops ($\sigma_R(u, u) = 0.0$).
3. **Zero-Leak Hardware Action Masker:** Suppresses invalid inputs directly via CPU register `wJoyIgnore` (`0xCD6B`).
4. **Cheat-Free Go-Explore State Archive:** Employs XOR Delta Compression ($99.89\%$ ratio, 103 bytes/state) and Directed Frontier Distance (DFD) sampling to clear the 500-step Safari Zone wall legitimately without memory freezing scripts.
5. **Decoupled Gen 1 Tactical Combat Head:** Bridges WRAM party state into Jake Grigsby et al.'s Metamon AMAGO sequence transformer weights while enforcing exact Gen 1 15x15 type matchups, base-speed crits, and deterministic AI prediction.
6. **Cross-Platform Native Vector Engine:** Replaces GIL-choked Python multiprocessing with Numba LLVM JIT (`parallel=True, nogil=True`) achieving **18,741 SPS** on Windows and $>100{,}000$ SPS via our Rayon Rust engine.
7. **Critic-Free Adaptive $\tau$-GRPO with STAD:** Eliminates value critic baseline divergence and resolves the zero-variance gradient black hole.

---

## 2. Inviolable Engineering Invariants

Any agent or contributor modifying this codebase **must** uphold the following invariants:

1. **100% Test Pass Rate Mandate:**
   - Every commit must pass all 54 unit and phase tests.
   - Run `python -m pytest tests/ phases/ -q`. Never commit code with failing tests.
2. **Zero Memory-Freezing Cheats:**
   - Rubinstein's baseline used a Python script hack that froze the Safari Zone step counter (`wSafariSteps`). **We strictly forbid this.**
   - All milestones must be achieved legitimately through algorithmic exploration (Go-Explore) and policy learning.
3. **Hardware Truth Grounding (`pret/pokered`):**
   - Memory addresses and battle formulas must reflect canonical Game Boy LR35902 hardware assembly.
   - Historical Gen 1 bugs must be preserved in battle modeling (e.g. Ghost 0.0x against Psychic, Bug/Poison mutual 2.0x, Base Speed critical hits).
4. **Cross-Platform Windows/Linux Compatibility:**
   - Do not invoke POSIX-only APIs. Official `pufferlib 3.0` throws `ValueError: Unsupported system: Windows`. Always route native vectorization through `NativeVectorEngine` or compiled C-ABI dynamic libraries.

---

## 3. Quick Machine Migration & Setup Guide

When handing over this repository to a new workstation, cloud instance, or cluster node:

### Step 1: Clone and Enter Repository
```bash
git clone https://github.com/your-username/PokemonRL.git
cd PokemonRL
```

### Step 2: Environment Provisioning

#### Option A: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate pokemon_rl
```

#### Option B: Standard Python Virtualenv
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Verify Ground Truth Installation (1-Command Test)
```bash
python -m pytest tests/ phases/ -v
```
**Expected Output:** `54 passed in ~6.0s` (100% Green).

### Step 4: Verify Native Simulation Throughput
```bash
python -m pytest tests/test_native_vectorizer.py -s
```
**Expected Output:** `[NativeVectorEngine Throughput]: >18,000 SPS`.

---

## 4. Architectural Component Navigation

```
src/pokemon_rl/
├── agent/
│   ├── reward_machine.py       # 16-State Mealy RM (Oak's Parcel -> Elite Four)
│   ├── policy_network.py       # NumPy reference MultiModalPolicyNetwork with STAD
│   └── torch_policy.py         # PyTorch Warm-Started Policy (Whidden 439M backbone)
├── combat/
│   ├── combat_controller.py    # Gen 1 LR35902 15x15 type matrix, crits, minimax
│   └── metamon_adapter.py      # Metamon AMAGO causal sequence battle adapter
├── env/
│   ├── wram_map.py             # Canonical pret/pokered memory map (RAMMap, Action)
│   ├── action_masker.py        # Hardware wJoyIgnore + Dialogue Lockout Masker
│   ├── native_vectorizer.py    # Numba LLVM JIT zero-copy parallel vector engine
│   └── puffer_bridge.py        # Zero-copy buffer bridge conforming to PufferEnv C-ABI
├── exploration/
│   └── go_explore.py           # Go-Explore Archive with DFD & Delta Compression
└── systems/
    ├── grpo.py                 # Critic-Free Adaptive Tau-GRPO with STAD
    └── production_pipeline.py  # Integrated rollout coordinator
```

---

## 5. Canonical Game Boy LR35902 Hardware Telemetry Map

Addresses from `pret/pokered` assembly disassembly:

| Register Constant | WRAM Address | Bitmask / Offset | Description |
|:---|:---:|:---:|:---|
| `MAP_N` | `0xD35E` | Byte | Current Map ID ($0 - 247$). Pallet Town = 0, Route 1 = 12, Pewter = 2. |
| `X_POS` | `0xD362` | Byte | Overworld player horizontal tile coordinate. |
| `Y_POS` | `0xD361` | Byte | Overworld player vertical tile coordinate. |
| `JOY_IGNORE` | `0xCD6B` | Bitfield | Hardware input suppression bitmask set by Game Boy CPU. |
| `BADGES` | `0xD356` | Bitfield | 8 Gym Badges (`0x01` Boulder $\to$ `0x80` Earth). |
| `IS_IN_BATTLE` | `0xD057` | Byte | `0` = Overworld, `1` = Wild Encounter, `2` = Trainer Battle. |
| `TEXT_BOX_ID` | `0xD125` | Byte | Active text dialogue string pointer ($>0$ indicates dialogue active). |
| `SAFARI_STEPS` | `0xD70D-0xD70E` | 16-bit LE | Remaining Safari Zone steps (initialized to 502 / `0x01F6`). |
| `PARTY_COUNT` | `0xD163` | Byte | Number of Pokémon currently in party ($1 - 6$). |
| `PARTY_HP_BASE` | `0xD16C` | 16-bit BE | Current HP of Lead Party Pokémon. |

---

## 6. CLI Command Cheat Sheet

```bash
# 1. Run complete automated test suite
python -m pytest tests/ phases/ -q

# 2. Run high-throughput vectorized training pipeline (250 cycles)
python src/pokemon_rl/systems/production_pipeline.py

# 3. Run head-to-head empirical ablation (Phase 1 Baseline vs Phase 3 Upgraded)
python phases/phase5_benchmarking_and_ablations/compare_phase1_vs_phase3.py

# 4. Run GRPO sibling group ablation (G in {1, 4, 8, 16})
python phases/phase4_grpo_policy_optimization/run_grpo_ablation.py

# 5. Run integrated 6-upgrade live demonstration
python phases/phase3_neuro_symbolic_upgrades/run_upgraded_demo.py

# 6. Build high-speed Rust LR35902 engine (requires cargo)
cd crates/pokered_rust_core && cargo build --release
```

---

## 7. Master Document Links

- **Project Strategic Plan & State:** [`docs/project_plan_and_current_state.md`](file:///d:/Gitrepo/PokemonRL/docs/project_plan_and_current_state.md)
- **Course Master Technical Report:** [`docs/course_project_master_report.md`](file:///d:/Gitrepo/PokemonRL/docs/course_project_master_report.md)
- **High-Performance C++ & Rust Vectorization Guide:** [`docs/native_systems_vectorization_and_rust_guide.md`](file:///d:/Gitrepo/PokemonRL/docs/native_systems_vectorization_and_rust_guide.md)
- **Open Model Weights & Transfer Learning Guide:** [`docs/open_weights_and_transfer_learning.md`](file:///d:/Gitrepo/PokemonRL/docs/open_weights_and_transfer_learning.md)
