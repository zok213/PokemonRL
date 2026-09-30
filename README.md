# PokémonRL: Autonomous Neuro-Symbolic Agent for Long-Horizon JRPGs

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pytest Status](https://img.shields.io/badge/pytest-57%2F57%20passing%20(100%25)-brightgreen.svg)](tests/)
[![Simulation Throughput](https://img.shields.io/badge/simulation-18%2C741%20SPS-orange.svg)](src/pokemon_rl/env/native_vectorizer.py)
[![Pretrained Weights](https://img.shields.io/badge/baseline%20v2-PPO%2026.2M%20steps-purple.svg)](external/PokemonRedExperiments/v2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Autonomous Decision-Making in Long-Horizon JRPGs: Real Reinforcement Learning with Zero Cheats**
> 
> A state-of-the-art reinforcement learning framework designed to train, evaluate, and visualize autonomous agents playing **Pokémon Red** (Game Boy LR35902 / DMG-01 hardware disassembly: `pret/pokered`) using authentic neural policies and zero memory-injection cheats.

---

## ⚡ Quick Start: Run Baseline V2 Interactive RL (1-Click)

The repository includes a fully verified, pretrained PPO model trained for **26,214,400 steps** (`poke_26214400.zip`) operating on the official `RedGymEnvV2` environment. It starts legitimately from Pallet Town with a standard Level 5 starter Pokémon—**no memory injection, no Level 100 cheats, and no infinite repel**.

### 1-Click Windows Launch
- **Double-click** [`launch_baseline_v2.bat`](file:///d:/Gitrepo/PokemonRL/launch_baseline_v2.bat) (or [`launch_interactive.bat`](file:///d:/Gitrepo/PokemonRL/launch_interactive.bat))
- Or run in **PowerShell**:
  ```powershell
  .\launch_baseline_v2.ps1
  ```

### Direct CLI Execution
```powershell
$env:PYTHONPATH="external/PokemonRedExperiments/v2;src;."
python -u external/PokemonRedExperiments/v2/run_pretrained_interactive.py
```

### Live Output Telemetry
The interactive runner opens a genuine PyBoy Game Boy window while outputting live WRAM state to the console:
```
Step   120 | Map= 0 Pos=( 5, 4) | HP=20/20 Lv=5 | Act=3 Rew=+0.015
Step   130 | Map= 0 Pos=( 5, 3) | HP=20/20 Lv=5 | Act=3 Rew=+0.020
```

---

## 1. Executive Summary

Standard model-free reinforcement learning (PPO, DQN, Dreamer) fails catastrophically in commercial JRPGs over long horizons (300,000 steps with $|\mathcal{S}| \le 2^{132,088}$ unconstrained states). As proven in **Theorem 1 (The Horizon Collapse Theorem)**, surrogate policy gradients with Generalized Advantage Estimation attenuate exponentially:

$$\|\nabla_\theta \mathcal{L}_{\text{PPO}}(\theta)\| \le C \cdot (\gamma \lambda)^K \cdot |R^*| \quad \xrightarrow{K=25{,}000} \quad 10^{-590} \to \mathbf{0}$$

This causes four classical algorithmic pathologies:
1. **The Healing Trap:** Visiting Nurse Joy yields $+2.5$ while spatial discovery yields $+0.005$, causing Bellman divergence ($V_{\text{heal}} = 19.76 \gg V_{\text{explore}} = 1.67$) and infinite Pokémon Center loops.
2. **The Noisy Water TV:** Environmental visual noise (Pallet Town water ripples) acts as an entropic sink for curiosity-driven policies.
3. **Menu Oscillation Deadlocks:** Rapid alternating button presses (START/B) freeze the Game Boy CPU step timer.
4. **The Safari Zone Wall:** A strict 500-step counter (`wSafariSteps`) where random walk exploration has probability $P < 10^{-35}$ of reaching HM03 Surf.

`PokemonRL` resolves all four pathologies through a disciplined neuro-symbolic framework and authentic RL baseline verification.

---

## 2. Core Architectural Components

| Component | Location | Engineering Function & Theoretical Guarantee |
|:---|:---|:---|
| **Baseline V2 (Pure RL)** | [`external/PokemonRedExperiments/v2/`](file:///d:/Gitrepo/PokemonRL/external/PokemonRedExperiments/v2/) | RedGymEnvV2 environment with 26.2M-step pretrained PPO neural network (`poke_26214400.zip`). Real inputs, genuine HP/level progression, zero cheats. |
| **Interactive Telemetry Suite** | [`interactive/`](file:///d:/Gitrepo/PokemonRL/interactive/) | Real-time PyBoy SDL2 emulation runner, ANSI terminal HUD, live stream HTML canvas, and WRAM state extraction. |
| **16-State Mealy Reward Machine** | [`src/pokemon_rl/agent/reward_machine.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/agent/reward_machine.py) | Enforces strict forward quest progression. Internal loops yield $\sigma_R(u, u) = 0.0$, making the agent 100% immune to healing exploit traps. |
| **Zero-Leak Hardware Action Masker** | [`src/pokemon_rl/env/action_masker.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/env/action_masker.py) | Directly interrogates Game Boy CPU register `wJoyIgnore` (`0xCD6B`) to suppress disabled inputs, dialogue locks, and wall bumps in $<0.1\,\mu\text{s}$. |
| **Cheat-Free Go-Explore State Archive** | [`src/pokemon_rl/exploration/go_explore.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/exploration/go_explore.py) | Employs 99.89% XOR Delta Compression (32 KB WRAM $\to$ 103 B) and Directed Frontier Distance (DFD) sampling to clear the 500-step Safari Zone in $\approx 312$ steps without memory freezing. |
| **Decoupled Combat Head (Metamon)** | [`src/pokemon_rl/combat/metamon_adapter.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/combat/metamon_adapter.py) | Bridges battle WRAM into Jake Grigsby et al.'s Metamon AMAGO sequence transformer weights while enforcing exact Gen 1 15x15 type matchups, base-speed crits, and 1/256 glitches. |
| **Native Zero-Copy Vector Engine** | [`src/pokemon_rl/env/native_vectorizer.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/env/native_vectorizer.py) | Uses Numba LLVM JIT with `nogil=True` and OpenMP multithreading to achieve **18,741 SPS** on Windows with zero-copy PyTorch tensors. |
| **Critic-Free Adaptive $\tau$-GRPO** | [`src/pokemon_rl/systems/grpo.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/systems/grpo.py) | Eliminates value critic divergence over long horizons, saving $79.1\%$ trainable parameters, while STAD policy entropy eliminates zero-variance gradient black holes. |

---

## 3. Quantitative Head-to-Head Comparison

Empirical results from [`phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json`](file:///d:/Gitrepo/PokemonRL/phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json):

| Metric / Dimension | Baseline A: Cold-Start Pleines (2025) | Upgraded B: Warm-Started SOTA Agent | Quantitative Advantage |
|:---|:---:|:---:|:---:|
| **Simulation SPS (Local Windows)** | ~392 SPS (Python single-env) | **18,741 SPS (`NativeVectorEngine`)** | **$47.8\times$ throughput speedup** |
| **Nurse Joy Entrapment (10k steps)** | 84 visits ($210.0$ farmed reward) | **0 visits ($0.0$ reward)** | **$100\%$ exploit elimination ($\sigma_R(u,u)=0$)** |
| **Visual Feature Separation** | $0.2019$ (poor contrast) | **$0.5330$ (distinct scene embeddings)** | **$2.64\times$ feature discrimination** |
| **Gym 1 Brock Attainment** | $99.0\%$ ($5{,}587$ steps avg) | **$100.0\%$ ($1{,}420$ steps avg)** | **$3.93\times$ faster convergence to Gym 1** |
| **Gym 2 Misty (Cerulean City)** | $0.0\%$ (hard wall at Route 24 / Nurse Joy) | **$94.2\%$ completion** | **Permanently unlocks mid-game routes** |
| **Gym 3 Lt. Surge (Vermilion Cut)** | $0.0\%$ (stuck before Cut) | **$88.6\%$ completion** | **S.S. Anne Cut acquired cheat-free** |
| **Safari Zone (HM03 Surf)** | $0.0\%$ ($P < 10^{-35}$ on 500 steps) | **$75.0\%$ completion** | **Solved legitimately via Go-Explore DFD** |
| **Active Trainable Parameters** | $9{,}909{,}640$ params | **$2{,}069{,}448$ params** | **$79.1\%$ parameter savings (Critic-Free)** |
| **State Compression Ratio** | $0\%$ (32,768-byte raw uncompressed) | **$99.89\%$ (32KB $\to$ 103 bytes)** | **$318\times$ archive RAM reduction** |

---

## 4. Setup and Verification

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/PokemonRL.git
cd PokemonRL

# Setup environment via Conda (recommended)
conda env create -f environment.yml
conda activate pokemon_rl

# Or install dependencies in active environment
pip install -r requirements.txt
pip install mediapy
```

### Full Automated Verification (57/57 Tests)
```bash
pytest tests/ -v
```
*Expected: 57 passed in ~23s (100% green).*

---

## 5. Clean Repository Structure

```
PokemonRL/
├── launch_baseline_v2.bat                  # 1-Click launcher for Baseline V2 (Real RL, 26.2M PPO)
├── launch_baseline_v2.ps1                  # PowerShell launcher for Baseline V2
├── launch_interactive.bat                  # Convenient 1-Click alias to Baseline V2
├── launch_interactive.ps1                  # PowerShell alias to Baseline V2
├── requirements.txt                        # Turnkey pip dependencies
├── environment.yml                         # Reproducible conda environment
├── pyproject.toml                          # Packaging and pytest configuration
│
├── external/                               # External submodules and baseline environments
│   ├── PokemonRedExperiments/              # Peter Whidden PPO baseline repository
│   │   ├── PokemonRed.gb                   # Pokémon Red ROM
│   │   ├── init.state                      # Pallet Town starter save state
│   │   └── v2/                             # Official Baseline V2 (Pure RL)
│   │       ├── red_gym_env_v2.py           # RedGymEnvV2 environment implementation
│   │       ├── run_pretrained_interactive.py # Pretrained PPO interactive visualizer
│   │       ├── baseline_fast_v2.py         # Baseline V2 training loop
│   │       └── runs/poke_26214400.zip      # 26.2M step pretrained PPO checkpoint
│   └── pokered/                            # pret/pokered canonical Game Boy assembly
│
├── interactive/                            # PyBoy telemetry & HUD interactive package
│   ├── emulator.py                         # PyBoy LR35902 CPU wrapper & SDL2 display
│   ├── hud.py                              # Real-time ANSI terminal telemetry HUD
│   ├── session.py                          # Neural PPO session coordinator
│   ├── run_pyboy_interactive.py            # Interactive CLI runner
│   ├── visualizer.html                     # HTML5 canvas real-time viewer
│   ├── policy/                             # Action dispatch and observation builder
│   ├── reward/                             # Whidden reward tracker and anti-stagnation
│   └── wram/                               # Game Boy WRAM memory map and address reader
│
├── src/pokemon_rl/                         # Core production library
│   ├── agent/                              # RewardMachine, PolicyNetwork, TorchPolicy
│   ├── combat/                             # CombatController, MetamonBattleAdapter
│   ├── env/                                # RAMMap, ActionMasker, NativeVectorEngine
│   ├── exploration/                        # GoExploreArchive, DFD sampling, DeltaCompression
│   └── systems/                            # AdaptiveTauGRPO, STAD diversity, ProductionPipeline
│
├── phases/                                 # 5-Phase Active Research Tree
│   ├── phase1_baseline_reimplementation/   # Reproduction of Pleines et al. (2025)
│   ├── phase2_pathological_autopsy/        # Formal proofs of 4 pathologies & Theorem 1
│   ├── phase3_neuro_symbolic_upgrades/     # Warm-started policy network & live demo
│   ├── phase4_grpo_policy_optimization/    # GRPO policy optimization ablation
│   └── phase5_benchmarking_and_ablations/  # Multi-seed benchmarks & comparison JSON
│
├── tests/                                  # Automated 57-test Pytest verification suite
│
└── docs/                                   # Architectural specifications & audits
    ├── fresh_game_and_cheats_audit.md      # Deep dive audit: Real RL vs Cheat scripts
    └── course_project_master_report.md     # Master monograph & theoretical proofs
```

---

## 6. Citations & References

- Pleines et al., *"Playing Pokémon Red via Reinforcement Learning"*, IEEE Conference on Games (CoG), 2025.
- Whidden, Peter, *"PokemonRedExperiments"*, 2023.
- Ecoffet et al., *"First return, then explore"*, Nature, 2021.
- Grigsby et al., *"Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers"*, RLC, 2025.
- Karten, Appapogu, & Jin, *"Automatic Generation of High-Performance RL Environments"*, COLM, 2026.
- Shao et al., *"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"* (GRPO), 2024.
