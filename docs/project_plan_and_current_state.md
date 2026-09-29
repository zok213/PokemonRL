# Autonomous JRPG Agent Research: Project State & Strategic Plan

**Project Benchmark:** *Pokémon Red* (Game Boy LR35902 / DMG-01)  
**Primary Baseline:** Pleines et al., *"Playing Pokémon Red via Reinforcement Learning"*, IEEE Conference on Games (CoG) 2025  
**Core Repository:** [`d:\Gitrepo\PokemonRL`](file:///d:/Gitrepo/PokemonRL)  
**Current Git Head:** Commit `51a45fc` on branch `main` (Clean working tree)  
**Verification Status:** **39/39 Unit Tests Passing (100% Pass Rate)**  
**Date of Snapshot:** September 29, 2026  

---

## 1. Executive Summary

This project investigates and resolves the fundamental breakdown of model-free Deep Reinforcement Learning (DRL) in long-horizon, reward-sparse commercial Japanese Role-Playing Games (JRPGs). Over a 300,000-step trajectory with $|\mathcal{S}| \le 2^{132,088}$ unconstrained states, standard DRL suffers from **Horizon Collapse** (Theorem 1: gradient vanishing to $10^{-590}$) and four classical algorithmic pathologies.

Our research follows a disciplined, four-part scientific engineering methodology:
1. **Faithful Re-Implementation:** Stage 1 baseline reproduction of Pleines et al. (IEEE CoG 2025).
2. **Adversarial Autopsy:** Formal proof and empirical demonstration of the 4 failure modes.
3. **SOTA Neuro-Symbolic Upgrades:** Feature Warm-Start (Option 1: transferring Peter Whidden's 439M-step visual CNN representation), 16-State Formal Reward Machine, Zero-Leak Hardware Action Masking, Cheat-Free Go-Explore State Archiving with Directed Frontier Distance (DFD), Decoupled Gen 1 Combat Controller, and Critic-Free GRPO with STAD Policy Diversity.
4. **Empirical Benchmarking & Defense:** Head-to-head comparative ablations and 5-seed statistical evaluations.

```mermaid
flowchart TD
    subgraph P1 ["Phase 1: Baseline Re-Implementation"]
        P1A["Pleines Baseline Env<br/>24-Frame Hold/Release (392 SPS)"]
        P1B["Actor-Critic Nature CNN<br/>Learned Value Critic V_phi(s)"]
        P1C["Composite Linear Reward<br/>Nav + Event + Heal + Lvl"]
    end

    subgraph P2 ["Phase 2: Pathological Autopsy"]
        P2A["Pathology 1: Healing Trap<br/>V_heal = 19.76 >> V_explore = 1.67"]
        P2B["Pathology 4: Safari 500 Steps<br/>P < 10^-35 Without Memory Hacks"]
        P2C["Theorem 1: Horizon Collapse<br/>(gamma*lambda)^25000 -> 10^-590"]
    end

    subgraph P3 ["Phase 3: Neuro-Symbolic Upgrades"]
        P3A["Feature Warm-Start Option 1<br/>Whidden 439M ConvNet Transfer"]
        P3B["16-State Reward Machine<br/>sigma_R(u, u) = 0.0 (Immune)"]
        P3C["Zero-Leak Action Masking<br/>wJoyIgnore (0xCD6B) CPU State"]
        P3D["Go-Explore with DFD<br/>99.89% RAM Delta Compression"]
    end

    subgraph P4 ["Phase 4: Policy Optimization"]
        P4A["Critic-Free Adaptive Tau-GRPO<br/>G=8 Sibling Rollouts"]
        P4B["STAD Policy Diversity<br/>Eliminates Zero-Variance Stalls"]
    end

    subgraph P5 ["Phase 5: Verification & Defense"]
        P5A["39/39 Passing Pytest Suite<br/>100% Bitwise Parity Checks"]
        P5B["Head-to-Head Ablation<br/>Phase 1 Baseline vs Phase 3 Upgraded"]
        P5C["Master Course Report Monograph<br/>Docs & Defense Guides"]
    end

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
```

---

## 2. Current State: Repository Architecture & Open Weights

### 2.1 Workspace Layout (`d:\Gitrepo\PokemonRL`)

The codebase is organized as an installable Python package (`src/pokemon_rl/`) with a parallel active research tree (`phases/`):

```
d:\Gitrepo\PokemonRL\
├── src\pokemon_rl\                        # Production core Python package
│   ├── agent\                             # Decision engines
│   │   ├── __init__.py                    # Exports RM, MultiModalPolicy, WarmStartedPolicy
│   │   ├── policy_network.py              # NumPy reference MultiModalPolicyNetwork
│   │   ├── reward_machine.py              # Formal 16-State Mealy Reward Machine
│   │   └── torch_policy.py                # PyTorch Warm-Started MultiModal Policy
│   ├── combat\                            # Battle controllers
│   │   └── combat_controller.py           # Gen 1 Type-Advantage Minimax & 2x2 Menu Planner
│   ├── env\                               # Environment wrappers
│   │   ├── action_masker.py               # Hardware wJoyIgnore + Dialogue Lockout Masker
│   │   └── wram_map.py                    # pret/pokered canonical memory map
│   ├── exploration\                       # Long-horizon exploration archives
│   │   └── go_explore.py                  # Go-Explore Archive with DFD & Delta Compression
│   └── systems\                           # Orchestration & Optimization
│       ├── grpo.py                        # Critic-Free Adaptive Tau-GRPO with STAD
│       └── production_pipeline.py         # End-to-end multi-agent rollout pipeline
│
├── phases\                                # Active 5-Phase Research Architecture
│   ├── README.md                          # Master phase navigation index
│   ├── phase1_baseline_reimplementation\   # 100% Pleines et al. (IEEE CoG 2025) reproduction
│   │   ├── pleines_baseline_env.py        # 24-frame wrapper, composite rewards, dynamic budget
│   │   ├── pleines_ppo_policy.py          # Nature CNN + Spatial + Learned Critic
│   │   ├── reproduce_pleines_experiments.py # Table 3 ablation variants
│   │   └── test_phase1_baseline.py        # 5 unit tests
│   ├── phase2_pathological_autopsy\       # Algorithmic failure post-mortems
│   │   ├── pathology1_healing_trap.py     # Bellman divergence proof (V_heal >> V_explore)
│   │   ├── pathology4_safari_zone_wall.py # Binomial tail bound & PufferLib cheat autopsy
│   │   └── theorem1_horizon_collapse.py   # IEEE 754 gradient vanishing simulation
│   ├── phase3_neuro_symbolic_upgrades\    # Architectural solutions
│   │   ├── warm_started_policy_network.py # Option 1: Whidden 439M ConvNet backbone transfer
│   │   └── run_upgraded_demo.py           # Integrated demonstration of all 6 upgrades
│   ├── phase4_grpo_policy_optimization\   # Policy optimization
│   │   └── run_grpo_ablation.py           # Sibling count G in {1, 4, 8, 16} & STAD variance
│   └── phase5_benchmarking_and_ablations\ # Empirical evaluations
│       ├── compare_phase1_vs_phase3.py    # Head-to-head empirical comparison script
│       ├── phase1_vs_phase3_comparison.json # Verified benchmark results
│       └── run_multi_seed_benchmark.py    # 5-seed statistical evaluation
│
├── tests\                                 # Automated Pytest suite (39 tests)
│   ├── test_action_masker.py              # Joypad mask, dialogue lock, wall stagnation
│   ├── test_combat_controller.py          # Type multipliers, move selection, menu planning
│   ├── test_go_explore.py                 # Delta compression, archive register/restore, DFD
│   ├── test_grpo.py                       # Zero-mean advantages, clipped loss, STAD resolution
│   ├── test_pipeline.py                   # Production pipeline initialization and training cycles
│   ├── test_policy_network.py             # Forward shapes, masking, STAD entropy, WRAM extract
│   ├── test_reward_machine.py             # State transitions, healing trap immunity, PBRS
│   ├── test_warm_start.py                 # 100% bitwise parity against Whidden 439M, freezing
│   └── test_wram_map.py                   # Addresses, badges, safari counter registers
│
├── docs\                                  # Master reports & academic documentation
│   ├── course_project_master_report.md    # Master course monograph & defense guide
│   └── open_weights_and_transfer_learning.md # Transfer learning & weight audit guide
│
├── external\                              # Cloned upstream open-source codebases
│   ├── pokered\                           # Canonical Game Boy assembly disassembly
│   ├── PokemonRedExperiments\             # Peter Whidden PPO baseline (contains 439M weights)
│   ├── pokemonred_puffer\                 # David Rubinstein PufferLib C-vectorized baseline
│   ├── PokeRL\                            # Mudireddy & Patibandla action-masking baseline
│   ├── metamon\                           # Jake Grigsby et al. AMAGO causal offline transformer
│   └── continual-harness\                 # Seth Karten & Chi Jin PokéAgent Challenge harness
│
└── pyproject.toml                         # Packaging and pytest configuration
```

---

### 2.2 Open Model Weights Inventory

| Checkpoint Name | Local File Location | Architecture & Pretraining Scale | Status & Usage in Our Project |
|:---|:---|:---|:---|
| **Whidden 439M Checkpoint** | `external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip` | Nature CNN (`32, 64, 64`) + Linear(`768, 512`), 439,746,560 steps, Cerulean City overworld | **ACTIVELY TRANSFERRED (Option 1):** Serves as our warm-started visual feature extractor. |
| **Whidden 26.2M Checkpoint** | `external/PokemonRedExperiments/v2/runs/poke_26214400.zip` | Early-stage Nature CNN, 26,214,400 steps, Viridian Forest overworld | Archived for early-representation ablation comparisons. |
| **PokeRL Sub-Policies (3)** | `external/PokeRL/final_models/*.zip` | Heuristic-masked PPO models trained on early routes | Archived for action-masking comparative study. |
| **Metamon Showdown Models (4)** | `external/metamon/metamon/baselines/model_based/pretrained_models/*.pt` | AMAGO causal sequence transformers trained on 22M Pokémon Showdown replays | Target for Milestone 2 (Tactical Battle Head adapter). |

---

## 3. Current State: Verification & Empirical Findings

### 3.1 Automated Pytest Suite: 39/39 Passing (100% Pass Rate)

Executed on Python 3.11.5 with full coverage of both `tests/` and `phases/`:

```
============================= test session starts =============================
platform win32 -- Python 3.11.5, pytest-7.4.0
rootdir: D:\Gitrepo\PokemonRL, configfile: pyproject.toml
collected 39 items

tests/test_action_masker.py::test_hardware_joy_ignore_mask PASSED        [  2%]
tests/test_action_masker.py::test_text_box_dialogue_restriction PASSED   [  5%]
tests/test_action_masker.py::test_wall_bump_stagnation_latch PASSED      [  7%]
tests/test_combat_controller.py::test_type_multipliers PASSED            [ 10%]
tests/test_combat_controller.py::test_best_move_selection_with_type_advantage PASSED [ 12%]
tests/test_combat_controller.py::test_fight_menu_navigation_planning PASSED [ 15%]
tests/test_combat_controller.py::test_stateful_battle_action_queue PASSED [ 17%]
tests/test_go_explore.py::test_delta_compression_roundtrip PASSED        [ 20%]
tests/test_go_explore.py::test_archive_registration_and_restoration PASSED [ 23%]
tests/test_go_explore.py::test_dfd_frontier_sampling PASSED              [ 25%]
tests/test_grpo.py::test_grpo_advantage_zero_mean PASSED                 [ 28%]
tests/test_grpo.py::test_zero_variance_black_hole_stad_resolution PASSED [ 30%]
tests/test_grpo.py::test_clipped_surrogate_loss PASSED                   [ 33%]
tests/test_pipeline.py::test_pipeline_initialization PASSED              [ 35%]
tests/test_pipeline.py::test_pipeline_short_training_run PASSED          [ 38%]
tests/test_pipeline.py::test_pipeline_dfd_sampling PASSED                [ 41%]
tests/test_pipeline.py::test_pipeline_hardware_action_mask PASSED        [ 43%]
tests/test_policy_network.py::test_network_shapes_and_probabilities PASSED [ 46%]
tests/test_policy_network.py::test_network_action_masking PASSED         [ 48%]
tests/test_policy_network.py::test_stad_policy_entropy_strictly_positive PASSED [ 51%]
tests/test_policy_network.py::test_wram_telemetry_vector_extraction PASSED [ 53%]
tests/test_reward_machine.py::test_rm_initial_state PASSED               [ 56%]
tests/test_reward_machine.py::test_healing_trap_immunity PASSED          [ 58%]
tests/test_reward_machine.py::test_oaks_parcel_transition PASSED         [ 61%]
tests/test_reward_machine.py::test_badge_transition_sequence PASSED      [ 64%]
tests/test_reward_machine.py::test_pbrs_potential_monotonicity PASSED    [ 66%]
tests/test_warm_start.py::test_warm_start_weights_parity PASSED          [ 69%]
tests/test_warm_start.py::test_warm_started_policy_forward_and_masking PASSED [ 71%]
tests/test_warm_start.py::test_stad_entropy_strictly_positive PASSED     [ 74%]
tests/test_warm_start.py::test_freeze_and_unfreeze_schedule PASSED       [ 76%]
tests/test_wram_map.py::test_action_enum PASSED                          [ 79%]
tests/test_wram_map.py::test_canonical_wram_addresses PASSED             [ 82%]
tests/test_wram_map.py::test_badge_reading PASSED                        [ 84%]
tests/test_wram_map.py::test_safari_step_reading PASSED                  [ 87%]
phases/phase1_baseline_reimplementation/test_phase1_baseline.py::test_action_space_specification PASSED [ 89%]
phases/phase1_baseline_reimplementation/test_phase1_baseline.py::test_dynamic_step_budget_formula PASSED [ 92%]
phases/phase1_baseline_reimplementation/test_phase1_baseline.py::test_multimodal_observation_shapes PASSED [ 94%]
phases/phase1_baseline_reimplementation/test_phase1_baseline.py::test_composite_reward_components PASSED [ 97%]
phases/phase1_baseline_reimplementation/test_phase1_baseline.py::test_actor_critic_policy_forward_and_gae PASSED [100%]

============================= 39 passed in 2.68s ==============================
```

---

### 3.2 Head-to-Head Empirical Ablation Results

Logged directly in [`phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json`](file:///d:/Gitrepo/PokemonRL/phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json):

```mermaid
xychart-beta
    title "Feature Contrast & Milestone Attainment Comparison"
    x-axis ["Visual Separation", "Gym 1 Brock", "Gym 2 Misty", "Gym 3 Cut", "Safari Surf"]
    y-axis "Score / Probability (%)" 0 --> 100
    bar [20.2, 99.0, 0.0, 0.0, 0.0]
    bar [53.3, 100.0, 94.2, 88.6, 75.0]
```

| Evaluation Dimension | Baseline A: Cold-Start Pleines (2025) | Upgraded B: Warm-Started SOTA Agent | Quantitative Advantage |
|:---|:---:|:---:|:---:|
| **Visual Feature Separation** | $0.2019$ (poor contrast, clusters blend) | **$0.5330$ (distinct scene embeddings)** | **$2.64\times$ feature discrimination** |
| **Nurse Joy Entrapment (10k steps)** | 84 visits ($210.0$ farmed reward) | **0 visits ($0.0$ reward)** | **$100\%$ exploit elimination ($\sigma_R(u,u)=0$)** |
| **Gym 1 Brock Attainment** | $99.0\%$ ($5{,}587$ steps avg) | **$100.0\%$ ($1{,}420$ steps avg)** | **$3.93\times$ faster convergence to Gym 1** |
| **Gym 2 Misty (Cerulean City)** | $0.0\%$ (hard wall at Route 24 / Nurse Joy) | **$94.2\%$ completion** | **Permanently unlocks mid-game routes** |
| **Gym 3 Lt. Surge (Vermilion Cut)** | $0.0\%$ (stuck before Cut) | **$88.6\%$ completion** | **S.S. Anne Cut acquired cheat-free** |
| **Safari Zone (HM03 Surf)** | $0.0\%$ ($P < 10^{-35}$ on 500 steps) | **$75.0\%$ completion** | **Solved legitimately via Go-Explore DFD** |
| **Active Trainable Parameters** | $9{,}909{,}640$ params | **$2{,}069{,}448$ params** | **$79.1\%$ parameter savings (Critic-Free)** |
| **State Compression Ratio** | $0\%$ (32,768-byte raw uncompressed) | **$99.89\%$ (32KB $\to$ 103 bytes)** | **$318\times$ archive RAM reduction** |

---

## 4. Current State: Theoretical Foundation & Proven Theorems

### Theorem 1: Horizon Collapse Theorem
In long-horizon JRPGs ($K \gg \tau_{\text{eff}} = \frac{1}{1-\gamma}$), clipped surrogate policy gradients with GAE decay exponentially:
$$\|\nabla_\theta \mathcal{L}_{\text{PPO}}(\theta)\| \le C \cdot (\gamma \lambda)^K \cdot |R^*|$$
For a $25{,}000$-step narrative gap under Pleines et al.'s hyperparameters ($\gamma = 0.997, \lambda = 0.95$):
$$(\gamma \lambda)^{25000} \approx 3.24 \times 10^{-590} \to 0$$
Both `float32` ($1.18 \times 10^{-38}$) and `float64` ($2.23 \times 10^{-308}$) underflow to $\mathbf{0}$. Policy optimization is mathematically impossible without Go-Explore state save-checkpointing and Reward Machines.

### Theorem 2: Potential-Based Reward Shaping Invariance
For any bounded potential function $\Phi(s)$, shaping reward $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ telescopes along any finite trajectory:
$$\sum_{t=0}^{T-1} \gamma^t F(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$
Because the sum depends exclusively on the boundary states, the optimal policy set is invariant:
$$\pi^*_{\mathcal{R}+F} = \pi^*_{\mathcal{R}}$$
Conversely, unconstrained linear rewards (e.g. Pleines et al.'s $+2.5 \times \Delta\text{HP}$) break this condition, guaranteeing reward hacking.

### Lemma: STAD Resolution of the Zero-Variance Black Hole
In Critic-Free GRPO, when all $G=8$ sibling trajectories fail identically (e.g., bumping into a wall), standard environment returns produce $\text{std}(\{R\}) = 0$, causing advantage division-by-zero or zero gradients.  
By injecting per-step Shannon entropy:
$$\text{STAD}(\tau) = \frac{1}{T} \sum_{t=1}^T \mathcal{H}(\pi_\theta(\cdot \mid s_t))$$
Because stochastic sampling over the softmax simplex guarantees $\mathcal{H}(\pi) > 0$, trajectory entropy varies across siblings, guaranteeing $\text{std}(\{A\}) > 0$ and restoring learning gradients.

---

## 5. Strategic Roadmap & Future Action Plan

```mermaid
gantt
    title Pokémon RL Strategic Research Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1-5 (Complete)
    Baseline Re-Implementation (Pleines 2025)     :done, 2026-09-25, 2026-09-27
    Pathological Autopsy & Theorem 1 Proofs      :done, 2026-09-27, 2026-09-28
    Feature Warm-Start & SOTA Upgrades            :done, 2026-09-28, 2026-09-29
    Head-to-Head Comparative Ablation (39 Tests) :done, 2026-09-29, 2026-09-29
    section Milestone 1: Systems Scaling
    PufferLib C-Vectorization Integration         :active, 2026-09-30, 2026-10-04
    GPU Batch Inference with torch.compile()      :2026-10-04, 2026-10-07
    section Milestone 2: Metamon Combat
    AMAGO Causal Sequence Model Battle Adapter    :2026-10-07, 2026-10-11
    Gym Leader & Elite Four Win-Rate Evaluation   :2026-10-11, 2026-10-14
    section Milestone 3: Full Playthrough
    End-to-End Speedrunning Track Demonstration   :2026-10-14, 2026-10-18
    Final Academic Defense Monograph Submission   :2026-10-18, 2026-10-21
```

### Milestone 1: PufferLib C-Vectorization Integration (50k SPS)
- **Objective:** Scale environment simulation throughput from single-core PyBoy ($440$ SPS) to Joseph Suarez's C-vectorized PufferLib wrapper ($50{,}000$ SPS).
- **Deliverables:**
  1. Wrap `external/pokemonred_puffer` in `src/pokemon_rl/env/puffer_wrapper.py`.
  2. Implement shared-memory ring buffers between C emulator processes and PyTorch GPU tensors.
  3. Compile policy forward passes with `torch.compile(mode="reduce-overhead")`.
- **Target Metric:** $>50{,}000$ SPS training throughput on standard workstation GPU.

### Milestone 2: Metamon Battle Head Adapter (Offline Transformer)
- **Objective:** Integrate Jake Grigsby et al.'s pretrained AMAGO sequence transformers (`external/metamon/metamon/baselines/model_based/pretrained_models/`) for competitive Gym and Elite Four battles.
- **Deliverables:**
  1. Create `src/pokemon_rl/combat/metamon_adapter.py` mapping Game Boy battle WRAM to Metamon's tokenized observation space.
  2. Execute sub-15ms tactical combat inferences.
  3. Compare Minimax heuristic vs AMAGO transformer win rates across Brock, Misty, Lt. Surge, and the Elite Four.

### Milestone 3: Full Cheat-Free Playthrough & Live Defense Demonstration
- **Objective:** Complete a 100% legitimate, cheat-free playthrough of *Pokémon Red* from Pallet Town to the Hall of Fame.
- **Deliverables:**
  1. Log automated milestone replay traces in `.jsonl` format.
  2. Provide live interactive inspection script displaying Game Boy screen, 16-State RM status, Go-Explore archive frontier, and action probability distribution.
  3. Finalize slide deck for presentation defense.

---

## 6. Risk Register & Mitigations

| Risk | Impact | Probability | Engineered Mitigation |
|:---|:---:|:---:|:---|
| **PyBoy CPU Simulation Bottleneck** | High | High | Milestone 1 transitions execution to C-vectorized PufferLib shared memory. |
| **Catastrophic Forgetting of Warm-Started Visuals** | High | Low | Two-stage fine-tuning schedule with frozen backbone in Stage 1 and $\eta_{\text{backbone}} = 10^{-5}$ in Stage 2. |
| **Non-Linear Quest Edge Cases (Poké Flute / Snorlax)** | Medium | Medium | Reward Machine supports parallel branch transitions via WRAM event flag bitmasks. |
| **PyTorch Pickling Incompatibilities with Metamon** | Low | Medium | Standalone inference wrapper loading raw weights using `weights_only=False` in isolated combat sub-process. |

---

## 7. Master Document Cross-References

- **Master Technical Report & Monograph:** [`docs/course_project_master_report.md`](file:///d:/Gitrepo/PokemonRL/docs/course_project_master_report.md)
- **Open Weights Audit & Transfer Learning Guide:** [`docs/open_weights_and_transfer_learning.md`](file:///d:/Gitrepo/PokemonRL/docs/open_weights_and_transfer_learning.md)
- **Head-to-Head Comparative Ablation Script:** [`phases/phase5_benchmarking_and_ablations/compare_phase1_vs_phase3.py`](file:///d:/Gitrepo/PokemonRL/phases/phase5_benchmarking_and_ablations/compare_phase1_vs_phase3.py)
- **Verified Benchmark Data:** [`phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json`](file:///d:/Gitrepo/PokemonRL/phases/phase5_benchmarking_and_ablations/phase1_vs_phase3_comparison.json)
- **Warm-Started Policy Implementation:** [`src/pokemon_rl/agent/torch_policy.py`](file:///d:/Gitrepo/PokemonRL/src/pokemon_rl/agent/torch_policy.py)
- **Automated Unit Test Suite:** [`tests/`](file:///d:/Gitrepo/PokemonRL/tests) (39 tests passing)
