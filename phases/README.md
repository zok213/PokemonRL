# Active 5-Phase Research Architecture

This directory organizes our entire research methodology into five reproducible, executable phases:

```
phases/
├── phase1_baseline_reimplementation/      # 100% faithful Pleines et al. (IEEE CoG 2025) baseline
│   ├── pleines_baseline_env.py             # 24-frame action wrapper, 7 actions, composite reward
│   ├── pleines_ppo_policy.py               # Nature CNN + Spatial + Learned Critic V_phi(s)
│   ├── reproduce_pleines_experiments.py    # Table 3 ablation variants from the paper
│   └── test_phase1_baseline.py             # 5 unit tests verifying baseline behavior
│
├── phase2_pathological_autopsy/            # Mathematical & empirical failure modes of baseline
│   ├── pathology1_healing_trap.py          # Nurse Joy reward loop (V_heal=19.76 >> V_explore=1.67)
│   ├── pathology4_safari_zone_wall.py      # 500-step countdown barrier (P < 10^-35)
│   └── theorem1_horizon_collapse.py        # IEEE 754 gradient vanishing ((gamma*lambda)^25000 -> 0)
│
├── phase3_neuro_symbolic_upgrades/         # SOTA upgrades overcoming the pathologies
│   ├── warm_started_policy_network.py      # Feature Warm-Start: Whidden 439M ConvNet backbone
│   └── run_upgraded_demo.py                # Integrated demo of all 6 upgrades (RM, Mask, Go-Explore, etc.)
│
├── phase4_grpo_policy_optimization/        # Policy optimization without Value Critic
│   └── run_grpo_ablation.py                # Sibling count G in {1, 4, 8, 16} & STAD variance injection
│
└── phase5_benchmarking_and_ablations/      # Multi-seed benchmarks & direct head-to-head ablations
    ├── compare_phase1_vs_phase3.py         # Side-by-side: Cold Baseline vs Warm-Started Agent
    └── run_multi_seed_benchmark.py         # 5-seed statistical benchmark (Throughput, Cells, Delta)
```
