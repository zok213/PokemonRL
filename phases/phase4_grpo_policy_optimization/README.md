# Phase 4: Critic-Free Policy Optimization & STAD Variance

This directory contains policy optimization benchmarks and ablations comparing standard Actor-Critic PPO against Critic-Free Adaptive Tau-GRPO.

## Running Phase 4 Ablations

```bash
python phases/phase4_grpo_policy_optimization/run_grpo_ablation.py
```

### Key Findings
- **Zero-Variance Black Hole (EXP-03):** When all $G = 8$ sibling trajectories fail at an identical obstacle (e.g. wall in Rock Tunnel), standard GRPO suffers from a 34% gradient stall rate.
- **STAD Resolution (EXP-04):** Injecting Trajectory State-Action Diversity guarantees non-zero variance, reducing gradient stalls to **0.0%**.
- **Memory Footprint:** Removing the Value Critic network eliminates 50% of model and optimizer parameters ($16P$ vs. $32P$ bytes).
