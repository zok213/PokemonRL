# Phase 1: Canonical Baseline Re-Implementation

This directory contains the exact, faithful re-implementation of the empirical baseline from:
> **Marco Pleines, Matthias Addis, David Rubinstein, Frank Zimmer, Mike Preuss, Peter Whidden**  
> *"Playing Pokémon Red via Deep Reinforcement Learning"*  
> **IEEE Conference on Games (CoG) 2025**  
> IEEE Xplore Document: 11114399 | arXiv: [2502.19920](https://arxiv.org/abs/2502.19920)

---

## 1. Specification Mapping: Paper vs. Code

| Paper Specification | Equation / Parameter in Paper | Implementation in Code | File / Line |
|:---|:---:|:---:|:---|
| **Action Cadence** | 1 action per 24 frames (8 hold, 16 release) | `FRAME_HOLD=8, FRAME_RELEASE=16, STRIDE=24` | `pleines_baseline_env.py` |
| **Action Space** | 7 discrete Game Boy buttons (SELECT omitted) | `PleinesAction` (UP, DOWN, LEFT, RIGHT, A, B, START) | `pleines_baseline_env.py` |
| **Visual Obs** | $72 \times 80 \times 3$ stacked grayscale frames | `screen_stack` (deque maxlen 3 of $72 \times 80$) | `pleines_baseline_env.py` |
| **Spatial Obs** | $48 \times 48$ visited coordinate binary map | `visited_tiles_map` centered on player | `pleines_baseline_env.py` |
| **Telemetry Obs** | Masked HP, levels, event bitflags (64 floats) | `telemetry` vector | `pleines_baseline_env.py` |
| **Step Budget** | $B_t = 10{,}240 + 2{,}048 \times N_{\text{events}}(s_t)$ | `calculate_step_budget(events)` | `pleines_baseline_env.py` |
| **Event Reward** | $R_{\text{event}} = +2.0 \cdot \Delta N_{\text{events}}$ | `r_event = 2.0 * delta_events` | `pleines_baseline_env.py` |
| **Nav Reward** | $R_{\text{nav}} = +0.005 \cdot \mathbb{I}[c_t \notin \mathcal{H}]$ | `r_nav = 0.005 if coords not in visited else 0` | `pleines_baseline_env.py` |
| **Healing Reward** | $R_{\text{heal}} = 2.5 \sum_i \frac{\Delta\text{HP}_i}{\text{MaxHP}_i}$ | `r_heal = 2.5 * hp_gain_ratio` | `pleines_baseline_env.py` |
| **Level Reward** | $R_{\text{lvl}} = 0.5 \min(\sum \text{lvl}, \frac{\sum \text{lvl}-22}{4}+22)$ | `r_lvl = 0.5 * delta_lvl_potential` | `pleines_baseline_env.py` |
| **Policy Body** | FeedForward (2.03M) & GRU (3.87M) | `PleinesActorCriticPolicy` | `pleines_ppo_policy.py` |
| **PPO GAE** | $\gamma = 0.997, \lambda = 0.95, \epsilon = 0.2, v = 0.5$ | `compute_gae(...)` | `pleines_ppo_policy.py` |

---

## 2. Running Phase 1 Re-Implementation

```bash
# 1. Run unit test suite confirming mathematical fidelity
python -m pytest phases/phase1_baseline_reimplementation/test_phase1_baseline.py -v

# 2. Run reproduction of all 6 Table 3 ablation variants
python phases/phase1_baseline_reimplementation/reproduce_pleines_experiments.py
```

---

## 3. Reproduced Ground Truth (Table 3 in Pleines et al. 2025)

| Variant | Beat Brock | Mt. Moon | Cerulean City | Beat Misty | Cerulean Steps | Heals Farmed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Human Playthroughs** | $1.00 \pm 0.0$ | $1.00 \pm 0.0$ | $1.00 \pm 0.0$ | $1.00 \pm 0.0$ | $11{,}188 \pm 5{,}224$ | $22.97 \pm 11.3$ |
| **Baseline (Squirtle)** | $0.99 \pm 0.0$ | $0.97 \pm 0.0$ | $0.85 \pm 0.1$ | $0.27 \pm 0.3$ | $25{,}299 \pm 4{,}950$ | $12.81 \pm 5.9$ |
| **Baseline $- R_{\text{lvl}}$** | $0.95 \pm 0.0$ | $0.91 \pm 0.1$ | $0.79 \pm 0.1$ | $0.31 \pm 0.3$ | $21{,}352 \pm 3{,}898$ | $11.73 \pm 4.0$ |
| **Baseline $- R_{\text{heal}}$** | $0.99 \pm 0.0$ | $0.97 \pm 0.0$ | $0.91 \pm 0.1$ | $0.07 \pm 0.1$ | $28{,}813 \pm 2{,}880$ | $1.45 \pm 0.7$ |
| **GRU Memory Body** | $0.76 \pm 0.4$ | $0.71 \pm 0.4$ | $0.49 \pm 0.4$ | $0.21 \pm 0.3$ | $27{,}118 \pm 4{,}236$ | $23.09 \pm 23.3$ |
| **Choose Starter (Pallet)** | $0.79 \pm 0.4$ | $0.79 \pm 0.4$ | $0.75 \pm 0.4$ | $0.29 \pm 0.4$ | $30{,}593 \pm 1{,}143$ | $33.75 \pm 34.1$ |
| **Bulbasaur Starter** | $0.94 \pm 0.0$ | $0.94 \pm 0.0$ | $\mathbf{0.00 \pm 0.0}$ | $\mathbf{0.00 \pm 0.0}$ | Trapped at Mt. Moon | $\mathbf{399.33 \pm 177.1}$ |

---

## 4. Key Takeaways for the Course Report

1. **The Starter Paradox:** In *Choose Starter*, Charmander is picked because Professor Oak's Pokéball is 1 tile closer, yielding earlier $+2.0$ event rewards. But Charmander loses to Misty $25{,}000$ steps later ($2\%$ win rate). The policy is blind to downstream catastrophe due to exponential discounting.
2. **The Bulbasaur Leech Life Trap:** When starting with Bulbasaur, wild Zubats in Mt. Moon heal via Leech Life while Bulbasaur heals via Leech Seed. The agent discovers endless battles yielding continuous $R_{\text{heal}}$ rewards ($399.33$ heals/run), causing **$0\%$ arrival at Cerulean City**.
3. **The 0% Vermilion Cut Impasse:** Across all variants, obtaining HM01 Cut, teaching it via multi-nested menus, and cutting the overworld shrub achieved **$0.0\%$ completion**.
