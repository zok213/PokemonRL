# Phase 2: Adversarial Pathological Autopsy

This directory contains standalone mathematical autopsies and simulation runners for the **Four Classical JRPG Pathologies** and **Theorem 1 (Horizon Collapse)**.

---

## 1. Pathology Autopsy Manifest

| File | Pathology Analyzed | Mathematical Proof / Calculation | Architectural SOTA Remedy |
|:---|:---|:---|:---|
| `pathology1_healing_trap.py` | **The Healing Trap** | $V^{\pi_{\text{heal}}} \approx 19.76 \gg 1.67 \approx V^{\pi_{\text{explore}}}$ ($11.8\times$ exploit) | **16-State Reward Machine** ($\sigma_R(u, u) = 0.0$ by construction) |
| `pathology4_safari_zone_wall.py` | **500-Step Safari Zone Wall** | $E[T] = 81{,}796$ steps ($163\times$ budget deficit); $P(\text{Success}) < 10^{-35}$ | **Go-Explore State Archive** with DFD sampling |
| `theorem1_horizon_collapse.py` | **Horizon Collapse Theorem** | $(\gamma\lambda)^{25000} \approx 3.24 \times 10^{-590} \to 0$ (IEEE 754 underflow) | **Critic-Free GRPO** + Average-Reward Poisson continuation |

---

## 2. Running Phase 2 Autopsy Scripts

```bash
# 1. Verify the Healing Trap Bellman value calculation
python phases/phase2_pathological_autopsy/pathology1_healing_trap.py

# 2. Verify Safari Zone hitting time deficit and binomial tail bound
python phases/phase2_pathological_autopsy/pathology4_safari_zone_wall.py

# 3. Verify Theorem 1 gradient credit attenuation underflow
python phases/phase2_pathological_autopsy/theorem1_horizon_collapse.py
```
