# Autonomous JRPG Agent Architecture Specification (2026 SOTA)

## 1. Mathematical POMDP Formalization

A commercial long-horizon JRPG is formalized as a partially observable Markov decision process:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

- **Hardware State Space $\mathcal{S}$:** Sharp SM83 (Z80-derivative) Game Boy volatile RAM:
  - 8,192 bytes Work RAM (WRAM: `0xC000–0xDFFF`)
  - 8,192 bytes Video RAM (VRAM: `0x8000–0x9FFF`)
  - 127 bytes High RAM (HRAM: `0xFF80–0xFFFE`)
  - Total: 16,511 addressable bytes ($132,088$ bits)
  - Raw unconstrained state cardinality: $|\mathcal{S}| \le 2^{132,088}$ (or $2^{65,536}$ in WRAM alone).
- **Action Space $\mathcal{A}$:** 8 Game Boy joypad lines: `UP, DOWN, LEFT, RIGHT, A, B, START, SELECT`.
- **Observation Space $\Omega$:**
  - Visual stream: downsampled grayscale $72 \times 80 \times 4$ frames.
  - Spatial map: $48 \times 48$ binary matrix of visited overworld tiles.
  - WRAM telemetry: 64 packed telemetry bytes.
- **Sparse Reward $\mathcal{R}$:** $+100.0$ strictly emitted upon milestone quest graph transitions.

---

## 2. Core Theorems & Governing Results

### Theorem 1: The Horizon Collapse Theorem
For a milestone reward $R^*$ positioned $K$ steps downstream with discount $\gamma$ and GAE parameter $\lambda$:

$$\left\| \nabla_\theta \mathcal{L}_{\text{PPO}}(\theta) \right\| \le C \cdot (\gamma \lambda)^K \cdot |R^*|$$

Under standard empirical settings ($\gamma = 0.997, \lambda = 0.95$):
- $\gamma \lambda = 0.94715$
- At $K = 25,000$ steps (e.g. Lavender Town $\to$ Celadon Gym):
  $$(\gamma \lambda)^{25000} \approx 3.24 \times 10^{-590} \to 0$$
- Policy gradients vanish past standard IEEE 754 float32 underflow ($1.18 \times 10^{-38}$) and float64 underflow ($2.23 \times 10^{-308}$).
- **Resolution:** Critic-Free GRPO removes temporal discounting from advantages; Average-Reward Poisson Bellman formulation ($\gamma = 1.0$).

### Theorem 2: PBRS Policy Invariance
Any shaping reward satisfying $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ preserves the set of optimal policies:

$$\pi^*_{\mathcal{R} + F} = \pi^*_{\mathcal{R}}$$

**Proof via Telescoping Sum:**
$$\sum_{t=0}^T \gamma^t F(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)$$
The sum depends exclusively on initial state $s_0$ and terminal state $s_T$, and is entirely independent of the intermediate action choices $(a_0, \dots, a_{T-1})$. Therefore $Q^*(s, a) - Q^*(s, a')$ remains invariant.

### Lemma: Zero-Variance Black Hole & STAD Resolution
When all $G = 8$ sibling rollouts stall at an identical topological obstacle (e.g., hitting a wall in Rock Tunnel), $\text{std}(\{R\}) \to 0$ and $\nabla_\theta \mathcal{L}_{\text{GRPO}} \to \mathbf{0}$.
**Resolution:** Inject normalized Trajectory State-Action Diversity (STAD):

$$\text{STAD}(\tau_i) = \frac{1}{H} \sum_{t=1}^H \mathcal{H}(\pi_\theta(\cdot \mid s_{i,t})) = \frac{1}{H} \sum_{t=1}^H \left[ -\sum_{a} \pi_\theta(a \mid s_{i,t}) \log \pi_\theta(a \mid s_{i,t}) \right]$$

Since softmax outputs are strictly positive, $\text{STAD}(\tau_i) > 0$ always, guaranteeing non-zero advantage variance.

---

## 3. Four Classical Pathologies & Architectural Remedies

```
+-----------------------------------+-----------------------------------+
| Classical Pathology               | Architectural Remedy              |
+-----------------------------------+-----------------------------------+
| 1. The Healing Trap               | 16-State Reward Machine           |
|    V^heal ≈ 19.76 >> 1.67 V^exp   | sigma_R(u, u) = 0.0 by definition |
+-----------------------------------+-----------------------------------+
| 2. Noisy TV Water Hypnosis        | Exact WRAM Coordinate Hashing     |
|    k-NN spikes on 4-frame water   | 0xD362 (X), 0xD361 (Y), 0xD35E (M)|
+-----------------------------------+-----------------------------------+
| 3. Menu Oscillation Deadlocks     | Dynamic Action Masking            |
|    START/B cycling at 30 Hz       | wJoyIgnore (0xCD6B) + START latch |
+-----------------------------------+-----------------------------------+
| 4. 500-Step Safari Zone Wall      | Go-Explore State Archive + DFD    |
|    P(Success) < 10^-35            | Deterministic state restoration   |
+-----------------------------------+-----------------------------------+
```
