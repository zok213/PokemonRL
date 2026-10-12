# pokemon_rl.agent — Decision Engines & Policy Networks

This module implements the core agent policy architectures, neuro-symbolic affordance engines, and formal reward structures.

## Modules

### 1. `v2_jev_hybrid_policy.py` — Residual Policy Steering (Baseline V2 Weights + Jev Affordances)
- **Flagship Architecture:** Reuses Peter Whidden's 26.2M-step pretrained PPO policy (`poke_26214400.zip`) as the visual-motor prior, steered by Jev-style System 1 affordances.
- **Formulation:**
  $$\pi_{\text{hybrid}}(a \mid s) = \text{Softmax}\left( \frac{z_{V2}(a \mid s) + \alpha(s) \cdot z_{\text{affordance}}(a \mid s, \mathcal{A})}{T} \right)$$
- **Features:**
  - Extracts raw 7-action logits directly from SB3 `MultiInputActorCriticPolicy`.
  - Injects topological navigation biases towards Kanto World Graph subgoals.
  - Enforces Gen 1 battle legality (masks RUN in trainer battles, boosts super-effective moves).
  - Dynamically activates **Anti-Stagnation Circuit Breaker** ($\alpha \to 3.5$) when local oscillations occur.

### 2. `tactical_battle.py` — Authentic Gen 1 Tactical Battle Controller
- Direct transcription of canonical 15-type interaction table from `pret/pokered` (`data/types/type_matchups.asm`).
- Computes expected damage via authentic Gen 1 damage formula:
  $$\text{Damage} = \left(\frac{2 \times \text{Level}}{5} + 2\right) \times \frac{\text{Power}}{50} \times \text{STAB} \times \text{TypeMult} \times \text{Accuracy}$$
- Dynamically generates ranked affordance choices (`[SUPER EFFECTIVE]`, `[NOT VERY EFFECTIVE]`).
- Strictly prohibits fleeing in Trainer battles (`wIsInBattle == 2`).

### 3. `local_decision_policy.py` — Local Affordance Decision Head & SMDP Harness
- Open-weight PyTorch neural policy running in $<2\text{ ms}$ on CPU with zero network dependencies.
- Calibrated Multi-Affordance Softmax over candidate discrete affordance embeddings.
- Saves and loads persistent weights from `models/pokemonrl_affordance_model.pt`.
- `SMDPDecisionHarness`: Bridges Game Boy Work RAM telemetry, encodes 32-dim state representations (including topological subgoal displacement $\Delta x, \Delta y, d$), and executes macro sequences.

### 4. `train_local_policy.py` — Neural Policy Training & Distillation
- Supervised multi-task training loop across 56 representative Kanto fork scenarios (navigation, super-effective combat, wild vs trainer low-HP survival, dialogue text advance).
- Converged to 100.0% accuracy across 120 epochs, saving persistent weights to `models/`.

### 5. `reward_machine.py` — Formal 16-State Mealy Reward Machine
- **Mathematical Definition:** $\mathcal{M}_{\text{RM}} = \langle \mathcal{U}, u_0, \Sigma, \delta, \sigma_R \rangle$.
- **States:** 16 discrete narrative states tracking canonical Game Boy quest milestones from Pallet Town (`OAKS_PARCEL`) to Indigo Plateau (`CHAMPION_DEFEATED`).
- **Healing Trap Immunity:** All self-loops and non-progressing transitions yield identically zero: $\sigma_R(u, u) = 0.0$.
- **Potential-Based Shaping:** Integrates strictly telescoping PBRS potential $\Phi(s)$ satisfying Theorem 2 policy invariance ($\pi^*_{\mathcal{R}+F} = \pi^*_{\mathcal{R}}$).

### 6. `torch_policy.py` — Warm-Started MultiModal Policy Network
- **Feature Warm-Start:** Loads Peter Whidden's 439M-step pretrained visual backbone (`features_extractor.cnn.*`) directly from `external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip`.
- **Multimodal Fusion:**
  - Visual Stream: Nature CNN (Conv2D `32, 64, 64`) processing downsampled screen `(B, 3, 72, 80)`.
  - Spatial Stream: Conv2D processing dynamic local explored tile grid `(B, 1, 48, 48)`.
  - WRAM Telemetry Stream: 2-layer MLP processing 64 continuous Game Boy RAM registers.
