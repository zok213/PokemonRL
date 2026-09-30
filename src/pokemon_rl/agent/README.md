# pokemon_rl.agent — Decision Engines & Policy Networks

This module implements the core agent policy architectures and formal reward structures.

## Modules

### 1. `reward_machine.py` — Formal 16-State Mealy Reward Machine
- **Mathematical Definition:** $\mathcal{M}_{\text{RM}} = \langle \mathcal{U}, u_0, \Sigma, \delta, \sigma_R \rangle$.
- **States:** 16 discrete narrative states tracking canonical Game Boy quest milestones from Pallet Town (`OAKS_PARCEL`) to Indigo Plateau (`CHAMPION_DEFEATED`).
- **Healing Trap Immunity:** All self-loops and non-progressing transitions yield identically zero: $\sigma_R(u, u) = 0.0$.
- **Potential-Based Shaping:** Integrates strictly telescoping PBRS potential $\Phi(s)$ satisfying Theorem 2 policy invariance ($\pi^*_{\mathcal{R}+F} = \pi^*_{\mathcal{R}}$).

### 2. `torch_policy.py` — Warm-Started MultiModal Policy Network
- **Feature Warm-Start:** Loads Peter Whidden's 439M-step pretrained visual backbone (`features_extractor.cnn.*`) directly from `external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip`.
- **Multimodal Fusion:**
  - Visual Stream: Nature CNN (Conv2D `32, 64, 64`) processing downsampled screen `(B, 3, 72, 80)`.
  - Spatial Stream: Conv2D processing dynamic local explored tile grid `(B, 1, 48, 48)`.
  - WRAM Telemetry Stream: 2-layer MLP processing 64 continuous Game Boy RAM registers.
- **Two-Stage Cosine Fine-Tuning:** Stage 1 freezes the visual backbone ($0 \to 500\text{k}$ steps) to protect pre-trained visual filters; Stage 2 fine-tunes with a conservative learning rate ($\eta_{\text{max}} = 10^{-5}$).

### 3. `policy_network.py` — NumPy MultiModal Policy Network
- NumPy reference implementation for fast forward evaluation, action masking, and State-Action Diversity (STAD) entropy estimation without PyTorch dependencies.
