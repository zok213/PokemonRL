# pokemon_rl.combat — Decoupled Tactical Combat Head

This module isolates battle decision-making from overworld navigation, eliminating gradient interference and preventing party blackouts.

## Modules

### 1. `combat_controller.py` — Gen 1 LR35902 Minimax Engine
- **Canonical 15x15 Gen 1 Type Matrix:** Implemented directly from `pret/pokered/data/types/type_matchups.asm`. Preserves historical engine bugs:
  - Ghost vs. Psychic = `0.0x` (famous Gen 1 immunity bug).
  - Bug vs. Poison = `2.0x` and Poison vs. Bug = `2.0x` (mutual super-effectiveness).
  - Fire does NOT resist Ice (`1.0x`).
- **Base Speed-Dependent Critical Hits:**
  $$P(\text{crit}) = \frac{\text{BaseSpeed}}{512}, \quad P(\text{high-crit}) = \min\left(255, \, 8 \times \frac{\text{BaseSpeed}}{512}\right)$$
  Moves like Slash on Persian (Base Speed 115) achieve $99.6\%$ critical hit rate.
- **The 1/256 Accuracy Glitch:** All moves capped at $\le 255/256 \approx 99.61\%$ effective hit probability.
- **Opponent AI Modeling:** Decodes deterministic Gym Leader move preference layers from `pret/pokered/engine/battle/ai/trainer_ai.asm`.
- **2x2 Menu Navigation Planner:** Compiles tactical move decisions into stateful D-Pad cursor paths (`[A, DOWN, RIGHT, A]`).

### 2. `metamon_adapter.py` — Metamon Sequence Transformer Adapter
- **WRAM Telemetry Inversion:** Translates raw Game Boy battle bytes (`wPlayerMonHP`, `wEnemyMonHP`, `wPlayerMonType`, `wEnemyMonType`, `wPlayerMonMoves`) into normalized token observation vectors.
- **Pretrained Transformer Query:** Interfaces with Jake Grigsby et al.'s AMAGO causal offline sequence models trained on 22M Pokémon Showdown replays (`external/metamon/metamon/baselines/model_based/pretrained_models/replays_v2_small_trial1_BEST.pt`).
- **Sub-15ms Minimax Fallback:** Automatically falls back to `Gen1CombatController` if Showdown weights are missing or for strict Gen 1 edge-case rules.
