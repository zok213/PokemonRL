# Automated Verification & Pytest Test Suite

The `tests/` directory contains unit and integration tests verifying all components of the `PokemonRL` system.

## Verification Status: 54/54 Passing (100% Green)

The suite is split across:
- **`tests/`**: 49 unit tests covering production modules.
- **`phases/phase1_baseline_reimplementation/test_phase1_baseline.py`**: 5 baseline reproduction tests.

```bash
# Run all 54 tests
python -m pytest tests/ phases/ -v

# Run with execution timing and captured stdout
python -m pytest tests/ phases/ -s
```

## Test Module Catalog

| Test File | Tests | Components Verified |
|:---|:---:|:---|
| [`test_action_masker.py`](file:///d:/Gitrepo/PokemonRL/tests/test_action_masker.py) | 3 | Hardware `wJoyIgnore` bitmasks, text dialogue lockouts, wall bump stagnation. |
| [`test_combat_controller.py`](file:///d:/Gitrepo/PokemonRL/tests/test_combat_controller.py) | 7 | 15x15 Gen 1 type multipliers, historical engine bugs, base-speed crits, 1/256 glitch, minimax. |
| [`test_go_explore.py`](file:///d:/Gitrepo/PokemonRL/tests/test_go_explore.py) | 5 | 99.89% XOR delta compression roundtrips, archive registration, hierarchical cells, stale culling. |
| [`test_grpo.py`](file:///d:/Gitrepo/PokemonRL/tests/test_grpo.py) | 3 | Zero-mean advantages, STAD zero-variance black hole resolution, clipped surrogate loss. |
| [`test_metamon_adapter.py`](file:///d:/Gitrepo/PokemonRL/tests/test_metamon_adapter.py) | 3 | Battle WRAM feature extraction, Showdown token inversion, minimax fallback. |
| [`test_native_vectorizer.py`](file:///d:/Gitrepo/PokemonRL/tests/test_native_vectorizer.py) | 4 | Numba LLVM parallel JIT, zero-copy PyTorch tensors, hardware masking, 18,741 SPS benchmark. |
| [`test_pipeline.py`](file:///d:/Gitrepo/PokemonRL/tests/test_pipeline.py) | 4 | Production pipeline initialization, 25-cycle training runs, DFD sampling, hardware masks. |
| [`test_policy_network.py`](file:///d:/Gitrepo/PokemonRL/tests/test_policy_network.py) | 4 | Multimodal tensor shapes, action masking, strictly positive STAD entropy, WRAM extraction. |
| [`test_puffer_bridge.py`](file:///d:/Gitrepo/PokemonRL/tests/test_puffer_bridge.py) | 3 | Vectorized environment reset shapes, step and reward calculation, batch throughput. |
| [`test_reward_machine.py`](file:///d:/Gitrepo/PokemonRL/tests/test_reward_machine.py) | 5 | Initial states, 100% healing trap immunity, Oak's Parcel, badge sequences, PBRS monotonicity. |
| [`test_warm_start.py`](file:///d:/Gitrepo/PokemonRL/tests/test_warm_start.py) | 4 | Bitwise weight parity against Whidden 439M, policy forward evaluation, freeze/unfreeze schedule. |
| [`test_wram_map.py`](file:///d:/Gitrepo/PokemonRL/tests/test_wram_map.py) | 4 | Discrete action enums, canonical WRAM addresses, badge bitfields, Safari step registers. |
