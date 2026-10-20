# Execution & Automation Scripts Directory

This directory contains standalone execution, benchmark, diagnostic, and milestone solver scripts for the Pokémon Red Reinforcement Learning project.

---

## 1. Primary Entrypoints (`scripts/`)

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| `compare_baseline_v2_models_demo.py` | **Live Dual Game Boy Showdown**: Runs Their Baseline V2 vs. Our Retrained Model concurrently in lockstep with real-time telemetry and white-bg visual capture | `python scripts/compare_baseline_v2_models_demo.py --steps 100` |
| `train.py` | Launches PPO training pipeline using native or baseline environments | `python scripts/train.py --steps 100000` |
| `evaluate.py` | Multi-episode policy evaluation and performance logging | `python scripts/evaluate.py --episodes 5` |
| `benchmark_throughput.py` | Measures raw emulation vs vectorized step throughput (SPS) | `python scripts/benchmark_throughput.py` |
| `run_v2_jev_hybrid_interactive.py` | Runs interactive hybrid policy execution | `python scripts/run_v2_jev_hybrid_interactive.py` |
| `clone_external_repos.bat / .sh` | Setup scripts to clone external reference repositories | `scripts/clone_external_repos.bat` |

---

## 2. Milestone Solvers (`scripts/milestones/`)

Deterministic and Joypad-grounded trajectory scripts used to reach, verify, and state-save key milestone locations across Kanto without memory hacks:

- `solve_bicycle_redemption.py`: Navigates from Vermilion City Pokémon Fan Club (Bike Voucher) to Cerulean City Bike Shop to redeem the Bicycle item (`0x06`).
- `solve_cerulean_bike_and_route9.py`: Uses the Bicycle to rapidly cross Route 9 towards Rock Tunnel.
- `solve_from_bicycle_to_route9.py`: Route navigation from Cerulean Bike Shop to Route 9 East gate.
- `solve_route9_to_route10.py`: Navigates through Route 9 trainers to Route 10 North Pokémon Center.
- `solve_rock_tunnel.py`: Traverses Rock Tunnel (1F & B1F) handling darkness navigation, item usage (Antidote `0x0B` for poison cure), and trainer battles.
- `solve_surge_complete.py` / `defeat_lt_surge.py`: Solves Vermilion Gym electric door switches and defeats Lt. Surge for the Thunder Badge (`0x07`).
- `catch_oddish_and_cut.py` / `catch_oddish_perfect.py` / `capture_cut_and_enter_gym.py`: Wild encounter search, Poké Ball throw, capture confirmation, and teaching HM01 Cut.
- `teach_cut.py`: Authentic Joypad menu sequence teaching HM01 Cut to party Pokémon.
- `reproduce_oddish_catch.py`: Deterministic test verifying wild catch Joypad routines.

---

## 3. Diagnostic & Audit Scripts (`scripts/diagnostics/`)

Diagnostic tools for examining WRAM state snapshots, debugging battle menus, and inspecting weights:

- `diagnose_b1f_battle.py`: Inspects active battle memory addresses (`0xD057`, player HP, enemy HP) during Rock Tunnel B1F encounters.
- `test_pewter_to_route4.py`: Diagnoses topological navigation from Pewter City through Route 3 to Mt. Moon.
- `test_catch_oddish.py` & `test_oddish_catch_pipeline.py`: Unit diagnostic harnesses for party expansion and item menu joypad handling.
- `inspect_open_weights.py`: Parses and inspects ported baseline model checkpoint state dictionaries.
- `continue_from_open_weights.py`: Resumes policy execution using reference checkpoint weights.
