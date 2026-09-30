# Interactive Pokémon Red RL Suite

Autonomous Reinforcement Learning Execution and Visualization Suite for Pokémon Red (Game Boy LR35902).

---

## 1. Modular Architecture Overview

The `interactive/` package provides real-time telemetry, emulator control, and visualization tools for genuine reinforcement learning agents:

```
interactive/
├── __init__.py                 # Package exports and public API
├── constants.py                # Hardware enums, WRAM bitmasks, Gen 1 moves, map dictionaries
├── emulator.py                 # PyBoy 2.7.0 LR35902 CPU wrapper, joypad pulse scheduling, SDL2 display
├── telemetry.py                # WRAM state observer, multimodal tensor builder, PBRS reward shaping
├── hud.py                      # Real-time ANSI terminal telemetry HUD & action probability visualizer
├── session.py                  # Main session coordinator (neural policy, dynamic masking, combat head)
├── run_pyboy_interactive.py    # Streamlined CLI entrypoint with argparse support
├── launch_baseline_v2.bat      # 1-click Windows batch launcher for Baseline V2 (Real RL, 26.2M PPO)
├── launch_baseline_v2.ps1      # 1-click PowerShell launcher for Baseline V2 (Real RL, 26.2M PPO)
├── launch_interactive.bat      # 1-click shortcut launcher forwarding to Baseline V2
├── launch_interactive.ps1      # 1-click PowerShell shortcut forwarding to Baseline V2
├── visualizer.html             # Standalone client-side HTML5 canvas visualizer
└── README.md                   # Interactive documentation
```

---

## 2. Quick Start Guide

### Launch Baseline V2 (Pretrained 26.2M PPO Policy, Zero Cheats)
Launches the genuine Game Boy screen running the 26.2M-step pretrained PPO neural network from Pallet Town:
```powershell
.\launch_baseline_v2.ps1
```
Or simply double-click `launch_baseline_v2.bat` (or `launch_interactive.bat`).

### Direct CLI Execution
```powershell
$env:PYTHONPATH="external/PokemonRedExperiments/v2;src;."
python external/PokemonRedExperiments/v2/run_pretrained_interactive.py
```

### Launch Interactive Session with Live HUD
```powershell
$env:PYTHONPATH="src;."
python interactive/run_pyboy_interactive.py --speed 16 --steps 0
```

---

## 3. Interactive Controls & Hotkeys

| Input | Target | Function |
|:---|:---|:---|
| **Close Window** | SDL2 GUI | Cleanly terminates emulation loop and flushes telemetry |
| **`Ctrl+C`** | Terminal | Sends SIGINT to emulator session and exits |
| **`agent_enabled.txt`** | File toggle | Write `no` to pause AI and take manual control; write `yes` to resume AI |
| **Directional Keys** | Game Boy | Down, Left, Right, Up arrow keys for manual movement |
| **`Z` / `X`** | Joypad | A and B buttons in PyBoy window |
| **`Enter` / `Space`**| Joypad | Start and Select buttons |

---

## 4. Telemetry HUD Output

The interactive runner outputs live Game Boy WRAM metrics every 10 steps:
```
Step   120 | Map= 0 Pos=( 5, 4) | HP=20/20 Lv=5 | Act=3 Rew=+0.015
Step   130 | Map= 0 Pos=( 5, 3) | HP=20/20 Lv=5 | Act=3 Rew=+0.020
```
- **Map:** Current map identifier (0 = Pallet Town, 1 = Viridian City, 12 = Route 1, etc.)
- **Pos:** Exact player grid coordinates `(X, Y)`
- **HP / Lv:** Player starter Pokémon HP and legitimate level (Starts at Lv 5, NO cheats)
- **Act / Rew:** Dispatched discrete joypad button action and step reward delta
