"""
run_pyboy_interactive.py — Real-Time Interactive Game Boy RL Visualizer CLI
===========================================================================
CLI entry point for running the interactive Pokémon Red RL session at arbitrary
emulation speeds (1x to 16x) with live SDL2 display and terminal telemetry.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 support on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root and src to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from interactive.session import InteractiveGameBoyRLSession


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Game Boy RL Visualizer")
    parser.add_argument(
        "--rom",
        type=str,
        default="roms/pokemon_red.gb",
        help="Path to Pokémon Red .gb ROM",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="external/PokemonRedExperiments/init.state",
        help="Path to initial save state (.state)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/PokemonRedExperiments/v2/runs/poke_26214400.zip",
        help="Path to pretrained PPO checkpoint (.zip)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum number of decision steps (0 = run indefinitely until window is closed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without SDL2 window (terminal HUD only)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ppo",
        choices=["ppo", "manual"],
        help="Execution mode: 'ppo' (real neural network policy) or 'manual' (interactive human play)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=16,
        help="Emulation speed multiplier (1 = 60 FPS normal speed, 16 = 16x fast speed, 0 = uncapped)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Game Boy screen window scale multiplier (default: 3 for 480x432 window)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable low-level hardware action masking",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Search paths for Pokémon Red ROM
    rom_candidates = [
        args.rom,
        "roms/pokemon_red.gb",
        "PokemonRed.gb",
        "external/PokemonRedExperiments/PokemonRed.gb",
    ]
    resolved_rom = None
    for cand in rom_candidates:
        if cand and Path(cand).exists():
            resolved_rom = str(Path(cand).resolve())
            break

    if not resolved_rom:
        print(f"[!] Error: No ROM found. Please place pokemon_red.gb in roms/.")
        sys.exit(1)

    # Resolve state path
    state_candidates = [
        args.state,
        f"saves/{Path(args.state).name}",
        f"external/PokemonRedExperiments/{Path(args.state).name}",
    ]
    resolved_state = None
    for cand in state_candidates:
        if cand and Path(cand).exists():
            resolved_state = str(Path(cand).resolve())
            break

    # Resolve checkpoint
    checkpoint_candidates = [
        args.checkpoint,
        "external/PokemonRedExperiments/v2/runs/poke_26214400.zip",
        "baselines/session_4da05e87_main_good/poke_439746560_steps.zip",
    ]
    resolved_checkpoint = None
    for cand in checkpoint_candidates:
        if cand and Path(cand).exists():
            resolved_checkpoint = str(Path(cand).resolve())
            break

    session = InteractiveGameBoyRLSession(
        rom_path=resolved_rom,
        state_path=resolved_state,
        checkpoint_path=resolved_checkpoint,
        headless=args.headless,
        action_freq=24,
        emulation_speed=args.speed,
        scale=args.scale,
        use_action_masking=not args.no_mask,
        use_pbrs=True,
        mode=args.mode,
    )
    session.run_loop(max_steps=args.steps)



if __name__ == "__main__":
    main()

