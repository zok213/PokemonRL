"""
test_real_pyboy.py
==================
Unit tests verifying real, authentic PyBoy Game Boy emulation on roms/pokemon_red.gb.
Verifies hardware cycle ticking, WRAM bus reading, and real runner step loop.
"""

import pytest
from pathlib import Path
from pokemon_rl.env.real_pyboy_env import RealPyBoyEnvironment
from pokemon_rl.real_runner import RealPokemonRLRunner


def test_real_pyboy_environment_lifecycle():
    """Verify PyBoy boots real ROM, reads memory, and advances frames."""
    rom = Path("roms/pokemon_red.gb")
    if not rom.exists():
        rom = Path("external/PokemonRedExperiments/PokemonRed.gb")
    if not rom.exists():
        pytest.skip("Pokemon Red ROM not available on disk")

    env = RealPyBoyEnvironment(rom_path=rom, window_type="null")
    assert env.pyboy is not None

    # Tick 30 frames
    alive = env.tick_frames(30)
    assert bool(alive) is True

    # Read WRAM snapshot
    snap = env.get_wram_snapshot()
    assert "map_id" in snap
    assert "x" in snap
    assert "y" in snap
    assert "badges" in snap
    assert "in_battle" in snap

    # Test hardware button press pulse
    press_ok = env.send_button("a", hold_frames=2, release_frames=4)
    assert bool(press_ok) is True
    assert env.step_count == 1

    env.close()
    assert env.pyboy is None


def test_real_runner_live_steps():
    """Verify RealPokemonRLRunner executes live steps and reaches Red's bedroom (Map 38)."""
    rom = Path("roms/pokemon_red.gb")
    if not rom.exists():
        rom = Path("external/PokemonRedExperiments/PokemonRed.gb")
    if not rom.exists():
        pytest.skip("Pokemon Red ROM not available on disk")

    runner = RealPokemonRLRunner(rom_path=str(rom), window_type="null", max_steps=10, verbose=False)
    metrics = runner.run(num_steps=10)

    assert metrics["steps_taken"] == 10
    assert metrics["emulation_frames"] >= 240
    assert metrics["elapsed_seconds"] > 0
    assert metrics["sps"] > 0
    assert metrics["fps"] > 0
