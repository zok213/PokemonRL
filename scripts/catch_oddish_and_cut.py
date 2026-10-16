"""
catch_oddish_and_cut.py — Autonomous Authentic Progression to Route 6, Catch Cut User, & Cut Gym Tree
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pokemon_rl.env.wram_map import RAMMap
from pokemon_rl.navigation.world_graph import WorldGraphNavigator

def run():
    rom_candidates = [
        REPO_ROOT / "roms" / "pokemon_red.gb",
        REPO_ROOT / "PokemonRed.gb",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "PokemonRed.gb",
    ]
    rom_path = next(str(p) for p in rom_candidates if p.exists())
    
    pb = PyBoy(rom_path, window="null")
    with open(REPO_ROOT / "saves" / "checkpoint_progress.state", "rb") as f:
        pb.load_state(f)

    # Allow ship cutscene to finish and settle in Vermilion City
    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    for _ in range(120):
        pb.tick()

    print(f"[+] Started at Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    def press(btn: str, hold: int = 8, wait: int = 16):
        mapping = {
            "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
            "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
        }
        p, r = mapping[btn]
        pb.send_input(p)
        for _ in range(hold):
            pb.tick()
        pb.send_input(r)
        for _ in range(wait):
            pb.tick()

    nav = WorldGraphNavigator()

    # Step 1: Walk from Vermilion City (18, 29) to North Gate at (18, 0)
    print("[+] Step 1: Walking to Vermilion City North Gate...")
    for s in range(40):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m != 5:
            print(f"[+] Transitioned to Map {m} ({x}, {y})")
            break
        if y == 0:
            press("up", wait=24)
            continue
        wp = nav._bfs_next_waypoint(5, (x, y), (18, 0))
        dx = wp[0] - x
        dy = wp[1] - y
        if dy < 0: press("up")
        elif dy > 0: press("down")
        elif dx > 0: press("right")
        elif dx < 0: press("left")
        else: press("up")

    # Step 2: In Gatehouse 74 (Vermilion North Gate), walk from (4, 7) to (4, 0) to enter Route 6
    print("[+] Step 2: Walking through Vermilion North Gate (Map 74)...")
    for s in range(25):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m == 17:  # Route 6
            print(f"[+] Entered Route 6 at ({x}, {y})")
            break
        if m == 74:
            if y <= 1:
                press("up", wait=24)
            else:
                press("up", wait=18)

    print(f"[+] Current location: Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")
    pb.stop()

if __name__ == "__main__":
    run()
