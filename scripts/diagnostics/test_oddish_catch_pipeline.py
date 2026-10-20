"""
test_oddish_catch_pipeline.py — Autonomous Test of Full Route 6 Oddish Capture & Cut Execution
"""
from __future__ import annotations
import sys
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

    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    for _ in range(120):
        pb.tick()

    print(f"[+] Starting at Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

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
        for _ in range(hold): pb.tick()
        pb.send_input(r)
        for _ in range(wait): pb.tick()

    nav = WorldGraphNavigator()

    # 1. Walk to Vermilion North Gate (18, 0)
    print("[+] Step 1: Navigating Vermilion City to North Gate (18, 0)...")
    for s in range(75):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m != 5:
            print(f"[+] Left Vermilion City, now at Map {m} ({x}, {y}) at step {s}")
            break
        if (x, y) == (18, 0):
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

    # 2. Gatehouse 74: walk from (4, 7) to (4, 0)
    print("[+] Step 2: Passing through Gatehouse 74...")
    for s in range(25):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m == 17:
            print(f"[+] Entered Route 6 at ({x}, {y})")
            break
        press("up", wait=18)

    # 3. On Route 6: walk to grass at (13, 31)
    print("[+] Step 3: Walking to Route 6 grass at (13, 31)...")
    for s in range(20):
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if (x, y) == (13, 31):
            print(f"[+] Reached grass at ({x}, {y})")
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (13, 31))
        dx = wp[0] - x
        dy = wp[1] - y
        if dy < 0: press("up")
        elif dy > 0: press("down")
        elif dx > 0: press("right")
        elif dx < 0: press("left")
        else: press("up")

    # 4. Patrol grass to trigger wild encounter
    print("[+] Step 4: Patrol grass until encounter...")
    patrol = ["left", "right"]
    pidx = 0
    for s in range(100):
        in_battle = pb.memory[RAMMap.IS_IN_BATTLE] != 0
        if in_battle:
            species = pb.memory[RAMMap.ENEMY_SPECIES]
            print(f"[+] Wild battle triggered at step {s}! Opponent species: {hex(species)}")
            break
        press(patrol[pidx % 2], wait=18)
        pidx += 1

    pb.stop()

if __name__ == "__main__":
    run()
