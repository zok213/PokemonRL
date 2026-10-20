"""
solve_from_bicycle_to_route9.py — From Bike Shop with Bicycle to Route 9
1. Load saves/bicycle_acquired.state (Bike Shop Map 66, Pos 6, 4)
2. Close dialogue window until hWY (0xFF4A) == 0x90
3. Navigate to door at (2, 7) and step down into Cerulean City (Map 3)
4. Navigate to cut tree approach at (19, 27) facing DOWN
5. Cut tree at (19, 28) using Oddish HM01 Cut
6. Walk through cut tree to row 29, col 36, row 17, (39, 17)
7. Step east into Route 9 (Map 20)
8. Verify Map 20 and save saves/route9_entered.state + saves/checkpoint_progress.state
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

def main():
    rom_candidates = [
        REPO_ROOT / "roms" / "pokemon_red.gb",
        REPO_ROOT / "PokemonRed.gb",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "PokemonRed.gb",
    ]
    rom_path = next(str(p) for p in rom_candidates if p.exists())

    pb = PyBoy(rom_path, window="null")
    with open(REPO_ROOT / "saves" / "bicycle_acquired.state", "rb") as f:
        pb.load_state(f)

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

    print("=== Step 1: Confirm Bicycle & Dismiss Bike Shop Dialogue ===", flush=True)
    bag_count = pb.memory[0xD31D]
    bag_items = [pb.memory[0xD31E + i*2] for i in range(bag_count)]
    print(f"Bag items: {bag_items}", flush=True)
    assert RAMMap.ITEM_BICYCLE in bag_items, f"Bicycle not in bag! {bag_items}"

    # Dismiss clerk dialogue until hWY == 0x90
    for i in range(10):
        hwy = pb.memory[0xFF4A]
        if hwy == 0x90:
            print(f"[+] Text box closed (hWY == 0x90) on iteration {i}", flush=True)
            break
        press("a", hold=6, wait=25)

    assert pb.memory[0xFF4A] == 0x90, "Failed to close text window!"

    print("\n=== Step 2: Exit Bike Shop to Cerulean City ===", flush=True)
    for step in range(25):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 3:
            print(f"[+] Successfully exited to Cerulean City at ({x}, {y}) in {step} steps.", flush=True)
            break
        if (x, y) in ((2, 7), (3, 7)):
            press("down", hold=10, wait=30)
            continue
        wp = nav._bfs_next_waypoint(66, (x, y), (2, 7), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[0xD35E] == 3, f"Expected Cerulean City (Map 3), got Map {pb.memory[0xD35E]}"

    print("\n=== Step 3: Navigate to Cerulean South Cut Tree (19, 27) ===", flush=True)
    for step in range(35):
        x, y = pb.memory[0xD362], pb.memory[0xD361]
        if (x, y) == (19, 27):
            print(f"[+] Reached Cut tree approach at ({x}, {y}) in {step} steps.", flush=True)
            break
        wp = nav._bfs_next_waypoint(3, (x, y), (19, 27), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert (pb.memory[0xD362], pb.memory[0xD361]) == (19, 27), "Failed to reach cut tree approach at (19, 27)!"

    print("\n=== Step 4: Cut Tree with Oddish HM01 Cut ===", flush=True)
    press("down", wait=20) # Face tree at (19, 28)
    press("start", wait=35) # Open Start Menu

    # Cursor to PKMN (item 1)
    for _ in range(8):
        cur = pb.memory[0xCC26]
        if cur == 1: break
        if cur > 1: press("up", wait=18)
        else: press("down", wait=18)
    press("a", wait=35) # Open PKMN menu

    # Cursor to Oddish (slot 1)
    for _ in range(8):
        cur = pb.memory[0xCC26]
        if cur == 1: break
        if cur > 1: press("up", wait=18)
        else: press("down", wait=18)
    press("a", wait=35) # Open Oddish action sub-menu (CUT is slot 0)

    # Select CUT
    press("a", wait=50)

    # Advance "ODDISH used CUT!" dialogue
    for _ in range(6): press("a", wait=25)
    for _ in range(60): pb.tick() # Wait for cut animation to complete

    # Step down through cut tree
    print("[+] Stepping south through cut tree...", flush=True)
    press("down", wait=25)
    press("down", wait=25)
    print(f"[+] Position past tree: ({pb.memory[0xD362]}, {pb.memory[0xD361]})", flush=True)
    assert pb.memory[0xD361] >= 29, "Failed to step through cut tree!"

    print("\n=== Step 5: Navigate to Route 9 East Border (39, 17) ===", flush=True)
    for step in range(60):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 20: # Route 9!
            print(f"[+] Entered Route 9! Pos: ({x}, {y}) in {step} steps.", flush=True)
            break
        if (x, y) in ((39, 16), (39, 17)):
            press("right", hold=10, wait=30)
            continue
        wp = nav._bfs_next_waypoint(3, (x, y), (39, 17), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[0xD35E] == 20, f"Expected Route 9 (Map 20), got Map {pb.memory[0xD35E]}"
    print(f"[+] Confirmed on Route 9 at ({pb.memory[0xD362]}, {pb.memory[0xD361]})!", flush=True)

    # Save checkpoints
    r9_save = REPO_ROOT / "saves" / "route9_entered.state"
    with open(r9_save, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to {r9_save}", flush=True)

    chk_path = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(chk_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Updated primary checkpoint: {chk_path}", flush=True)

    pb.stop()
    print("\n=== Route 9 Traversal Milestone Completed! ===", flush=True)

if __name__ == "__main__":
    main()
