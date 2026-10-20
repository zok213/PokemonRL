"""
catch_oddish_perfect.py — Exact, 100% authentic capture of Oddish on Route 6
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

    # Allow ship cutscene to finish and settle in Vermilion City
    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    for _ in range(120):
        pb.tick()

    print(f"[+] Started at Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    def press(btn: str, hold: int = 8, wait: int = 20):
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

    # 1. Walk from Vermilion City (18, 29) to North Gate (18, 0)
    print("[+] Step 1: Navigating Vermilion City to North Gate (18, 0)...")
    for s in range(75):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m != 5:
            print(f"[+] Entered Gatehouse 74 at ({x}, {y})")
            break
        if (x, y) == (18, 0):
            press("up", wait=25)
            continue
        wp = nav._bfs_next_waypoint(5, (x, y), (18, 0))
        dx = wp[0] - x
        dy = wp[1] - y
        if dy < 0: press("up")
        elif dy > 0: press("down")
        elif dx > 0: press("right")
        elif dx < 0: press("left")
        else: press("up")

    # 2. Walk through Gatehouse 74 into Route 6 (Map 17)
    print("[+] Step 2: Passing Gatehouse 74...")
    for s in range(25):
        m = pb.memory[RAMMap.CUR_MAP]
        if m == 17:
            print(f"[+] Entered Route 6 at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")
            break
        press("up", wait=18)

    # 3. Walk to grass at (13, 31)
    print("[+] Step 3: Walking to Route 6 grass at (13, 31)...")
    for s in range(20):
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if (x, y) == (13, 31):
            print(f"[+] Arrived in grass at ({x}, {y})")
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (13, 31))
        dx = wp[0] - x
        dy = wp[1] - y
        if dy < 0: press("up")
        elif dy > 0: press("down")
        elif dx > 0: press("right")
        elif dx < 0: press("left")
        else: press("up")

    # 4. Patrol grass and handle wild combat
    print("[+] Step 4: Patrolling grass and catching Oddish...")
    patrol = ["left", "right"]
    pidx = 0
    in_bag = False

    for step in range(500):
        party_count = pb.memory[RAMMap.PARTY_COUNT]
        if party_count >= 2:
            print(f"\n{'='*65}\n  [+] ODDISH SUCCESSFULLY CAPTURED! Party Count: {party_count}\n{'='*65}\n")
            break

        in_battle = pb.memory[RAMMap.IS_IN_BATTLE] != 0
        if not in_battle:
            in_bag = False
            # Clear any text box or dialog left over
            if pb.memory[0xCF13] != 0:
                press("b", wait=18)
            else:
                press(patrol[pidx % 2], wait=18)
                pidx += 1
            continue

        # In battle!
        species = pb.memory[RAMMap.ENEMY_SPECIES]
        menu_y = pb.memory[0xCC24]
        menu_x = pb.memory[0xCC25]
        cur_item = pb.memory[0xCC26]
        text_active = pb.memory[0xCF13] != 0

        # Case A: Opponent is Oddish (0xB9)
        if species == 0xB9:
            # If in Bag menu (wListMenuID or menu_y in (2, 4))
            if menu_y in (2, 4):
                # Poké Ball is at cursor 0
                if cur_item == 0:
                    print("  [*] Bag open! Using Poké Ball (slot 0)...")
                    press("a", wait=40)
                    in_bag = True
                else:
                    press("up", wait=20)
            elif in_bag:
                # Capture animation and text ("All right! ODDISH was caught!") -> advance with A
                press("a", wait=35)
            elif menu_y == 14:  # Main battle menu
                if menu_x >= 10:  # in right column (PKMN/RUN) -> move LEFT
                    press("left", wait=20)
                elif cur_item == 0:  # on FIGHT -> move DOWN to ITEM
                    press("down", wait=20)
                elif cur_item == 1:  # on ITEM! -> select with Button A
                    print("  [*] Oddish encountered! Selecting ITEM menu...")
                    press("a", wait=35)
            else:
                # Battle intro text ("Wild ODDISH appeared!", "Go! WARTORTLE!") -> advance with A
                press("a", wait=30)

        # Case B: Opponent is not Oddish (Mankey/Pidgey) -> Flee!
        else:
            if menu_y == 14:  # Main battle menu
                if menu_x < 10:  # in left column (FIGHT/ITEM) -> move RIGHT to (PKMN/RUN)
                    press("right", wait=20)
                elif cur_item == 0:  # on PKMN -> move DOWN to RUN
                    press("down", wait=20)
                elif cur_item == 1:  # on RUN! -> flee with Button A
                    press("a", wait=35)
                else:
                    press("b", wait=20)
            else:
                # Advancing text or flee message ("Got away safely!")
                press("b", wait=20)

    party_count = pb.memory[RAMMap.PARTY_COUNT]
    species_list = [pb.memory[RAMMap.PARTY_COUNT + 1 + i] for i in range(party_count)]
    print(f"[+] Party Status: Count={party_count}, Species={[hex(s) for s in species_list]}")

    if party_count >= 2:
        out_state = REPO_ROOT / "saves" / "oddish_caught.state"
        out_state.parent.mkdir(parents=True, exist_ok=True)
        with open(out_state, "wb") as f:
            pb.save_state(f)
        print(f"[+] Saved state with Oddish to: {out_state}")

    pb.stop()
    return party_count >= 2

if __name__ == "__main__":
    run()
