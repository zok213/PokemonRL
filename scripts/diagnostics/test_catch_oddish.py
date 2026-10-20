"""
test_catch_oddish.py — Autonomous Oddish Encounter, Capture, and Party Verification
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
    print("[+] Step 1: Walking to North Gate (18, 0)...")
    for s in range(75):
        m = pb.memory[RAMMap.CUR_MAP]
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if m != 5:
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

    # 2. Gatehouse 74
    print("[+] Step 2: Passing Gatehouse 74...")
    for s in range(25):
        m = pb.memory[RAMMap.CUR_MAP]
        if m == 17:
            break
        press("up", wait=18)

    # 3. Walk to grass at (13, 31)
    print("[+] Step 3: Entering grass at (13, 31)...")
    for s in range(20):
        x = pb.memory[RAMMap.X_POS]
        y = pb.memory[RAMMap.Y_POS]
        if (x, y) == (13, 31):
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (13, 31))
        dx = wp[0] - x
        dy = wp[1] - y
        if dy < 0: press("up")
        elif dy > 0: press("down")
        elif dx > 0: press("right")
        elif dx < 0: press("left")
        else: press("up")

    # 4. Search and catch loop
    print("[+] Step 4: Patrol and catch Oddish...")
    patrol = ["left", "right"]
    pidx = 0
    caught = False
    for step in range(300):
        party_count = pb.memory[RAMMap.PARTY_COUNT]
        if party_count >= 2:
            print(f"[+] CAUGHT! Party size is now {party_count}!")
            caught = True
            break

        in_battle = pb.memory[RAMMap.IS_IN_BATTLE] != 0
        if not in_battle:
            press(patrol[pidx % 2], wait=18)
            pidx += 1
            continue

        # In battle!
        species = pb.memory[RAMMap.ENEMY_SPECIES]
        menu_y = pb.memory[0xCC24]
        menu_x = pb.memory[0xCC25]
        cur_item = pb.memory[0xCC26]
        text_box = pb.memory[0xCF13]
        print(f"Battle step {step}: species={hex(species)}, menu_y={menu_y}, menu_x={menu_x}, cur={cur_item}, text={text_box}", flush=True)

        if species == 0xB9:  # ODDISH!
            print(f"[*] Oddish encountered! Throwing Poke Ball... (menu_y={menu_y}, menu_x={menu_x}, cur={cur_item})")
            # If in root menu (menu_y == 14)
            if menu_y == 14:
                # ITEM is down from FIGHT
                if menu_x < 10 and cur_item == 0:  # on FIGHT
                    press("down", wait=20)
                elif menu_x >= 10:  # in right column (PKMN/RUN) -> go left
                    press("left", wait=20)
                else:  # on ITEM!
                    press("a", wait=30)
            elif menu_y == 4 or menu_y == 2:  # in Bag menu during battle
                # Poke Ball is item 0 -> press A to use!
                press("a", wait=40)
            else:
                # Text or ball throwing animation
                press("a", wait=30)
        else:
            # Not Oddish (Mankey/Pidgey): Run away!
            if menu_y == 14:
                if menu_x < 10:
                    press("right", wait=20)
                elif cur_item == 0:
                    press("down", wait=20)
                elif cur_item == 1:
                    press("a", wait=30)
                else:
                    press("b", wait=20)
            else:
                press("b", wait=20)

    party_count = pb.memory[RAMMap.PARTY_COUNT]
    species_list = [pb.memory[RAMMap.PARTY_COUNT + 1 + i] for i in range(party_count)]
    print(f"[+] Final party count: {party_count}, Species: {[hex(s) for s in species_list]}")

    if party_count >= 2:
        out_state = REPO_ROOT / "saves" / "oddish_caught.state"
        with open(out_state, "wb") as f:
            pb.save_state(f)
        print(f"[+] Milestone saved to: {out_state}")

    pb.stop()

if __name__ == "__main__":
    run()
