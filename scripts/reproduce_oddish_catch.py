"""
reproduce_oddish_catch.py — Replicate exact deterministic inputs from task-15476 and capture Oddish
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
    with open(REPO_ROOT / "saves" / "checkpoint_progress.state", "rb") as f:
        pb.load_state(f)

    # 1. Wait for ship departure cutscene
    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    for _ in range(120):
        pb.tick()

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

    # Walk from Vermilion City to North Gate
    for _ in range(54):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] != 5:
            break
        wp = nav._bfs_next_waypoint(5, (x, y), (18, 0))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    # Pass through gatehouse 74
    for _ in range(6):
        press("up")

    # Walk to grass at (13, 30)
    for _ in range(12):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (13, 30):
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (13, 30))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    print(f"[+] Reached position: Map={pb.memory[RAMMap.CUR_MAP]}, X={pb.memory[RAMMap.X_POS]}, Y={pb.memory[RAMMap.Y_POS]}")

    # Patrol until battle
    battle_started = False
    for s in range(50):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            print(f"[+] Battle triggered at step {s}! Opponent: {hex(pb.memory[RAMMap.ENEMY_SPECIES])}")
            battle_started = True
            break
        press("right" if s % 2 == 0 else "left")

    if not battle_started:
        print("[-] Battle did not trigger!")
        pb.stop()
        return

    # In battle!
    print("[+] Handling battle...")
    ball_thrown = False
    for bstep in range(150):
        in_battle = pb.memory[RAMMap.IS_IN_BATTLE] != 0
        party_cnt = pb.memory[RAMMap.PARTY_COUNT]
        
        if not in_battle and party_cnt >= 2:
            print(f"[+] Battle finished! Party Count: {party_cnt}")
            break

        menu_y = pb.memory[0xCC24]
        menu_x = pb.memory[0xCC25]
        cur_item = pb.memory[0xCC26]
        list_menu = pb.memory[0xCF94]
        text_box = pb.memory[0xCF13]
        species = pb.memory[RAMMap.ENEMY_SPECIES]

        print(f"bstep {bstep:02d}: species={hex(species)} menu_y={menu_y} menu_x={menu_x} cur={cur_item} list_menu={list_menu} in_battle={in_battle} ball_thrown={ball_thrown}")

        if not ball_thrown:
            if list_menu == 3:  # ITEMLISTMENU is open!
                if cur_item == 0:  # Poké Ball (slot 0)
                    print("  [*] Bag open! Using Poké Ball (slot 0)...")
                    press("a", wait=40)
                    ball_thrown = True
                else:
                    press("up", wait=20)
            elif menu_y == 14:  # Main battle menu
                if menu_x >= 10:  # In right column (PKMN/RUN) -> move LEFT
                    press("left", wait=20)
                elif cur_item == 0:  # On FIGHT -> move DOWN to ITEM
                    press("down", wait=20)
                elif cur_item == 1:  # On ITEM! -> select with A
                    print("  [*] On ITEM menu! Pressing A...")
                    press("a", wait=35)
            elif menu_y == 12:  # In Move menu or sub-menu -> cancel back with B
                press("b", wait=20)
            else:
                # Advancing text box (Wild Oddish appeared, Go Wartortle)
                # In Gen 1, A or B advances text
                press("b", wait=20)
        else:
            # Ball was thrown! Press B to decline nickname, advance text, and complete battle
            press("b", wait=25)

    final_party = pb.memory[RAMMap.PARTY_COUNT]
    print(f"\n[+] Final Party Count: {final_party}")
    if final_party >= 2:
        species_list = [pb.memory[RAMMap.PARTY_COUNT + 1 + i] for i in range(final_party)]
        print(f"[+] Party Species: {[hex(s) for s in species_list]}")
        out_state = REPO_ROOT / "saves" / "oddish_caught.state"
        with open(out_state, "wb") as f:
            pb.save_state(f)
        print(f"[+] Successfully saved {out_state}!")

    pb.stop()

if __name__ == "__main__":
    main()
