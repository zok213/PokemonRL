"""
capture_cut_and_enter_gym.py — 100% Autonomous, Authentic Pipeline:
1. S.S. Anne exit -> Vermilion North Gate -> Route 6
2. Encounter & Catch Oddish (Party Count = 2)
3. Teach HM01 Cut to Oddish
4. Save saves/oddish_caught.state
5. Walk to Vermilion Gym cut tree at (15, 18)
6. Execute Cut via Oddish
7. Enter Vermilion Gym (Map 92)
8. Save saves/vermilion_gym_entered.state
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

    # Allow ship departure cutscene to finish and settle in Vermilion City
    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    for _ in range(120):
        pb.tick()

    def press(btn: str, hold: int = 8, wait: int = 18):
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

    print(f"[+] 1. Navigating from Vermilion City ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}) to North Gate (18, 0)...")
    for _ in range(54):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] != 5:
            break
        wp = nav._bfs_next_waypoint(5, (x, y), (18, 0))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    print("[+] 2. Passing through Gatehouse 74...")
    for _ in range(6):
        press("up")

    print("[+] 3. Walking to Route 6 grass at (13, 30)...")
    for _ in range(12):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (13, 30):
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (13, 30))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    print(f"[+] 4. Patrolling grass at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})...")
    for s in range(50):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            print(f"[+] Battle triggered at patrol step {s}! Opponent: {hex(pb.memory[RAMMap.ENEMY_SPECIES])}")
            break
        press("right" if s % 2 == 0 else "left")

    print("[+] 5. Catching Oddish...")
    for battle_step in range(150):
        party = pb.memory[RAMMap.PARTY_COUNT]
        in_b = pb.memory[RAMMap.IS_IN_BATTLE]
        if party >= 2 and not in_b:
            print(f"[+] Capture complete! Party = {party} at battle step {battle_step}")
            break

        m_y = pb.memory[0xCC24]
        m_x = pb.memory[0xCC25]
        cur = pb.memory[0xCC26]
        lm = pb.memory[0xCF94]

        if lm == 3:  # Bag open -> Slot 0 is Poké Ball
            if cur == 0:
                press("a", wait=35)
            else:
                press("up", wait=20)
        elif m_y == 14:  # Main battle menu
            if m_x >= 10:
                press("left", wait=20)
            elif cur == 0:
                press("down", wait=20)
            elif cur == 1:
                press("a", wait=35)
        elif m_y == 12:  # Move sub-menu
            press("b", wait=20)
        else:  # Text dialog / animations / Pokedex / nickname
            press("b", wait=22)

    # Wait for JoyIgnore to drop to 0
    while pb.memory[RAMMap.JOY_IGNORE] != 0:
        pb.tick()
    for _ in range(30):
        pb.tick()

    party_cnt = pb.memory[RAMMap.PARTY_COUNT]
    species_list = [hex(pb.memory[RAMMap.PARTY_COUNT + 1 + i]) for i in range(party_cnt)]
    print(f"[+] Party Status: Count={party_cnt}, Species={species_list}")
    assert party_cnt == 2 and species_list == ['0xb3', '0xb9'], "Oddish capture failed!"

    print("[+] 6. Teaching HM01 Cut to Oddish...")
    press("start", wait=35)
    press("down", wait=20)
    press("down", wait=20)
    press("a", wait=35)  # Open Bag

    for _ in range(8):
        press("down", wait=18)  # Scroll to HM01 Cut (slot 8)
    press("a", wait=35)  # Open USE/TOSS menu
    press("a", wait=40)  # Select USE

    # Advance 'Booted up an HM!' and 'It contained CUT!'
    press("a", wait=35)
    press("a", wait=35)
    press("a", wait=45)  # YES to 'Teach CUT to a POKEMON?'

    # Party menu: move down to Oddish (slot 1)
    press("down", wait=25)
    press("a", wait=45)  # Select Oddish

    # Advance 'ODDISH learned CUT!'
    press("a", wait=40)
    press("a", wait=40)

    # Close menus back to overworld
    for _ in range(8):
        press("b", wait=20)

    oddish_moves = [pb.memory[0xD19F + i] for i in range(4)]
    print(f"[+] Oddish moves after teaching: {oddish_moves}")
    assert 15 in oddish_moves, "Cut not taught to Oddish!"

    # Save oddish_caught.state
    out_oddish = REPO_ROOT / "saves" / "oddish_caught.state"
    with open(out_oddish, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to: {out_oddish}")

    # 7. Navigate from Route 6 back to Vermilion City
    print(f"[+] 7. Navigating Route 6 from ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}) to Gatehouse 74 (8, 35)...")
    for _ in range(30):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] != 17:
            break
        if (x, y) == (8, 35):
            press("down", wait=20)
            break
        wp = nav._bfs_next_waypoint(17, (x, y), (8, 35))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    print(f"[+] 8. Passing through Gatehouse 74 from ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})...")
    for _ in range(10):
        if pb.memory[RAMMap.CUR_MAP] == 5:
            break
        press("down", wait=18)

    print(f"[+] 9. Arrived in Vermilion City at Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}). Navigating to Cut tree at (15, 17)...")
    for _ in range(40):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (15, 17):
            break
        wp = nav._bfs_next_waypoint(5, (x, y), (15, 17))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    # Face DOWN towards the tree at (15, 18)
    press("down", wait=20)
    print(f"[+] Facing Gym cut tree: Pos=({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}), Facing={pb.memory[RAMMap.PLAYER_FACING]}")

    print("[+] 10. Executing CUT via Party Menu (Oddish)...")
    press("start", wait=35)
    press("down", wait=20)  # Move to PKMN
    press("a", wait=35)     # Open Party Menu
    press("down", wait=20)  # Move to Oddish (slot 1)
    press("a", wait=35)     # Open Oddish sub-menu (CUT is item 0)
    press("a", wait=50)     # Select CUT!

    # Advance Cut text and animation
    for _ in range(4):
        press("a", wait=30)
    for _ in range(60):
        pb.tick()

    print(f"[+] Tree cut! Player pos: ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    print("[+] 11. Stepping through cut tree path into Gym door at (12, 19)...")
    press("down", wait=22)  # Step onto (15, 18)
    press("down", wait=22)  # Step onto (15, 19)
    for _ in range(3):
        press("left", wait=20)  # Walk left along row 19 to (12, 19)
    press("up", wait=25)    # Step up into Vermilion Gym door

    # Wait for Gym map to load
    for _ in range(40):
        pb.tick()

    current_map = pb.memory[RAMMap.CUR_MAP]
    cur_x = pb.memory[RAMMap.X_POS]
    cur_y = pb.memory[RAMMap.Y_POS]
    print(f"[+] Current Map: {current_map} ({cur_x}, {cur_y})")

    if current_map == 92:
        print("\n" + "="*65)
        print("  [+] MILESTONE ACHIEVED: VERMILION GYM ENTERED! (Map 92)")
        print("="*65 + "\n")
        out_gym = REPO_ROOT / "saves" / "vermilion_gym_entered.state"
        with open(out_gym, "wb") as f:
            pb.save_state(f)
        print(f"[+] Saved gym entry state to: {out_gym}")

    pb.stop()
    return current_map == 92

if __name__ == "__main__":
    main()
