from pathlib import Path
import sys
from pyboy import PyBoy
from pyboy.utils import WindowEvent

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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

    def cut_tree_ahead():
        press("start", wait=35)
        while pb.memory[0xCC26] > 1: press("up", wait=18)
        while pb.memory[0xCC26] < 1: press("down", wait=18)
        press("a", wait=35)
        while pb.memory[0xCC26] > 1: press("up", wait=18)
        while pb.memory[0xCC26] < 1: press("down", wait=18)
        press("a", wait=35)
        press("a", wait=50)
        for _ in range(6): press("a", wait=25)
        for _ in range(60): pb.tick()

    def run_fsm_battle(max_steps: int = 200, debug: bool = False):
        for s in range(max_steps):
            in_battle = pb.memory[RAMMap.IS_IN_BATTLE]
            if in_battle == 0:
                print(f"  [Battle] Ended at step {s}!", flush=True)
                break

            tb = pb.memory[0xCF13]
            h_ui = pb.memory[0xFFF6]
            m_y = pb.memory[0xCC24]
            m_x = pb.memory[0xCC25]
            cur = pb.memory[0xCC26]
            php = pb.memory[0xD015] * 256 + pb.memory[0xD016]
            ehp = pb.memory[0xCFE6] * 256 + pb.memory[0xCFE7]
            pps = [pb.memory[0xD02D + i] for i in range(4)]

            if debug or s % 10 == 0:
                print(f"  Step {s:3d}: InBattle={in_battle}, tb=0x{tb:02X}, h_ui=0x{h_ui:02X}, menu=({m_y},{m_x}), cur={cur}, php={php}, ehp={ehp}, pps={pps}", flush=True)

            if (h_ui & 0x02) != 0:
                if pps[0] > 0: target = 1
                elif pps[3] > 0: target = 4
                elif pps[2] > 0: target = 3
                else: target = 1

                if cur == target:
                    press("a", wait=25)
                elif cur < target:
                    press("down", wait=16)
                else:
                    press("up", wait=16)
            elif (m_y in (7, 8) and m_x == 1) or tb == 0x14:
                if cur == 0:
                    press("down", wait=16)
                else:
                    press("a", wait=22)
            elif tb == 0x0B or (m_y == 14 and m_x == 9):
                press("up", wait=14)
                press("left", wait=14)
                press("a", wait=22)
            elif m_y == 1 and m_x == 0:
                if php == 0:
                    press("a", wait=25)
                else:
                    press("b", wait=20)
            else:
                press("a", wait=20)

        for _ in range(30):
            if pb.memory[0xCF13] == 0 and pb.memory[RAMMap.IS_IN_BATTLE] == 0:
                break
            press("b", wait=18)

    nav = WorldGraphNavigator()

    # Step to PokeCenter
    for _ in range(50):
        if pb.memory[RAMMap.CUR_MAP] == 81: break
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (11, 19): press("up", wait=30); continue
        wp = nav._bfs_next_waypoint(21, (cx, cy), (11, 19), lookahead=1, can_cut=False)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # Heal
    for _ in range(15):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (3, 2): break
        wp = nav._bfs_next_waypoint(81, (cx, cy), (3, 2), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    press("up", wait=20)
    press("a", wait=35)
    for _ in range(12): press("a", wait=30)
    for _ in range(120): pb.tick()
    for _ in range(8): press("b", wait=22)

    # Exit
    for _ in range(6): press("down", wait=20)
    press("down", wait=35)
    for _ in range(40): pb.tick()

    # Cut trees and enter
    for _ in range(10):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (9, 21): break
        wp = nav._bfs_next_waypoint(21, (cx, cy), (9, 21), lookahead=1, can_cut=False)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    press("up", wait=20); cut_tree_ahead(); press("up", wait=25)
    press("up", wait=20); cut_tree_ahead(); press("up", wait=25)
    press("left", wait=25); press("up", wait=35)
    for _ in range(60): pb.tick()

    print(f"In Map {pb.memory[RAMMap.CUR_MAP]} at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    # Partition 1: to (37, 3)
    for step in range(250):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            run_fsm_battle()
        if pb.memory[RAMMap.CUR_MAP] == 232: break
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (37, 3):
            press("up", wait=30)
            for _ in range(30): pb.tick()
            continue
        wp = nav._bfs_next_waypoint(82, (cx, cy), (37, 3), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    print(f"Reached Map {pb.memory[RAMMap.CUR_MAP]} at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    # Now Partition 2: walk towards (26, 31) and debug battle!
    for step in range(150):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            print("=== Triggered Battle in Partition 2 ===", flush=True)
            run_fsm_battle(max_steps=150, debug=True)
            print("Battle debug done. Current InBattle:", pb.memory[RAMMap.IS_IN_BATTLE], flush=True)
            break
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        wp = nav._bfs_next_waypoint(232, (cx, cy), (27, 3), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    pb.stop()

if __name__ == "__main__":
    main()
