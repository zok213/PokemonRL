"""
test_pewter_to_route4.py — Navigate Pewter City -> Route 3 -> Route 4
"""
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
    with open(REPO_ROOT / "saves" / "pewter_via_diglett.state", "rb") as f:
        pb.load_state(f)

    def press(btn: str, hold: int = 8, wait: int = 14):
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

    def handle_battle():
        while pb.memory[0xD057] != 0:
            # Battle active
            text_id = pb.memory[0xCF13]
            menu_type = pb.memory[0xCC24]
            cur_item = pb.memory[0xCC26]
            # Try to run: press B, select RUN
            press("b", hold=4, wait=10)
            press("down", hold=4, wait=10)
            press("right", hold=4, wait=10)
            press("a", hold=6, wait=25)
            # Advance any text
            press("a", hold=4, wait=15)

    nav = WorldGraphNavigator()

    print(f"Starting at Map {pb.memory[0xD35E]} ({pb.memory[0xD362]}, {pb.memory[0xD361]})")

    # Step 1: Pewter City to Route 3 (target 39, 17)
    print("[1] Walking across Pewter City towards Route 3...")
    for step in range(120):
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 14: # Reached Route 3!
            print(f"[+] Reached Route 3! Pos: ({x}, {y}) in {step} steps.")
            break
        
        if (x, y) == (39, 17):
            press("right", hold=10, wait=25)
            continue
            
        wp = nav._bfs_next_waypoint(2, (x, y), (39, 17), lookahead=1)
        dx = wp[0] - x
        dy = wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # Step 2: Route 3 to Route 4 pass (target 59, 0)
    print("[2] Walking Route 3 towards Route 4 pass...")
    for step in range(300):
        if pb.memory[0xD057] != 0:
            print(f"Wild battle encountered at step {step}! Fleeing...")
            handle_battle()
            print("Escaped battle.")

        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 15: # Reached Route 4!
            print(f"[+] Reached Route 4! Pos: ({x}, {y}) in {step} steps.")
            break

        if (x, y) == (59, 0):
            press("up", hold=10, wait=25)
            continue

        wp = nav._bfs_next_waypoint(14, (x, y), (59, 0), lookahead=1)
        dx = wp[0] - x
        dy = wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    print(f"Current State: Map {pb.memory[0xD35E]} Pos ({pb.memory[0xD362]}, {pb.memory[0xD361]})")
    pb.stop()

if __name__ == "__main__":
    main()
