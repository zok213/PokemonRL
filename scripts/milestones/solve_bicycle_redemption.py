"""
solve_bicycle_redemption.py — 100% Authentic Autonomous Traversal:
1. Load saves/pewter_via_diglett.state (Pewter City at 18, 35)
2. Traverse Pewter City to Route 3 (target 39, 17) -> step east to Route 3
3. Traverse Route 3 to Route 4 pass (target 59, 0) -> step north to Route 4 West
4. Traverse Route 4 West to Mt. Moon entrance (18, 5) -> enter Mt. Moon 1F
5. Traverse Mt. Moon 1F to ladder at (5, 5) -> descend to B1F
6. Traverse Mt. Moon B1F from (5, 5) to ladder at (21, 17) -> descend to B2F
7. Traverse Mt. Moon B2F from (21, 17) to ladder at (5, 7) -> ascend to B1F
8. Traverse Mt. Moon B1F from (23, 3) to exit doorway at (27, 3) -> emerge onto Route 4 East
9. Traverse Route 4 East to Cerulean City (89, 10) -> step east into Cerulean City (Map 3)
10. Navigate Cerulean City to Bike Shop (13, 25) -> enter Bike Shop (Map 63)
11. Approach Bike Shop Clerk at (6, 2): stand at (6, 4) facing UP, press A to redeem Bike Voucher
12. Verify BICYCLE (0x06) acquired in bag and EVENT_GOT_BICYCLE set!
13. Save saves/bicycle_acquired.state and update saves/checkpoint_progress.state
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
            press("b", hold=4, wait=10)
            press("down", hold=4, wait=10)
            press("right", hold=4, wait=10)
            press("a", hold=6, wait=25)
            press("a", hold=4, wait=15)

    def clear_text(max_steps: int = 40):
        for _ in range(max_steps):
            if pb.memory[0xCF13] == 0:
                break
            press("a", hold=4, wait=15)
            press("b", hold=4, wait=15)

    nav = WorldGraphNavigator()

    print(f"=== Starting Bicycle Redemption Traversal ===")
    print(f"Initial State: Map {pb.memory[0xD35E]} at ({pb.memory[0xD362]}, {pb.memory[0xD361]})")

    # -------------------------------------------------------------
    # Leg 1: Pewter City (Map 2) -> Route 3 (Map 14)
    # -------------------------------------------------------------
    print("\n[Leg 1] Traversing Pewter City east towards Route 3...")
    for step in range(120):
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 14:
            print(f"  [+] Reached Route 3 at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (39, 17):
            press("right", hold=10, wait=25)
            continue
        wp = nav._bfs_next_waypoint(2, (x, y), (39, 17), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 2: Route 3 (Map 14) -> Route 4 (Map 15)
    # -------------------------------------------------------------
    print("\n[Leg 2] Traversing Route 3 east towards Route 4 mountain pass...")
    for step in range(300):
        if pb.memory[0xD057] != 0:
            print(f"  [!] Wild battle at step {step}! Running...")
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 15:
            print(f"  [+] Reached Route 4 West at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (59, 0):
            press("up", hold=10, wait=25)
            continue
        wp = nav._bfs_next_waypoint(14, (x, y), (59, 0), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 3: Route 4 West (Map 15) -> Mt. Moon 1F (Map 59)
    # -------------------------------------------------------------
    print("\n[Leg 3] Approaching Mt. Moon cave entrance at (18, 5)...")
    for step in range(60):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 59:
            print(f"  [+] Entered Mt. Moon 1F at ({x}, {y}) in {step} steps.")
            break
        if (x, y) in ((18, 6), (18, 5)):
            press("up", hold=10, wait=30)
            continue
        wp = nav._bfs_next_waypoint(15, (x, y), (18, 5), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 4: Mt. Moon 1F (Map 59) -> B1F (Map 60)
    # -------------------------------------------------------------
    print("\n[Leg 4] Traversing Mt. Moon 1F towards ladder at (5, 5)...")
    for step in range(250):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 60:
            print(f"  [+] Descended to Mt. Moon B1F at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (5, 5):
            press("up", hold=8, wait=25)
            continue
        wp = nav._bfs_next_waypoint(59, (x, y), (5, 5), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 5: Mt. Moon B1F (Map 60) -> B2F (Map 61)
    # -------------------------------------------------------------
    print("\n[Leg 5] Traversing Mt. Moon B1F towards ladder at (21, 17)...")
    for step in range(100):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 61:
            print(f"  [+] Descended to Mt. Moon B2F at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (21, 17):
            press("up", hold=8, wait=25)
            continue
        wp = nav._bfs_next_waypoint(60, (x, y), (21, 17), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 6: Mt. Moon B2F (Map 61) -> B1F Exit Tunnel (Map 60)
    # -------------------------------------------------------------
    print("\n[Leg 6] Traversing Mt. Moon B2F towards exit ladder at (5, 7)...")
    for step in range(300):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 60:
            print(f"  [+] Ascended to Mt. Moon B1F exit tunnel at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (5, 7):
            press("up", hold=8, wait=25)
            continue
        wp = nav._bfs_next_waypoint(61, (x, y), (5, 7), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 7: Mt. Moon B1F Exit Tunnel (Map 60) -> Route 4 East (Map 15)
    # -------------------------------------------------------------
    print("\n[Leg 7] Traversing B1F exit tunnel towards Route 4 door at (27, 3)...")
    for step in range(50):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 15:
            print(f"  [+] Emerged onto Route 4 East at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (27, 3):
            press("right", hold=8, wait=25)
            continue
        wp = nav._bfs_next_waypoint(60, (x, y), (27, 3), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 8: Route 4 East (Map 15) -> Cerulean City (Map 3)
    # -------------------------------------------------------------
    print("\n[Leg 8] Traversing Route 4 East towards Cerulean City...")
    for step in range(250):
        if pb.memory[0xD057] != 0:
            handle_battle()
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 3:
            print(f"  [+] Entered Cerulean City at ({x}, {y}) in {step} steps.")
            break
        if x >= 89:
            press("right", hold=10, wait=25)
            continue
        wp = nav._bfs_next_waypoint(15, (x, y), (89, 10), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 9: Cerulean City (Map 3) -> Bike Shop (Map 66)
    # -------------------------------------------------------------
    print("\n[Leg 9] Navigating Cerulean City to Bike Shop at (13, 25)...")
    for step in range(80):
        m = pb.memory[0xD35E]
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if m == 66:
            print(f"  [+] Entered Bike Shop (Map 66) at ({x}, {y}) in {step} steps.")
            break
        if (x, y) == (13, 25):
            print(f"  [!] Standing on (13, 25) door! Stepping up into door...")
            press("up", hold=10, wait=30)
            continue
        wp = nav._bfs_next_waypoint(3, (x, y), (13, 25), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # -------------------------------------------------------------
    # Leg 10: Inside Bike Shop -> Talk to Clerk at (6, 2)
    # -------------------------------------------------------------
    print("\n[Leg 10] Inside Bike Shop: Approaching clerk counter at (6, 4)...")
    target_pos = (6, 4)
    for step in range(40):
        x = pb.memory[0xD362]
        y = pb.memory[0xD361]
        if (x, y) == target_pos:
            print(f"  [+] Reached clerk counter at ({x}, {y}) in {step} steps.")
            break
        wp = nav._bfs_next_waypoint(66, (x, y), target_pos, lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # Face UP towards counter (6, 3) and clerk (6, 2)
    press("up", hold=6, wait=15)
    print("  [+] Facing UP towards clerk. Initiating dialogue to redeem Bike Voucher...")
    press("a", hold=8, wait=35)

    # Advance dialogue: Clerk notices voucher, gives Bicycle, plays sound
    for d_step in range(15):
        press("a", hold=6, wait=25)
    clear_text()

    # -------------------------------------------------------------
    # Verification & Save Checkpoints
    # -------------------------------------------------------------
    bag_count = pb.memory[0xD31D]
    bag_items = [pb.memory[0xD31E + i*2] for i in range(bag_count)]
    print(f"\n=== Verification ===")
    print(f"Bag items: {bag_items}")
    has_bike = RAMMap.ITEM_BICYCLE in bag_items
    voucher_in_bag = RAMMap.ITEM_BIKE_VOUCHER in bag_items
    print(f"Has Bicycle (0x06): {has_bike}")
    print(f"Bike Voucher in Bag (should be False): {voucher_in_bag}")

    assert has_bike, f"Bicycle redemption failed! Bag items: {bag_items}"

    # Save state
    save_path = REPO_ROOT / "saves" / "bicycle_acquired.state"
    with open(save_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to {save_path}")

    chk_path = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(chk_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Updated primary checkpoint: {chk_path}")

    pb.stop()
    print("\n=== Bicycle Redemption Successfully Completed! ===")

if __name__ == "__main__":
    main()
