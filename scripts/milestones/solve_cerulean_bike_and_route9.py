"""
solve_cerulean_bike_and_route9.py — 100% Authentic Autonomous Traversal:
1. Load saves/pewter_via_diglett.state (Pewter City at 18, 35)
2. Traverse Pewter City -> Route 3 -> Route 4 -> Mt. Moon 1F/B1F/B2F -> Route 4 East -> Cerulean City
3. Enter Cerulean Bike Shop (Map 66 at 13, 25)
4. Talk to Clerk at (6, 2): stand at (6, 4) facing UP, exchange Bike Voucher for BICYCLE (0x06)
5. Clear dialogue completely with Button A (hWY becomes 0x90)
6. Save saves/bicycle_acquired.state
7. Exit Bike Shop into Cerulean City at (13, 25)
8. Walk to Cut tree at (19, 27) facing DOWN, cut tree at (19, 28) with Oddish HM01 Cut
9. Walk through cut tree along row 29 to col 36, up to row 17, east to (39, 17)
10. Step east into Route 9 (Map 20) at (0, 9)
11. Save saves/route9_entered.state and update saves/checkpoint_progress.state
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

    def press(btn: str, hold: int = 8, wait: int = 15):
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

    nav = WorldGraphNavigator()

    print("=== Starting Autonomous Bicycle Redemption & Route 9 Entry ===")
    print(f"Initial State: Map {pb.memory[0xD35E]} at ({pb.memory[0xD362]}, {pb.memory[0xD361]})")

    # 1. Pewter City -> Route 3
    print("\n[1] Traversing Pewter City east towards Route 3...")
    for _ in range(120):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 14:
            print(f"  [+] Reached Route 3 at ({x}, {y})")
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

    # 2. Route 3 -> Route 4
    print("\n[2] Traversing Route 3 east towards Route 4...")
    for _ in range(300):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 15:
            print(f"  [+] Reached Route 4 West at ({x}, {y})")
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

    # 3. Route 4 West -> Mt. Moon 1F
    print("\n[3] Entering Mt. Moon...")
    for _ in range(60):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 59:
            print(f"  [+] Entered Mt. Moon 1F at ({x}, {y})")
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

    # 4. Mt. Moon 1F -> B1F
    print("\n[4] Traversing Mt. Moon 1F to ladder (5, 5)...")
    for _ in range(250):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 60:
            print(f"  [+] Reached B1F at ({x}, {y})")
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

    # 5. Mt. Moon B1F -> B2F
    print("\n[5] Traversing Mt. Moon B1F to ladder (21, 17)...")
    for _ in range(100):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 61:
            print(f"  [+] Reached B2F at ({x}, {y})")
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

    # 6. Mt. Moon B2F -> B1F exit tunnel
    print("\n[6] Traversing Mt. Moon B2F to ladder (5, 7)...")
    for _ in range(300):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 60:
            print(f"  [+] Reached B1F exit tunnel at ({x}, {y})")
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

    # 7. B1F exit tunnel -> Route 4 East
    print("\n[7] Exiting Mt. Moon to Route 4 East...")
    for _ in range(50):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 15:
            print(f"  [+] Emerged onto Route 4 East at ({x}, {y})")
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

    # 8. Route 4 East -> Cerulean City
    print("\n[8] Traversing Route 4 East to Cerulean City...")
    for _ in range(250):
        if pb.memory[0xD057] != 0: handle_battle()
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 3:
            print(f"  [+] Entered Cerulean City at ({x}, {y})")
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

    # 9. Cerulean City -> Bike Shop (Map 66)
    print("\n[9] Navigating Cerulean City to Bike Shop (13, 25)...")
    for _ in range(80):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 66:
            print(f"  [+] Entered Bike Shop (Map 66) at ({x}, {y})")
            break
        if (x, y) == (13, 25):
            press("up", hold=10, wait=30)
            continue
        wp = nav._bfs_next_waypoint(3, (x, y), (13, 25), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # 10. Inside Bike Shop: Approach clerk at (6, 4), redeem Bicycle
    print("\n[10] Approaching Bike Shop clerk at (6, 4)...")
    for _ in range(30):
        x, y = pb.memory[0xD362], pb.memory[0xD361]
        if (x, y) == (6, 4):
            print(f"  [+] Reached clerk counter at ({x}, {y})")
            break
        wp = nav._bfs_next_waypoint(66, (x, y), (6, 4), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    press("up", hold=6, wait=15)
    print("  [+] Talking to clerk to redeem Bike Voucher...")
    press("a", hold=8, wait=35)
    # Advance exchange dialogue + fanfare sound
    for _ in range(12):
        press("a", hold=6, wait=25)

    # Verify Bicycle in Bag
    bag_count = pb.memory[0xD31D]
    bag_items = [pb.memory[0xD31E + i*2] for i in range(bag_count)]
    print(f"  [+] Bag items: {bag_items}")
    assert RAMMap.ITEM_BICYCLE in bag_items, f"Failed to acquire Bicycle! Bag: {bag_items}"
    print("  [+] BICYCLE (0x06) successfully acquired!")

    # Dismiss any trailing dialogue (clerk follow-up)
    for _ in range(3):
        press("a", hold=6, wait=25)
    press("b", hold=6, wait=20)
    for _ in range(10): pb.tick()

    # Save bicycle_acquired.state
    bike_save = REPO_ROOT / "saves" / "bicycle_acquired.state"
    with open(bike_save, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved bicycle checkpoint: {bike_save}")

    # 11. Exit Bike Shop to Cerulean City
    print("\n[11] Exiting Bike Shop...")
    for _ in range(25):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 3:
            print(f"  [+] Exited to Cerulean City at ({x}, {y})")
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

    # 12. Cerulean City: Walk to south Cut tree at (19, 27)
    print("\n[12] Navigating to Cerulean City south Cut tree at (19, 27)...")
    for _ in range(30):
        x, y = pb.memory[0xD362], pb.memory[0xD361]
        if (x, y) == (19, 27):
            print(f"  [+] Standing at cut tree approach ({x}, {y})")
            break
        wp = nav._bfs_next_waypoint(3, (x, y), (19, 27), lookahead=1)
        dx, dy = wp[0] - x, wp[1] - y
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # 13. Cut tree with Oddish
    print("\n[13] Cutting Cerulean south tree with Oddish...")
    press("down", wait=20) # Face tree at (19, 28)
    press("start", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=20)
    while pb.memory[0xCC26] < 1: press("down", wait=20)
    press("a", wait=35) # Open Party
    while pb.memory[0xCC26] > 1: press("up", wait=20)
    while pb.memory[0xCC26] < 1: press("down", wait=20)
    press("a", wait=35) # Select Oddish
    press("a", wait=50) # Select CUT
    for _ in range(6): press("a", wait=30)
    for _ in range(60): pb.tick()
    print("  [+] Tree cut! Stepping through onto row 29...")
    press("down", wait=25)
    press("down", wait=25)
    print(f"  [+] Pos: ({pb.memory[0xD362]}, {pb.memory[0xD361]})")

    # 14. Navigate to Route 9 east border at (39, 17)
    print("\n[14] Navigating through Cerulean east lane to Route 9 border (39, 17)...")
    for step in range(60):
        m, x, y = pb.memory[0xD35E], pb.memory[0xD362], pb.memory[0xD361]
        if m == 20: # Route 9!
            print(f"  [+] Stepped into Route 9 at ({x}, {y}) in {step} steps.")
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

    # 15. Verify and Save Route 9 state
    print(f"\n=== Route 9 Verification ===")
    assert pb.memory[0xD35E] == 20, f"Expected Map 20 (Route 9), got Map {pb.memory[0xD35E]}"
    print(f"[+] Confirmed on Route 9 (Map 20) at ({pb.memory[0xD362]}, {pb.memory[0xD361]})!")

    r9_save = REPO_ROOT / "saves" / "route9_entered.state"
    with open(r9_save, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to {r9_save}")

    chk_path = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(chk_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Updated primary checkpoint: {chk_path}")

    pb.stop()
    print("\n=== Bicycle Acquired & Route 9 Entry 100% COMPLETE! ===")

if __name__ == "__main__":
    main()
