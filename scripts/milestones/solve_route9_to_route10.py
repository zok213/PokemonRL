"""
solve_route9_to_route10.py — 100% Autonomous, Authentic Pipeline:
1. Load saves/checkpoint_progress.state (Route 9, Map 20 at (0, 9))
2. Full Party Heal in Cerulean:
   - Step west into Cerulean City (Map 3) at (38, 17)
   - Walk west to Cerulean Pokécenter (Map 64) at (19, 17)
   - Heal with Nurse Joy -> 100% full HP (86/86 HP) & 100% PP (10/10 Dig, 25/25 Water Gun)
   - Exit Pokécenter, cut Cerulean South tree at (19, 28) with Oddish
   - Walk east to (39, 17) and enter Route 9 (Map 20) at (0, 9)
3. Clear Route 9 barrier cut tree at (5, 8) with Oddish
4. Traverse Route 9 along BFS waypoints to Route 10 (59, 8):
   - FSM battle controller dynamically handles Jr. Trainer F (sweeps 4 Pokémon with Dig),
     Jr. Trainer M, and wild encounters using canonical disassembly signals (0xFFF6, 0xCF13, 0xCC24)
   - Avoids static trainer collision traps via calibrated NPC obstacles
5. Step east into Route 10 (Map 21) at (0, 8)
6. Save saves/route10_entered.state and update saves/checkpoint_progress.state
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

    def run_fsm_battle(max_steps: int = 500):
        print("  [Battle] Engaging bulletproof FSM battle controller...", flush=True)
        for s in range(max_steps):
            if pb.memory[RAMMap.IS_IN_BATTLE] == 0:
                print(f"  [Battle] Battle ended at step {s}!", flush=True)
                break

            tb = pb.memory[0xCF13]
            h_ui = pb.memory[0xFFF6]
            m_y = pb.memory[0xCC24]
            m_x = pb.memory[0xCC25]
            cur = pb.memory[0xCC26]
            php = pb.memory[0xD015] * 256 + pb.memory[0xD016]
            pps = [pb.memory[0xD02D + i] for i in range(4)]

            # 1. MoveSelectionMenu: hUILayoutFlags bit 1 is active
            if (h_ui & 0x02) != 0:
                if pps[0] > 0: target = 1    # Dig
                elif pps[3] > 0: target = 4  # Water Gun
                elif pps[2] > 0: target = 3  # Bubble
                else: target = 1

                if cur == target:
                    press("a", wait=25)
                elif cur < target:
                    press("down", wait=16)
                else:
                    press("up", wait=16)

            # 2. Shift Prompt: Will RED change POKéMON? -> Decline
            elif (m_y in (7, 8) and m_x == 1) or tb == 0x14:
                if cur == 0:
                    press("down", wait=16)
                else:
                    press("a", wait=22)

            # 3. Main Battle Menu (FIGHT / ITEM / PKMN / RUN)
            elif tb == 0x0B or (m_y == 14 and m_x == 9):
                press("up", wait=14)
                press("left", wait=14)
                press("a", wait=22)

            # 4. Party Menu
            elif m_y == 1 and m_x == 0:
                if php == 0:
                    press("a", wait=25)
                else:
                    press("b", wait=20)

            # 5. Dialogue, text scrolling, attack animations
            else:
                press("a", wait=20)

        for _ in range(40):
            if pb.memory[0xCF13] == 0 and pb.memory[RAMMap.IS_IN_BATTLE] == 0:
                break
            press("b", wait=18)
        print(f"  [Battle] Final InBattle: {pb.memory[RAMMap.IS_IN_BATTLE]}, php={pb.memory[0xD015]*256 + pb.memory[0xD016]}", flush=True)

    nav = WorldGraphNavigator()

    print("=== Step 1: Verify Initial State ===", flush=True)
    m = pb.memory[RAMMap.CUR_MAP]
    x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
    print(f"Map: {m}, Pos: ({x}, {y})", flush=True)
    assert m == 20, f"Expected Route 9 (Map 20), got {m}"

    # Step 2: Full Party Heal in Cerulean Pokécenter
    print("\n=== Step 2: Full Party Heal in Cerulean Pokécenter ===", flush=True)
    press("left", wait=30)
    press("left", wait=30)
    assert pb.memory[RAMMap.CUR_MAP] == 3, "Failed to enter Cerulean City!"

    # Walk west to Cerulean Pokécenter at (19, 17)
    for step in range(35):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] == 64: break
        if (cx, cy) == (19, 17):
            press("up", wait=30)
            continue
        wp = nav._bfs_next_waypoint(3, (cx, cy), (19, 17), lookahead=1, can_cut=True)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[RAMMap.CUR_MAP] == 64, "Failed to enter Cerulean Pokécenter!"
    print("[+] In Cerulean Pokécenter (Map 64). Healing with Nurse Joy...", flush=True)

    # Walk to Nurse Joy at (3, 2)
    for step in range(15):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (3, 2): break
        wp = nav._bfs_next_waypoint(64, (cx, cy), (3, 2), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    press("up", wait=20)
    press("a", wait=35)
    for _ in range(12): press("a", wait=30)
    for _ in range(120): pb.tick() # Healing jingle
    for _ in range(8): press("b", wait=22)

    # Verify 100% full heal
    w_hp = pb.memory[0xD16C]*256 + pb.memory[0xD16D]
    w_max = pb.memory[0xD18D]*256 + pb.memory[0xD18E]
    dig_pp = pb.memory[0xD188]
    print(f"[+] Healed: Wartortle HP {w_hp}/{w_max}, Dig PP: {dig_pp}/10", flush=True)
    assert w_hp == w_max == 86, f"HP not full! {w_hp}/{w_max}"
    assert dig_pp == 10, f"Dig PP not full! {dig_pp}"

    # Exit Pokécenter
    for _ in range(6): press("down", wait=20)
    press("down", wait=35)
    assert pb.memory[RAMMap.CUR_MAP] == 3, "Failed to exit Pokécenter to Cerulean!"

    # Walk to Cut tree approach at (19, 27)
    for step in range(35):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (19, 27): break
        wp = nav._bfs_next_waypoint(3, (cx, cy), (19, 27), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # Cut Cerulean South tree
    press("down", wait=20)
    press("start", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=18)
    while pb.memory[0xCC26] < 1: press("down", wait=18)
    press("a", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=18)
    while pb.memory[0xCC26] < 1: press("down", wait=18)
    press("a", wait=35)
    press("a", wait=50) # Select CUT
    for _ in range(6): press("a", wait=25)
    for _ in range(60): pb.tick()
    press("down", wait=25); press("down", wait=25)

    # Walk to (39, 17) -> Route 9
    for step in range(60):
        if pb.memory[RAMMap.CUR_MAP] == 20: break
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) in ((39, 16), (39, 17)):
            press("right", hold=10, wait=30); continue
        wp = nav._bfs_next_waypoint(3, (cx, cy), (39, 17), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[RAMMap.CUR_MAP] == 20, "Failed to re-enter Route 9!"
    print(f"[+] Re-entered Route 9 with full HP and PP! Pos: ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})", flush=True)

    print("\n=== Step 3: Cut Route 9 Tree at (5, 8) ===", flush=True)
    for step in range(15):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (4, 8): break
        wp = nav._bfs_next_waypoint(20, (cx, cy), (4, 8), lookahead=1, can_cut=True)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    press("right", wait=20); press("start", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=18)
    while pb.memory[0xCC26] < 1: press("down", wait=18)
    press("a", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=18)
    while pb.memory[0xCC26] < 1: press("down", wait=18)
    press("a", wait=35)
    press("a", wait=50) # Select CUT
    for _ in range(6): press("a", wait=25)
    for _ in range(60): pb.tick()
    press("right", wait=25); press("right", wait=25)
    print(f"[+] Past Route 9 cut tree at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})", flush=True)

    print("\n=== Step 4: Traverse Route 9 to Route 10 Border (59, 8) ===", flush=True)
    last_pos = None
    stuck_cnt = 0
    for step in range(350):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            run_fsm_battle()

        cur_m = pb.memory[RAMMap.CUR_MAP]
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]

        if cur_m == 21: # Route 10
            print(f"[+] Entered Route 10! Pos: ({cx}, {cy}) in {step} steps.", flush=True)
            break

        if step % 15 == 0 or (cx, cy) != last_pos:
            print(f"  Step {step:3d}: Map {cur_m}, Pos ({cx}, {cy})", flush=True)

        if (cx, cy) == last_pos:
            stuck_cnt += 1
            if stuck_cnt > 3:
                press("b", wait=20)
                press("a", wait=20)
        else:
            stuck_cnt = 0
        last_pos = (cx, cy)

        if pb.memory[0xCF13] != 0:
            press("a", wait=20)
            continue

        if cx >= 59:
            press("right", hold=10, wait=30)
            continue

        wp = nav._bfs_next_waypoint(20, (cx, cy), (59, 8), lookahead=1, can_cut=True)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[RAMMap.CUR_MAP] == 21, f"Expected Route 10 (Map 21), got {pb.memory[RAMMap.CUR_MAP]}"
    print(f"[+] Successfully arrived on Route 10 at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})!", flush=True)

    # Save checkpoints
    r10_save = REPO_ROOT / "saves" / "route10_entered.state"
    with open(r10_save, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to {r10_save}", flush=True)

    chk_path = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(chk_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Updated primary checkpoint: {chk_path}", flush=True)

    pb.stop()
    print("\n=== Route 9 -> Route 10 Traversal Complete! ===", flush=True)


if __name__ == "__main__":
    main()
