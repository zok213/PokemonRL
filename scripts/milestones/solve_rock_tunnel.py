"""
solve_rock_tunnel.py — 100% Autonomous, Authentic Pipeline:
1. Load saves/checkpoint_progress.state (Route 10 North, Map 21 at (0, 8))
2. Full Party Heal in Rock Tunnel Pokécenter (Map 81):
   - Walk from (0, 8) to (11, 19)
   - Step into Pokécenter (Map 81)
   - Heal with Nurse Joy at (3, 1) -> 100% full HP (89/89 HP) & 100% PP (10/10 Dig)
   - Exit back to Route 10 at (11, 20)
3. Enter Rock Tunnel 1F:
   - Walk to (9, 21) facing UP towards tree at (9, 20)
   - Cut tree 1 at (9, 20) with Oddish and step UP to (9, 20)
   - Cut tree 2 at (9, 18) with Oddish and step UP to (9, 18)
   - Step left to (8, 18) and step UP through cave entrance (8, 17) into Rock Tunnel 1F (Map 82) at (15, 3)
4. Traverse Rock Tunnel via 5 Canonical Ladder Partitions:
   - Partition 1 (1F, Map 82): (15, 3) -> Ladder 1 at (37, 3)
   - Partition 2 (B1F, Map 232): (33, 25) -> Ladder 2 at (27, 3)
   - Partition 3 (1F, Map 82): (5, 3) -> Ladder 3 at (17, 11)
   - Partition 4 (B1F, Map 232): (23, 11) -> Ladder 4 at (3, 3)
   - Partition 5 (1F, Map 82): (37, 17) -> South Exit at (15, 33)
   - Bulletproof FSM battle controller handles all wild encounters and trainer battles
5. Route 10 South Traversal:
   - Step south through cave exit onto Route 10 South (Map 21)
   - Walk south into Lavender Town (Map 4)
6. Save saves/lavender_town.state and update saves/checkpoint_progress.state
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
            in_b = pb.memory[RAMMap.IS_IN_BATTLE]
            if in_b == 0:
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
                enemy_sp = pb.memory[0xCFE5]
                # Rock/Ground types: Geodude (0xA9), Graveler (0x27), Onix (0x22), Rhyhorn (0x12), Cubone (0x90)
                # Water Gun / Bubblebeam deal 4x damage!
                if enemy_sp in (0xA9, 0x27, 0x22, 0x12, 0x90):
                    if pps[1] > 0: target = 2     # Bubblebeam (65 power, 4x damage!)
                    elif pps[3] > 0: target = 4   # Water Gun (4x damage)
                    elif pps[2] > 0: target = 3   # Bubble (4x damage)
                    elif pps[0] > 0: target = 1   # Dig
                    else: target = 1
                # Poison/Grass types: Oddish (0xB9), Gloom (0xBB), Bulbasaur (0x99), Ivysaur (0x09), Bellsprout (0xBC), Weepinbell (0xBD)
                elif enemy_sp in (0xB9, 0xBB, 0x99, 0x09, 0xBC, 0xBD):
                    if pps[0] > 0: target = 1     # Dig (100 power physical, 2x vs Poison!)
                    elif pps[1] > 0: target = 2   # Bubblebeam (65 power)
                    elif pps[3] > 0: target = 4   # Water Gun
                    else: target = 1
                else:
                    # Neutral / Flying / Normal types: Bubblebeam (Move 2) or Dig (Move 1)
                    if pps[1] > 0: target = 2     # Bubblebeam
                    elif pps[0] > 0: target = 1   # Dig
                    elif pps[3] > 0: target = 4   # Water Gun
                    elif pps[2] > 0: target = 3   # Bubble
                    else: target = 1

                if cur == target:
                    press("a", wait=25)
                elif cur < target:
                    press("down", wait=16)
                else:
                    press("up", wait=16)

            # 2. Shift Prompt: Will RED change POKéMON? -> Instantly decline with B
            elif (m_y in (1, 2, 7, 8) and m_x in (1, 14)) or tb == 0x14:
                press("b", wait=22)

            # 3. Main Battle Menu (FIGHT / ITEM / PKMN / RUN)
            elif tb == 0x0B or (m_y == 14 and m_x == 9):
                if in_b == 1:
                    # Wild encounter: 100% clean escape! Saves all HP & PP
                    press("down", wait=14)
                    press("right", wait=14)
                    press("a", wait=22)
                else:
                    # Trainer battle: FIGHT!
                    press("up", wait=14)
                    press("left", wait=14)
                    press("a", wait=22)

            # 4. Party Menu (if active mon faints or opened accidentally)
            elif m_y == 1 and m_x == 0:
                if php == 0:
                    if cur == 0:
                        press("down", wait=18)
                    else:
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
            press("a", wait=18)
        print(f"  [Battle] Final InBattle: {pb.memory[RAMMap.IS_IN_BATTLE]}, php={pb.memory[0xD015]*256 + pb.memory[0xD016]}", flush=True)

    def teach_bubblebeam():
        moves = [pb.memory[0xD173 + i] for i in range(4)]
        if 61 in moves:
            print("[+] Wartortle already knows Bubblebeam!", flush=True)
            return
        print("[+] Teaching TM11 Bubblebeam to Wartortle...", flush=True)
        press("start", wait=30)
        while pb.memory[0xCC26] < 2: press("down", wait=16)
        while pb.memory[0xCC26] > 2: press("up", wait=16)
        press("a", wait=35) # Select ITEM
        for _ in range(4): press("down", wait=16) # Item 4 is TM11
        press("a", wait=35) # Click on TM11
        press("a", wait=40) # Click USE
        press("a", wait=50) # Select Wartortle
        for _ in range(6): press("a", wait=25) # Dialog: Booted up TM! Teach BUBBLEBEAM?
        press("a", wait=40) # YES
        for _ in range(6): press("a", wait=25) # Dialog: Delete older move?
        press("a", wait=40) # YES
        # Move menu: 0: Dig, 1: Tail Whip, 2: Bubble, 3: Water Gun
        press("down", wait=20) # Select Tail Whip
        press("a", wait=40) # Replace Tail Whip
        for _ in range(12): press("a", wait=25) # Dialog: learned Bubblebeam!
        for _ in range(6): press("b", wait=20) # Exit back to overworld
        new_moves = [pb.memory[0xD173 + i] for i in range(4)]
        print(f"[+] Wartortle moves after teaching: {new_moves}", flush=True)

    def cure_poison_if_needed():
        status = pb.memory[0xD16F]
        if (status & 0x08) != 0:
            print("[+] Wartortle is poisoned! Administering Antidote...", flush=True)
            press("start", wait=30)
            while pb.memory[0xCC26] < 2: press("down", wait=16)
            while pb.memory[0xCC26] > 2: press("up", wait=16)
            press("a", wait=35) # Select ITEM
            press("down", wait=16) # Item 1 is Antidote
            press("a", wait=35) # Click Antidote
            press("a", wait=40) # Click USE
            press("a", wait=50) # Select Wartortle
            for _ in range(6): press("a", wait=25) # Dialog: Cured of poison!
            for _ in range(6): press("b", wait=20) # Exit back to overworld
            new_status = pb.memory[0xD16F]
            print(f"[+] Poison cured! Status is now: 0x{new_status:02X}", flush=True)

    def cut_tree_ahead():
        press("start", wait=35)
        while pb.memory[0xCC26] > 1: press("up", wait=18)
        while pb.memory[0xCC26] < 1: press("down", wait=18)
        press("a", wait=35) # PKMN
        while pb.memory[0xCC26] > 1: press("up", wait=18)
        while pb.memory[0xCC26] < 1: press("down", wait=18)
        press("a", wait=35) # Oddish
        press("a", wait=50) # CUT
        for _ in range(6): press("a", wait=25)
        for _ in range(60): pb.tick() # Cut animation

    nav = WorldGraphNavigator()

    print("=== Step 1: Verify Initial State on Route 10 ===", flush=True)
    m = pb.memory[RAMMap.CUR_MAP]
    x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
    print(f"Map: {m}, Pos: ({x}, {y})", flush=True)
    assert m == 21, f"Expected Route 10 (Map 21), got {m}"

    print("\n=== Step 2: Navigate to Route 10 Pokécenter ===", flush=True)
    for step in range(50):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            run_fsm_battle()
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] == 81:
            print(f"[+] Entered Rock Tunnel Pokécenter (Map 81) at ({cx}, {cy})!", flush=True)
            break
        if (cx, cy) == (11, 19):
            press("up", wait=30)
            continue
        wp = nav._bfs_next_waypoint(21, (cx, cy), (11, 19), lookahead=1, can_cut=False)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[RAMMap.CUR_MAP] == 81, "Failed to enter Rock Tunnel Pokécenter!"

    # Teach TM11 Bubblebeam safely inside Pokécenter before healing
    teach_bubblebeam()

    # Heal with Nurse Joy at (3, 1)
    for step in range(15):
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
    for _ in range(120): pb.tick() # Healing jingle
    for _ in range(8): press("b", wait=22)

    w_hp = pb.memory[0xD16C]*256 + pb.memory[0xD16D]
    w_max = pb.memory[0xD18D]*256 + pb.memory[0xD18E]
    dig_pp = pb.memory[0xD188]
    print(f"[+] Fully Healed: Wartortle HP {w_hp}/{w_max}, Dig PP: {dig_pp}/10", flush=True)
    assert w_hp == w_max == 89, f"HP not full! {w_hp}/{w_max}"
    assert dig_pp == 10, f"Dig PP not full! {dig_pp}"

    # Exit Pokécenter back to Route 10
    for _ in range(6): press("down", wait=20)
    press("down", wait=35)
    for _ in range(40): pb.tick()
    assert pb.memory[RAMMap.CUR_MAP] == 21, f"Failed to exit Pokécenter to Route 10! Map={pb.memory[RAMMap.CUR_MAP]}"
    print(f"[+] Back on Route 10 at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})!", flush=True)

    print("\n=== Step 3: Cut Trees & Enter Rock Tunnel 1F ===", flush=True)
    for _ in range(10):
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (cx, cy) == (9, 21): break
        wp = nav._bfs_next_waypoint(21, (cx, cy), (9, 21), lookahead=1, can_cut=False)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    # Cut tree 1 at (9, 20)
    print("[+] Cutting tree 1 at (9, 20)...", flush=True)
    press("up", wait=20)
    cut_tree_ahead()
    press("up", wait=25)
    print(f"[+] Stepped to ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})", flush=True)

    # Cut tree 2 at (9, 18)
    print("[+] Cutting tree 2 at (9, 18)...", flush=True)
    press("up", wait=20)
    cut_tree_ahead()
    press("up", wait=25)
    print(f"[+] Stepped to ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})", flush=True)

    # Step into Rock Tunnel door at (8, 17)
    press("left", wait=25)
    press("up", wait=35)
    for _ in range(60): pb.tick() # Warp transition

    assert pb.memory[RAMMap.CUR_MAP] == 82, f"Failed to enter Rock Tunnel 1F! Map={pb.memory[RAMMap.CUR_MAP]}"
    print(f"[+] ENTERED ROCK TUNNEL 1F at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})!", flush=True)

    print("\n=== Step 4: Traverse Rock Tunnel 5 Partitions ===", flush=True)
    partitions = [
        ("Partition 1 (1F -> B1F Ladder 1)", 82, (37, 3), 232),
        ("Partition 2 (B1F -> 1F Ladder 2)", 232, (27, 3), 82),
        ("Partition 3 (1F -> B1F Ladder 3)", 82, (17, 11), 232),
        ("Partition 4 (B1F -> 1F Ladder 4)", 232, (3, 3), 82),
        ("Partition 5 (1F -> South Exit)", 82, (15, 33), 21),
    ]

    for part_name, current_map_id, target_goal, next_map_id in partitions:
        print(f"\n--- Starting {part_name} to {target_goal} ---", flush=True)
        last_pos = None
        stuck_cnt = 0
        for step in range(200):
            if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
                run_fsm_battle()
                cure_poison_if_needed()

            cur_m = pb.memory[RAMMap.CUR_MAP]
            cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]

            if cur_m == next_map_id:
                print(f"[+] Completed {part_name}! Arrived in Map {cur_m} at ({cx}, {cy}) in {step} steps.", flush=True)
                break

            if (cx, cy) == target_goal:
                # Step onto ladder / door warp
                press("up" if target_goal[1] <= 11 else "down", wait=30)
                for _ in range(30): pb.tick()
                continue

            if step % 15 == 0 or (cx, cy) != last_pos:
                print(f"  [{part_name}] Step {step:3d}: Map {cur_m}, Pos ({cx}, {cy})", flush=True)

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

            cure_poison_if_needed()

            wp = nav._bfs_next_waypoint(current_map_id, (cx, cy), target_goal, lookahead=1)
            dx, dy = wp[0] - cx, wp[1] - cy
            if dx > 0: press("right")
            elif dx < 0: press("left")
            elif dy > 0: press("down")
            elif dy < 0: press("up")

        assert pb.memory[RAMMap.CUR_MAP] == next_map_id, f"Failed {part_name}! Map={pb.memory[RAMMap.CUR_MAP]}"

    print(f"\n[+] Successfully Exited Rock Tunnel onto Route 10 South at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})!", flush=True)

    print("\n=== Step 5: Route 10 South to Lavender Town ===", flush=True)
    # Walk south to (8, 54) -> Lavender Town (Map 4)
    last_pos = None
    stuck_cnt = 0
    for step in range(80):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            run_fsm_battle()

        cur_m = pb.memory[RAMMap.CUR_MAP]
        cx, cy = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]

        if cur_m == 4: # Lavender Town!
            print(f"[+] ENTERED LAVENDER TOWN (Map 4) at ({cx}, {cy}) in {step} steps!", flush=True)
            break

        if step % 10 == 0 or (cx, cy) != last_pos:
            print(f"  Step {step:2d}: Map {cur_m}, Pos ({cx}, {cy})", flush=True)

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

        if cy >= 71:
            press("down", hold=10, wait=30)
            continue

        wp = nav._bfs_next_waypoint(21, (cx, cy), (8, 71), lookahead=1)
        dx, dy = wp[0] - cx, wp[1] - cy
        if dx > 0: press("right")
        elif dx < 0: press("left")
        elif dy > 0: press("down")
        elif dy < 0: press("up")

    assert pb.memory[RAMMap.CUR_MAP] == 4, f"Expected Lavender Town (Map 4), got {pb.memory[RAMMap.CUR_MAP]}"
    print(f"[+] Confirmed in Lavender Town at ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})!", flush=True)

    # Save checkpoints
    lav_save = REPO_ROOT / "saves" / "lavender_town.state"
    with open(lav_save, "wb") as f:
        pb.save_state(f)
    print(f"[+] Saved checkpoint to {lav_save}", flush=True)

    chk_path = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(chk_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Updated primary checkpoint: {chk_path}", flush=True)

    pb.stop()
    print("\n=== Rock Tunnel Traversal to Lavender Town Complete! ===", flush=True)


if __name__ == "__main__":
    main()
