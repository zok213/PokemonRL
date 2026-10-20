"""
solve_surge_complete.py — 100% Autonomous, Authentic Pipeline:
1. Load saves/oddish_caught.state (Route 6 at 13, 30)
2. Teach TM28 Dig to Wartortle (replaces slot 0 Withdraw/Bite)
3. Walk south through Gatehouse 74 into Vermilion City (Map 5)
4. Enter Vermilion Pokécenter (Map 89) at (11, 3)
5. Heal party with Nurse Joy -> 100% Full HP (83/83 HP, all PP full)
6. Exit Pokécenter back to Vermilion City at (11, 4)
7. Walk to Gym Cut tree at (15, 17)
8. Cut tree using Oddish
9. Enter Vermilion Gym (Map 92) via (12, 19)
10. Can 2 at (1, 11) -> Switch 1
11. Defeat Sailor at (2, 10) with Dig
12. Defeat Super Nerd at (2, 8) with Dig
13. Can 0 at (1, 7) -> Switch 2 (electric doors open)
14. Walk through doors to (5, 2) facing UP
15. Battle Lt. Surge:
    - Dig sweeps Voltorb (one-shot)
    - Dig sweeps Pikachu (one-shot)
    - Wartortle tanks Thunderbolt with full HP, Dig sweeps Raichu (one-shot)
16. Lt. Surge defeated!
17. Claim Thunder Badge (Badge 3, wObtainedBadges == 0x07) & TM24 Thunderbolt
18. Save saves/thunder_badge.state and update saves/checkpoint_progress.state
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
    with open(REPO_ROOT / "saves" / "oddish_caught.state", "rb") as f:
        pb.load_state(f)

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

    # Step 1: Teach TM28 Dig to Wartortle
    print("[+] 1. Teaching TM28 Dig to Wartortle (slot 0)...")
    press("start", wait=35)
    while pb.memory[0xCC26] > 2: press("up", wait=20)
    while pb.memory[0xCC26] < 2: press("down", wait=20)
    press("a", wait=35) # Open Bag
    
    target_idx = 7 # TM28 Dig
    for _ in range(20):
        cur_idx = pb.memory[0xCC36] + pb.memory[0xCC26]
        if cur_idx == target_idx: break
        if cur_idx > target_idx: press("up", wait=18)
        else: press("down", wait=18)

    press("a", wait=35) # Open USE/TOSS
    press("a", wait=40) # Select USE
    press("a", wait=35) # Booted up TM!
    press("a", wait=35) # It contained DIG!
    press("a", wait=45) # Teach to a Pokemon? -> YES
    
    # In Party menu: move up to Wartortle (slot 0)
    while pb.memory[0xCC26] > 0:
        press("up", wait=20)
    press("a", wait=45) # Select Wartortle
    
    for _ in range(15): press("a", wait=35) # Advance forget old move & learn Dig
    for _ in range(8): press("b", wait=20)  # Exit back to overworld

    moves = [pb.memory[0xD173 + i] for i in range(4)]
    print(f"[+] Wartortle moves: {moves}")
    assert moves[0] == 91, f"Failed to learn Dig in slot 0! Moves: {moves}"

    # Step 2: Navigate Route 6 to Gatehouse 74
    print("[+] 2. Navigating Route 6 to Gatehouse 74 (8, 35)...")
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

    # Step 3: Pass through Gatehouse 74
    print("[+] 3. Passing through Gatehouse 74...")
    for _ in range(10):
        if pb.memory[RAMMap.CUR_MAP] == 5:
            break
        press("down", wait=18)

    print(f"[+] Arrived in Vermilion City: Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    # Step 4: Walk to Vermilion Pokécenter at (11, 3)
    print("[+] 4. Walking to Vermilion Pokécenter door (11, 3)...")
    for _ in range(40):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if pb.memory[RAMMap.CUR_MAP] == 89:
            break
        if (x, y) == (11, 3):
            press("up", wait=25)
            break
        wp = nav._bfs_next_waypoint(5, (x, y), (11, 3))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    for _ in range(40): pb.tick()
    print(f"[+] Inside Pokécenter: Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")
    assert pb.memory[RAMMap.CUR_MAP] == 89, "Failed to enter Vermilion Pokécenter!"

    # Step 5: Walk to Nurse Joy counter at (3, 3) and heal
    print("[+] 5. Healing party with Nurse Joy...", flush=True)
    for _ in range(10):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (3, 3): break
        wp = nav._bfs_next_waypoint(89, (x, y), (3, 3))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    press("up", wait=20) # Face Nurse Joy counter
    press("a", wait=35)  # Talk
    for _ in range(25):  # Full Joy dialogue + healing machine + farewell
        press("a", wait=30)
    for _ in range(5):
        press("b", wait=20)

    hp = (pb.memory[RAMMap.PARTY_HP_BASE] << 8) | pb.memory[RAMMap.PARTY_HP_BASE + 1]
    max_hp = (pb.memory[RAMMap.PARTY_MAX_HP_BASE] << 8) | pb.memory[RAMMap.PARTY_MAX_HP_BASE + 1]
    print(f"[+] Healed! Wartortle HP: {hp}/{max_hp}", flush=True)
    assert hp == max_hp, f"Wartortle not fully healed! HP: {hp}/{max_hp}"

    # Step 6: Exit Pokécenter to Vermilion City
    print("[+] 6. Exiting Pokécenter...", flush=True)
    for _ in range(10):
        if pb.memory[RAMMap.CUR_MAP] == 5: break
        press("down", wait=22)

    for _ in range(40): pb.tick()
    print(f"[+] Back in Vermilion City: Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})", flush=True)
    assert pb.memory[RAMMap.CUR_MAP] == 5, f"Failed to exit to Vermilion City! Cur Map = {pb.memory[RAMMap.CUR_MAP]}"

    # Step 7: Walk to Cut tree at (15, 17)
    print("[+] 7. Walking to Vermilion Gym cut tree at (15, 17)...")
    for _ in range(40):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (15, 17): break
        wp = nav._bfs_next_waypoint(5, (x, y), (15, 17))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    press("down", wait=20) # Face tree at (15, 18)

    # Step 8: Cut Gym tree with Oddish
    print("[+] 8. Cutting Gym tree with Oddish...")
    press("start", wait=35)
    while pb.memory[0xCC26] > 1: press("up", wait=20)
    while pb.memory[0xCC26] < 1: press("down", wait=20)
    press("a", wait=35) # Open Party
    while pb.memory[0xCC26] > 1: press("up", wait=20)
    while pb.memory[0xCC26] < 1: press("down", wait=20)
    press("a", wait=35) # Open Oddish sub-menu (CUT is slot 0)
    press("a", wait=50) # Select CUT

    for _ in range(6): press("a", wait=30)
    for _ in range(60): pb.tick()

    # Step 9: Enter Vermilion Gym door (12, 19)
    print("[+] 9. Stepping into Vermilion Gym door (12, 19)...")
    press("down", wait=22) # (15, 18)
    press("down", wait=22) # (15, 19)
    press("down", wait=22) # (15, 20)
    press("left", wait=20) # (14, 20)
    press("left", wait=20) # (13, 20)
    press("left", wait=20) # (12, 20)
    press("up", wait=25)   # Step up into door at (12, 19)

    for _ in range(40): pb.tick()
    assert pb.memory[RAMMap.CUR_MAP] == 92, f"Failed to enter Vermilion Gym! Cur Map = {pb.memory[RAMMap.CUR_MAP]}"
    print(f"[+] In Vermilion Gym (Map 92)! Player pos = ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    # Step 10: Solve Can 2 at (1, 11) (Switch 1)
    print("[+] 10. Navigating to Can 2 at (1, 11)...")
    nav._map_grids[92][14][4] = '#' # Gym guide block
    for s in range(20):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (1, 12): break
        wp = nav._bfs_next_waypoint(92, (x, y), (1, 12))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    press("up", wait=20) # Face Can 2 at (1, 11)
    press("a", wait=35)  # Inspect Can 2
    for _ in range(10):
        if pb.memory[0xCF13] == 0: break
        press("b", wait=22)

    def run_battle(max_steps: int = 250):
        for _ in range(max_steps):
            if pb.memory[RAMMap.IS_IN_BATTLE] == 0:
                break
            m_y = pb.memory[0xCC24]
            m_x = pb.memory[0xCC25]
            cur = pb.memory[0xCC26]
            tb = pb.memory[0xCF13]
            
            if m_y == 14 and m_x == 9:
                if cur != 0:
                    press("up", wait=18)
                press("a", wait=25)
            elif m_y == 12 and m_x == 5:
                if cur > 1:
                    press("up", wait=18)
                press("a", wait=35)
            elif m_y in (7, 8) and m_x == 1:
                # 'Will RED change POKéMON?' -> select NO
                if cur == 0:
                    press("down", wait=18)
                else:
                    press("a", wait=25)
            elif m_y == 8 and m_x == 15:
                # Move learn prompts: Decline Delete move, confirm Abandon learning
                l14 = ''.join([chr(ord('A')+t-0x80) if 0x80<=t<=0x99 else (chr(ord('a')+t-0xA0) if 0xA0<=t<=0xB9 else ' ') for t in [pb.memory[0xC3A0 + 14*20 + c] for c in range(1, 19)]])
                if 'Delete' in l14 or 'make' in l14 or 'room' in l14:
                    press("down", wait=18)
                    press("a", wait=25)
                elif 'Abandon' in l14:
                    press("a", wait=25)
                else:
                    press("a", wait=25)
            elif m_y == 1 and m_x == 0:
                press("b", wait=22)
            else:
                press("b", wait=22)
        
        for _ in range(30):
            if pb.memory[0xCF13] == 0 and pb.memory[RAMMap.IS_IN_BATTLE] == 0:
                break
            press("b", wait=20)

    # Step 11: Defeat Sailor at (2, 10) with Dig
    print("[+] 11. Defeating Sailor at (2, 10)...", flush=True)
    press("right", wait=20)
    press("up", wait=20)
    press("up", wait=20)

    for _ in range(15):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0: break
        press("a", wait=25)

    run_battle(max_steps=120)
    print(f"[+] Sailor defeated! Pos = ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}), InBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}", flush=True)

    # Step 12: Defeat Super Nerd at (2, 8) with Dig
    print("[+] 12. Defeating Super Nerd at (2, 8)...", flush=True)
    press("up", wait=20)
    press("up", wait=20)

    for _ in range(15):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0: break
        press("a", wait=25)

    run_battle(max_steps=160)
    print(f"[+] Super Nerd defeated! Pos = ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}), InBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}", flush=True)

    # Step 13: Inspect Can 0 at (1, 7) for Switch 2
    print("[+] 13. Inspecting Can 0 at (1, 7) for Switch 2...", flush=True)
    press("left", wait=20)
    press("up", wait=20)
    press("a", wait=35)
    for _ in range(15):
        if pb.memory[0xCF13] == 0: break
        press("b", wait=22)

    # Step 14: Walk through open electric barrier to Lt. Surge
    print("[+] 14. Walking through open electric barrier to Lt. Surge...", flush=True)
    path_to_surge = [(2, 8), (3, 8), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4), (4, 3), (4, 2), (5, 2)]
    for target in path_to_surge:
        for _ in range(10):
            x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
            if (x, y) == target: break
            dx, dy = target[0] - x, target[1] - y
            press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    press("up", wait=20) # Face UP to Lt. Surge at (5, 1)
    pos = (pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS])
    facing = pb.memory[RAMMap.PLAYER_FACING]
    hp = (pb.memory[RAMMap.PARTY_HP_BASE] << 8) | pb.memory[RAMMap.PARTY_HP_BASE + 1]
    max_hp = (pb.memory[RAMMap.PARTY_MAX_HP_BASE] << 8) | pb.memory[RAMMap.PARTY_MAX_HP_BASE + 1]
    print(f"[+] Reached Lt. Surge! Pos = {pos}, Facing = {facing}, Wartortle HP = {hp}/{max_hp}", flush=True)
    assert pos == (5, 2) and facing == 4, f"Unexpected pos/facing: {pos}, {facing}"

    # Step 15: Engage and battle Lt. Surge
    print("\n" + "="*60, flush=True)
    print("  [+] ENGAGING LT. SURGE (GYM LEADER 3)...", flush=True)
    print("="*60 + "\n", flush=True)

    press("a", wait=35)
    for _ in range(30):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0: break
        press("a", wait=25)

    print(f"[+] Battle with Lt. Surge started! IsInBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}", flush=True)
    run_battle(max_steps=150)
    print(f"[+] Battle ended! IsInBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}", flush=True)

    # Step 16: Advance post-battle dialogue until Thunder Badge awarded
    print("[+] Advancing post-battle dialogue...")
    for _ in range(80):
        badges = pb.memory[RAMMap.OBTAINED_BADGES]
        if (badges & 0x04) != 0 and pb.memory[0xCF13] == 0:
            break
        press("b", wait=22)

    for _ in range(60): pb.tick()

    badges = pb.memory[RAMMap.OBTAINED_BADGES]
    party_hp = (pb.memory[RAMMap.PARTY_HP_BASE] << 8) | pb.memory[RAMMap.PARTY_HP_BASE + 1]
    print(f"\n[+] FINAL BADGES VALUE: {bin(badges)} ({hex(badges)})")
    print(f"[+] WARTORTLE FINAL HP: {party_hp}")
    assert (badges & 0x04) != 0, f"Thunder Badge (bit 2) not set! Badges = {bin(badges)}"
    assert badges == 0x07, f"Expected 3 badges (0x07), got {hex(badges)}"

    # Save milestone saves/thunder_badge.state
    out_badge = REPO_ROOT / "saves" / "thunder_badge.state"
    with open(out_badge, "wb") as f:
        pb.save_state(f)
    print(f"[+] Milestone saved successfully to: {out_badge}")

    out_ckpt = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(out_ckpt, "wb") as f:
        pb.save_state(f)
    print(f"[+] Progress checkpoint updated to: {out_ckpt}")

    pb.stop()
    print("\n" + "="*65)
    print("  >>> MILESTONE 3 COMPLETE: THUNDER BADGE EARNED FROM LT. SURGE! <<<")
    print("="*65 + "\n")
    return True

if __name__ == "__main__":
    main()
