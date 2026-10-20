"""
defeat_lt_surge.py — 100% Autonomous, Authentic Pipeline:
1. Load saves/oddish_caught.state
2. Walk Route 6 -> Vermilion City (Map 5)
3. Cut tree at (15, 18) via Oddish
4. Enter Vermilion Gym (Map 92)
5. Bypass Gym Guide via col 2, inspect Can 2 at (1, 11) -> Switch 1 triggered
6. Defeat Sailor at (2, 10)
7. Defeat Super Nerd at (2, 8)
8. Inspect Can 0 at (1, 7) -> Switch 2 triggered & electric doors open
9. Walk through doors to (5, 2) facing UP
10. Talk to Lt. Surge, battle and defeat Voltorb, Pikachu, Raichu
11. Receive Thunder Badge (Badge 3) and TM24
12. Verify wObtainedBadges == 0x07 (bit 2 set)
13. Save saves/thunder_badge.state
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
    save_path = REPO_ROOT / "saves" / "oddish_caught.state"
    with open(save_path, "rb") as f:
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

    print("[+] 1. Navigating Route 6 to Gatehouse 74 (8, 35)...")
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

    print("[+] 2. Passing through Gatehouse 74...")
    for _ in range(10):
        if pb.memory[RAMMap.CUR_MAP] == 5:
            break
        press("down", wait=18)

    print(f"[+] 3. Vermilion City arrived: Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]}). Moving to Cut tree (15, 17)...")
    for _ in range(40):
        x, y = pb.memory[RAMMap.X_POS], pb.memory[RAMMap.Y_POS]
        if (x, y) == (15, 17):
            break
        wp = nav._bfs_next_waypoint(5, (x, y), (15, 17))
        dx, dy = wp[0] - x, wp[1] - y
        press("up" if dy < 0 else ("down" if dy > 0 else ("right" if dx > 0 else "left")))

    press("down", wait=20) # Face tree at (15, 18)

    print("[+] 4. Cutting Gym tree with Oddish...")
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

    print("[+] 5. Stepping into Vermilion Gym door (12, 19)...")
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

    print("[+] 6. Navigating to Can 2 at (1, 11)...")
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

    print("[+] 7. Defeating Sailor at (2, 10)...")
    press("right", wait=20)
    press("up", wait=20)
    press("up", wait=20)

    for _ in range(15):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0: break
        press("a", wait=25)

    for bstep in range(60):
        if pb.memory[RAMMap.IS_IN_BATTLE] == 0: break
        m_y = pb.memory[0xCC24]
        if m_y == 14: press("a", wait=25)
        elif m_y == 12: press("a", wait=35)
        elif m_y in (1, 2, 7): press("b", wait=22)
        else: press("a", wait=22)

    for _ in range(15):
        if pb.memory[0xCF13] == 0 and pb.memory[RAMMap.IS_IN_BATTLE] == 0: break
        press("b", wait=20)

    print("[+] 8. Defeating Super Nerd at (2, 8)...")
    press("up", wait=20)
    press("up", wait=20)

    for _ in range(15):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0: break
        press("a", wait=25)

    for bstep in range(130):
        if pb.memory[RAMMap.IS_IN_BATTLE] == 0: break
        m_y = pb.memory[0xCC24]
        if m_y == 14: press("a", wait=25)
        elif m_y == 12: press("a", wait=35)
        elif m_y in (1, 2, 7): press("b", wait=22)
        else: press("a", wait=22)

    for _ in range(15):
        if pb.memory[0xCF13] == 0 and pb.memory[RAMMap.IS_IN_BATTLE] == 0: break
        press("b", wait=20)

    print("[+] 9. Inspecting Can 0 at (1, 7) for Switch 2...")
    press("left", wait=20)
    press("up", wait=20)
    press("a", wait=35)
    for _ in range(15):
        if pb.memory[0xCF13] == 0: break
        press("b", wait=22)

    print("[+] 10. Walking through open electric barrier to Lt. Surge...")
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
    print(f"[+] Reached Lt. Surge! Pos = {pos}, Facing = {facing}")
    assert pos == (5, 2) and facing == 4, f"Unexpected position/facing: {pos}, {facing}"

    # Save state right before Surge
    before_surge_path = REPO_ROOT / "saves" / "before_surge.state"
    with open(before_surge_path, "wb") as f:
        pb.save_state(f)
    print(f"[+] Checkpoint saved: {before_surge_path}")

    print("\n" + "="*60)
    print("  [+] ENGAGING LT. SURGE (GYM LEADER 3)...")
    print("="*60 + "\n")

    # Talk to Lt. Surge
    press("a", wait=35)
    for _ in range(30):
        if pb.memory[RAMMap.IS_IN_BATTLE] != 0:
            break
        press("a", wait=25)

    print(f"[+] Battle with Lt. Surge started! InBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}")

    # Combat loop
    battle_turn = 0
    while pb.memory[RAMMap.IS_IN_BATTLE] != 0 and battle_turn < 250:
        battle_turn += 1
        m_y = pb.memory[0xCC24]
        cur = pb.memory[0xCC26]
        enemy_hp = (pb.memory[0xCFE6] << 8) | pb.memory[0xCFE7]
        party_hp = (pb.memory[RAMMap.PARTY_HP] << 8) | pb.memory[RAMMap.PARTY_HP + 1]
        
        if m_y == 14:  # Main battle menu (Fight, Item, Pkmn, Run)
            # Select FIGHT (cur == 0)
            if cur != 0:
                press("up", wait=18)
            press("a", wait=25)
        elif m_y == 12:  # Move menu
            # Select Move 0 (Bite)
            if cur != 0:
                press("up", wait=18)
            press("a", wait=35)
        elif m_y in (1, 2, 7):  # Switch prompt ("Will RED change POKéMON?")
            press("b", wait=22)  # Decline switch
        else:  # Text, animations, dialog
            press("a", wait=22)

    print(f"[+] Battle ended after {battle_turn} turns! InBattle = {pb.memory[RAMMap.IS_IN_BATTLE]}")

    # Advance post-battle dialogue (Lt. Surge confers Thunder Badge and TM24)
    print("[+] Advancing post-battle dialogue...")
    for _ in range(40):
        if pb.memory[RAMMap.IS_IN_BATTLE] == 0 and pb.memory[0xCF13] == 0:
            # Check if badge was awarded
            badges = pb.memory[RAMMap.BADGES]
            if (badges & 0x04) != 0:
                print(f"[+] Badges in WRAM: {bin(badges)} ({hex(badges)})")
                break
        press("b", wait=22)

    # Let overworld settle
    for _ in range(60):
        pb.tick()

    badges = pb.memory[RAMMap.BADGES]
    print(f"\n[+] FINAL BADGES VALUE: {bin(badges)} ({hex(badges)})")
    assert (badges & 0x04) != 0, f"Thunder Badge (bit 2) not set! Badges = {bin(badges)}"
    assert badges == 0x07, f"Expected 3 badges (0x07), got {hex(badges)}"

    # Save milestone: saves/thunder_badge.state
    out_badge = REPO_ROOT / "saves" / "thunder_badge.state"
    with open(out_badge, "wb") as f:
        pb.save_state(f)
    print(f"[+] Milestone saved successfully to: {out_badge}")

    # Also update checkpoint_progress.state
    out_ckpt = REPO_ROOT / "saves" / "checkpoint_progress.state"
    with open(out_ckpt, "wb") as f:
        pb.save_state(f)
    print(f"[+] Progress checkpoint updated to: {out_ckpt}")

    pb.stop()
    print("\n" + "="*65)
    print("  >>> MILESTONE 3: THUNDER BADGE EARNED FROM LT. SURGE! <<<")
    print("="*65 + "\n")
    return True

if __name__ == "__main__":
    main()
