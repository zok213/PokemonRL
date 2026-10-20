"""
teach_cut.py — Authentic Joypad sequence to teach HM01 Cut to Wartortle
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyboy import PyBoy
from pyboy.utils import WindowEvent
from pokemon_rl.env.wram_map import RAMMap

def teach_cut_to_wartortle(state_file: str, save_output: str | None = None) -> bool:
    rom_candidates = [
        REPO_ROOT / "roms" / "pokemon_red.gb",
        REPO_ROOT / "PokemonRed.gb",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "PokemonRed.gb",
    ]
    rom_path = next(str(p) for p in rom_candidates if p.exists())
    
    pb = PyBoy(rom_path, window="null")
    with open(state_file, "rb") as f:
        pb.load_state(f)

    def send_btn(btn: str, hold: int = 8, wait: int = 25):
        mapping = {
            "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
            "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        }
        press_ev, release_ev = mapping[btn]
        pb.send_input(press_ev)
        for _ in range(hold):
            pb.tick()
        pb.send_input(release_ev)
        for _ in range(wait):
            pb.tick()

    # If Red is in cutscene (joy_ignore == 255), tick through the ship departure cutscene
    ticks = 0
    while pb.memory[RAMMap.JOY_IGNORE] == 255 and ticks < 3000:
        pb.tick()
        ticks += 1
    # Allow walking into Vermilion City at (18, 29) to fully settle
    for _ in range(120):
        pb.tick()

    print(f"[+] Overworld ready at Map {pb.memory[RAMMap.CUR_MAP]} ({pb.memory[RAMMap.X_POS]}, {pb.memory[RAMMap.Y_POS]})")

    # 1. Open Start Menu
    send_btn("start", wait=35)
    # 2. Navigate to ITEM (down twice from POKEDEX)
    send_btn("down", wait=20)
    send_btn("down", wait=20)
    # 3. Press A to open Bag
    send_btn("a", wait=35)

    # 4. In Bag: Find HM01 Cut
    num_items = pb.memory[RAMMap.NUM_BAG_ITEMS]
    bag_items = [pb.memory[RAMMap.BAG_ITEMS_START + 2 * i] for i in range(num_items)]
    if RAMMap.ITEM_HM01_CUT not in bag_items:
        print("[!] Error: HM01 Cut not found in Bag!")
        pb.stop()
        return False
    hm_slot = bag_items.index(RAMMap.ITEM_HM01_CUT)
    print(f"[+] HM01 Cut is at bag index {hm_slot}")

    # Scroll down to hm_slot
    for _ in range(hm_slot):
        send_btn("down", wait=18)

    # 5. Press A on HM01 Cut -> opens USE menu
    send_btn("a", wait=35)
    # 6. Press A on USE -> prompts "Teach CUT to a POKEMON?" with YES/NO
    send_btn("a", wait=40)
    # 7. Press A on YES -> opens Party Menu
    send_btn("a", wait=45)

    # 8. On Party Menu: Wartortle is slot 0 -> Press A
    print(f"[+] Selecting Wartortle in slot 0 (party menu_y={pb.memory[0xCC24]})")
    send_btn("a", wait=50)

    # 9. Now text plays:
    # "Booted up an HM!" -> A
    # "It contained CUT!" -> A
    # "Teach CUT to WARTORTLE?" (YES/NO prompt) -> A on YES
    # "WARTORTLE is trying to learn CUT! But WARTORTLE can't learn more than 4 moves! Delete an older move to make room for CUT?" (YES/NO prompt) -> A on YES
    # Move selection screen: 1: Bite, 2: Tail Whip, 3: Bubble, 4: Water Gun.
    # We want to replace move 1 (Tail Whip): press DOWN once, then press A!
    # "Abandon TAIL WHIP?" (YES/NO prompt) -> A on YES
    # "1, 2 and... Poof! WARTORTLE forgot TAIL WHIP! And... WARTORTLE learned CUT!"
    # Let's inspect loop to handle this sequence cleanly:
    for step in range(40):
        moves = [pb.memory[RAMMap.PARTY_HP_BASE + 7 + i] for i in range(4)]
        if 15 in moves:
            print(f"[+] SUCCESS! Wartortle learned Cut at step {step}! Moves: {moves}")
            break

        top_y = pb.memory[0xCC24]
        top_x = pb.memory[0xCC25]
        cur_item = pb.memory[0xCC26]
        max_item = pb.memory[0xCC28]
        print(f"step {step}: top_y={top_y}, top_x={top_x}, cur_item={cur_item}, max_item={max_item}", flush=True)

        # Check if we are in the 4-move selection list (wTopMenuItemY == 1, wTopMenuItemX == 0)
        # Move list items: 0 = Bite, 1 = Tail Whip, 2 = Bubble, 3 = Water Gun
        if top_y == 1 and top_x == 0 and max_item == 3:
            if cur_item == 0:
                print(f"[+] On move selection list: moving down from Bite to Tail Whip")
                send_btn("down", wait=25)
            else:
                print(f"[+] Selecting move {cur_item} (Tail Whip) to replace with Cut")
                send_btn("a", wait=40)
            continue

        # Normal text or YES/NO prompt: confirm with A
        send_btn("a", wait=35)

    # Clear any residual dialogue / close menu back to overworld with B presses
    for _ in range(6):
        send_btn("b", wait=20)

    final_moves = [pb.memory[RAMMap.PARTY_HP_BASE + 7 + i] for i in range(4)]
    print(f"[+] Final Wartortle moves: {final_moves}")

    success = 15 in final_moves
    if success and save_output:
        out_path = Path(save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pb.save_state(f)
        print(f"[+] Saved state with Cut taught to: {out_path}")

    pb.stop()
    return success

if __name__ == "__main__":
    state = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "saves" / "checkpoint_progress.state")
    out = sys.argv[2] if len(sys.argv) > 2 else None
    teach_cut_to_wartortle(state, out)
