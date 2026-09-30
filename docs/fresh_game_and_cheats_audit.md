# AI Engineering & Research Audit: Memory Injection vs. Authentic Gameplay

> **Target**: Pokémon Red Autonomous Agent System  
> **Key Question**: *Why was the Pokémon Level 100? Why were the gates not blocking? Is this good or bad, and how do real AI researchers build an authentic, cheat-free, 100% reliable system?*

---

## 1. The Direct Technical Diagnosis: What Just Happened?

### A. Why Was the Pokémon Level 100?
The save state itself (`has_pokedex_nballs.state`) is a **legitimate Level 6 Squirtle with 22 HP, 0 Badges, and 0 Repel**:
```python
# Real RAM values in has_pokedex_nballs.state:
memory[0xD18C] = 6    # Squirtle Level = 6
memory[0xD16D] = 22   # Current HP = 22
memory[0xD356] = 0    # Badges = 0
memory[0xD0DB] = 0    # Repel Steps Remaining = 0
```
**Why did it show Level 100 in the runner?**  
In the baseline scripts inherited from `external/PokemonRedExperiments/jippity-6-test` (`play.py` line 109 and `verify_full_run_cerulean.py` line 22), the script was executing:
```python
p.memory[0xD18C] = 100  # Forcibly overwrite Squirtle's Level with 100 in RAM on every step!
```
Every single tick, the code was injecting `100` directly into the LR35902 CPU's Work RAM (WRAM).

### B. Why Were the Gates Open?
In a legitimate, unmodified playthrough of Pokémon Red:
1. **The Pewter Gym Gatekeeper**: A boy stands at the east exit of Pewter City heading to Route 3. If you haven't defeated Gym Leader Brock, he stops you: *"If you want to go to MT. MOON, you'd better beat BROCK first!"* and drags your character directly to the Gym door.
2. **The Route 3 Trainers**: 8 mandatory trainers with exclamation marks (`!`) stand across the road, triggering battle screens.
3. **The Mt. Moon B2F Corridor**: Super Nerd Miguel stands at `(13, 8)` blocking the tunnel to guard the Helix and Dome Fossils. He cannot be passed without defeating his Grimer and Voltorb.
4. **The Team Rocket Ambush**: Two Team Rocket Grunts block the final exit ladder leading out to Route 4.
5. **Wild Tall Grass**: Every 3 to 10 steps in tall grass or caves, the game triggers a random encounter battle.

**Why did the agent walk through unobstructed?**  
Because `play.py` was injecting **five event-flag cheats**:
```python
p.memory[0xD0DB] = 255          # Infinite Repel (Suppresses 100% of wild battles)
p.memory[0xD755] |= 0x80        # Brock Defeated Flag (Opens Route 3 gate)
p.memory[0xD7F3] = 0xFF         # Viridian Forest Trainer flags (Disables trainer sightlines)
p.memory[0xD7C3] = 0xFF         # Route 3 Trainer flags
p.memory[0xD5B3] |= 0x60        # Fossil collected / Rocket Grunts despawned
if p.memory[0xD35E] == 61:      # Mt Moon B2F: Clears sprite picture IDs to remove physical NPC collisions
    for s in [1, 2, 6, 7]:
        p.memory[0xC100 + s * 16] = 0
```

---

## 2. Research & Engineering Evaluation: Is It Good or Bad?

### A. The "Good" (Scientific Value of the Ablation)
In Reinforcement Learning and Robotics, this technique is called an **Oracle Ablation / Kinematic Feasibility Test**:
- **Separation of Concerns**: Before trying to solve a 40-hour multi-agent stochastic RPG, researchers isolate the **Geometric Navigation Problem** (Can an agent traverse 18 distinct map topologies, warp coordinates, and one-way ledges without falling into infinite loops?).
- **Establishing the Upper Bound**: It proved that with collision-free paths, the optimal trajectory from Pallet Town to Cerulean City takes **808 macro-steps (< 4 seconds)**. This provides a baseline against which pure RL exploration (439M steps) can be measured.

### B. The "Bad" (The Reality Gap / Cheating Trap)
- **It is NOT a Real Game-Playing AI**: It is a GPS walking simulator operating inside an emulator with god-mode turned on.
- **Catastrophic Failure Without Cheats**: If you turn off `apply_cheats()` right now:
  1. The agent enters tall grass on Route 1.
  2. A wild Level 2 Pidgey appears (`memory[0xD057] = 1`).
  3. The corridor walker keeps sending `UP` joypad pulses.
  4. In the battle menu, `UP` just cycles between `FIGHT` and `ITEM`.
  5. The agent never attacks, never runs, and stays frozen on the battle screen forever!
  6. Even if it survived Route 1, in Pewter City the boy drags it to Brock's gym, trapping it in an infinite escort loop!

---

## 3. How a Real AI Engineer & Researcher Builds the Legit System

To build a **100% real, authentic, reliable, cheat-free AI that plays from a completely fresh cartridge boot to Cerulean City**, we must replace the memory-injection hacks with **4 Genuine Autonomous Systems**:

```
                       ┌────────────────────────────────────────┐
                       │        Strategic State Router          │
                       │    Inspects WRAM Memory & Game Mode    │
                       └───────────────────┬────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐              ┌──────────────────┐
│  Mode 0: Battle  │             │ Mode 1: Dialogue │              │ Mode 2: Overworld│
│  (0xD057 != 0)   │             │ (0xCFCB != 0)    │              │  (In Transit)    │
│ Tactical Combat  │             │ Dismiss Text &   │              │ Closed-Loop Nav  │
│ Gen 1 Type Math  │             │ NPC Prompts      │              │ & Gym Battles    │
└──────────────────┘             └──────────────────┘              └──────────────────┘
```

### Pillar 1: Authentic Pewter Gym & Brock Defeat
Instead of spoofing `memory[0xD755] |= 0x80`, the agent must legitimately earn the **Boulder Badge**:
1. In Pewter City, route into **Pewter Gym (Map 54)**.
2. Defeat the Jr. Trainer (Level 11 Diglett, Level 11 Sandshrew).
3. Defeat Leader Brock (Level 12 Geodude, Level 14 Onix).
4. *Why this works legitimately with Squirtle:*
   - Squirtle learns `Bubble` at Level 8 (Water type).
   - Geodude and Onix are dual Rock/Ground type ($4.0\times$ Water damage weakness!).
   - A Level 8-10 Squirtle defeats Brock's entire gym in 2 to 3 turns without cheats!
5. When Brock is defeated, the game's actual Z80 assembly code awards the Boulder Badge, sets `0xD755 |= 0x80` in ROM, and the Route 3 gate opens naturally.

### Pillar 2: Turn-Based Combat Policy (Decoupled Combat Controller)
Whenever a wild Pokémon or trainer appears (`memory[0xD057] != 0`):
1. Overworld navigation pauses immediately.
2. The tactical combat engine (`DecoupledCombatController` in `src/pokemon_rl/combat/combat_controller.py`) takes control:
   - Reads opponent type from `0xCFE5`.
   - Computes expected move damage using the Gen 1 Type Chart (Water beats Rock/Ground/Fire; Normal is neutral).
   - Navigates the 2x2 FIGHT menu using `[Action.A, Action.RIGHT, Action.DOWN]`.
   - Executes `Bubble` / `Water Gun` / `Tackle`.
3. Mashes `A` to advance EXP and Level-up screens.
4. When `memory[0xD057] == 0`, resumes overworld corridor traversal!

### Pillar 3: Pokémon Center Healing & Attrition Routine
In a real game, Pokémon take damage and get Poisoned in Viridian Forest (Weedle `Poison Sting` deals 1 HP damage every 4 overworld steps!):
1. Monitor Party HP (`0xD16C - 0xD16D`) and Status Byte (`0xD16F` & 0x08 = Poison).
2. If HP < 30% or Poisoned:
   - Interrupt corridor navigation.
   - Enter Viridian or Pewter Pokémon Center.
   - Walk to `(3, 3)`, face UP, press `A` to talk to Nurse Joy.
   - Wait for healing chime, exit building, and resume corridor!

### Pillar 4: Fresh Game Cartridge Boot (Bedroom to Pokédex)
To start a truly fresh game from frame 0:
1. Boot ROM without save states.
2. Send input sequence: Title Screen -> Press START -> Select NEW GAME.
3. Advance Professor Oak introduction -> Select Name "RED" -> Rival "BLUE".
4. Red starts in bedroom (Map 37, Pos (3, 6)).
5. Walk downstairs, exit house, trigger Oak's grass cutscene.
6. Select Squirtle from table.
7. Defeat Rival Blue in Battle 1.
8. Fetch Oak's Parcel from Viridian Mart, deliver to Oak, receive Pokédex and 5 Pokéballs.
9. Hand off to main exploration policy!

---

## 4. Immediate Engineering Action Plan

1. **Keep the Geometric Corridor** as the spatial routing foundation.
2. **Remove Hardcoded Cheats** (`level=100`, `repel=255`, and pre-set badges) so the simulation matches reality.
3. **Wire `DecoupledCombatController` into the interactive loop** so when battles trigger on Route 1, Viridian Forest, and Mt. Moon, the agent fights and wins legitimately.
4. **Add the Pewter Gym Detour** so Brock is defeated by Squirtle's `Bubble` to open Route 3 legitimately.
