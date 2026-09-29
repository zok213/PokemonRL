# Game Boy LR35902 WRAM Telemetry Reference Guide

Comprehensive register memory map for Pokémon Red, verified against the canonical `pret/pokered` disassembly.

---

## 1. Primary Overworld & Progression Registers

| Address | Symbol | Type | Description |
|:---:|:---:|:---:|:---|
| `0xD35E` | `wCurMap` | `uint8` | Current Map ID index (0–247). Pallet Town = 0, Viridian = 1, Pewter = 2. |
| `0xD362` | `wXCoord` | `uint8` | Player X tile coordinate on current map (0-indexed). |
| `0xD361` | `wYCoord` | `uint8` | Player Y tile coordinate on current map (0-indexed). |
| `0xD356` | `wObtainedBadges` | `uint8` | 8-bit bitfield of acquired Gym Badges (bit 0 = Boulder .. bit 7 = Earth). |
| `0xD057` | `wIsInBattle` | `uint8` | `0` = Overworld, `1` = Wild Battle, `2` = Trainer Battle, `-1` = Defeated. |
| `0xCF13` | `wTextBoxID` | `uint8` | Non-zero when dialogue or prompt window is active on-screen. |
| `0xCD6B` | `wJoyIgnore` | `uint8` | **Hardware Joypad Mask**: Set by Game Boy CPU during cutscenes/evolution. |
| `0xD70D` | `wSafariSteps` (Lo) | `uint8` | Low byte of Safari Zone countdown counter (502 steps initially). |
| `0xD70E` | `wSafariSteps` (Hi) | `uint8` | High byte of Safari Zone countdown counter. |
| `0xCC26` | `wCurrentMenuItem` | `uint8` | Currently selected menu or battle cursor slot (0-indexed). |

---

## 2. Event Flags Bitfield (`0xD747` – `0xD886`)

The game tracks global narrative and storyline progression across 320 contiguous bytes ($2,560$ individual bits).
Address formula:
$$\text{Address} = \text{0xD747} + \text{ByteOffset}$$

| Milestone | Byte Offset | Bit Index | Hex Address |
|:---|:---:|:---:|:---:|
| Oak's Parcel Received | `0x0D` | 0 | `0xD754` |
| Delivered Parcel to Oak | `0x0D` | 1 | `0xD754` |
| Received Pokédex | `0x0D` | 2 | `0xD754` |
| Defeated Brock (Gym 1) | `0x6A` | 1 | `0xD7B1` |
| Received Boulder Badge | `0x6A` | 2 | `0xD7B1` |
| Defeated Misty (Gym 2) | `0xEE` | 0 | `0xD835` |
| Received Cascade Badge | `0xEE` | 1 | `0xD835` |
| Defeated Lt. Surge (Gym 3) | `0x7E` | 0 | `0xD7C5` |
| Received Thunder Badge | `0x7E` | 1 | `0xD7C5` |
| Defeated Erika (Gym 4) | `0x8A` | 0 | `0xD7D1` |
| Received Rainbow Badge | `0x8A` | 1 | `0xD7D1` |
| Defeated Koga (Gym 5) | `0xA2` | 0 | `0xD7E9` |
| Received Soul Badge | `0xA2` | 1 | `0xD7E9` |
| Defeated Sabrina (Gym 6) | `0xD4` | 0 | `0xD81B` |
| Received Marsh Badge | `0xD4` | 1 | `0xD81B` |
| Defeated Blaine (Gym 7) | `0xCB` | 0 | `0xD812` |
| Received Volcano Badge | `0xCB` | 1 | `0xD812` |
| Defeated Giovanni (Gym 8) | `0x25` | 0 | `0xD76C` |
| Received Earth Badge | `0x25` | 1 | `0xD76C` |
| Entered Hall of Fame | `0x10` | 0 | `0xD757` |

---

## 3. Safari Zone Address Correction

> **CRITICAL NOTE ON PRIOR ERRORS:**
> Previous community codebases (including Rubinstein's PufferLib script cheat) referenced `0xDA38` as the Safari Zone step counter.
> Disassembly inspection of `pret/pokered` reveals:
> - Canonical addresses: `wSafariSteps` is located at `0xD70D` (low byte) and `0xD70E` (high byte).
> - Initialized to `502` ($0x01F6$) upon entering the gatehouse.
> - Bypassing this counter via script injection violates the cheat-free benchmark contract.
> - The true scientific remedy is Go-Explore state archiving with Directed Frontier Distance (DFD) sampling.
