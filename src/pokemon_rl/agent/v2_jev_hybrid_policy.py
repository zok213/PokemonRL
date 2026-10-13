"""
v2_jev_hybrid_policy.py — Residual Policy Steering: Baseline V2 Weights + Jev Affordances
========================================================================================
Implements a publication-grade neuro-symbolic hybrid policy combining:
  1. Peter Whidden's 26.2M-step pretrained PPO policy (poke_26214400.zip) as the visual-motor prior.
  2. Jev-style System 1 local affordance decision engine (TacticalBattleController + WorldGraphNavigator)
     as the topological and tactical supervisor.

Mathematical Formulation:
  pi_hybrid(a | s) = Softmax( (z_v2(a | s) + alpha * z_affordance(a | s, A)) / T )
where:
  - z_v2 in R^7: Raw logit distribution from Baseline V2 MultiInputActorCriticPolicy.
  - z_affordance in R^7: Affordance logit vector grounded in Gen 1 battle math & topological subgoals.
  - alpha in [0.0, 3.0]: Dynamic steering strength (scales up during stagnation / combat forks).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# Add repo root and external directories to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
V2_DIR = REPO_ROOT / "external" / "PokemonRedExperiments" / "v2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from pokemon_rl.agent.tactical_battle import TacticalBattleController
from pokemon_rl.combat.combat_controller import DecoupledCombatController
from pokemon_rl.navigation.world_graph import WorldGraphNavigator, MapSubgoal
from pokemon_rl.navigation.occupancy_router import OccupancyGridRouter

# Action mapping matching Baseline V2
# 0: DOWN, 1: LEFT, 2: RIGHT, 3: UP, 4: A, 5: B, 6: START
ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]
ACTION_TO_IDX = {name.lower(): idx for idx, name in enumerate(ACTION_NAMES)}


@dataclass
class HybridDecision:
    """Outcome of Residual Policy Steering."""
    action_idx: int
    action_name: str
    v2_probs: Dict[str, float]
    steered_probs: Dict[str, float]
    affordance_situation: str
    steering_weight: float
    is_overridden: bool
    latency_ms: float


class BaselineV2JevSteeredPolicy:
    """
    Residual Policy Steering:
    Harnesses 26.2 million frames of visual pretraining from Baseline V2 PPO checkpoint,
    while eliminating long-horizon stagnation and tactical combat blindness via Jev-style
    affordances and topological navigation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        default_alpha: float = 1.2
    ):
        self.device = device
        self.default_alpha = default_alpha

        # Resolve checkpoint
        if checkpoint_path is None:
            cand = V2_DIR / "runs" / "poke_26214400.zip"
            self.checkpoint_path = cand if cand.exists() else None
        else:
            self.checkpoint_path = Path(checkpoint_path)

        self.v2_model = None
        self._load_v2_model()

        # Initialize Jev modular controllers
        self.battle_controller = TacticalBattleController()
        self.combat_controller = DecoupledCombatController()
        self.world_navigator = WorldGraphNavigator()
        self.router = OccupancyGridRouter()

        # Anti-stagnation memory
        self.last_pos: Optional[Tuple[int, int, int]] = None
        self.pos_history: deque[Tuple[int, int, int]] = deque(maxlen=16)
        self.stuck_counter: int = 0
        self.blocked_dirs: set[str] = set()
        self.last_action_name: Optional[str] = None

        # Anti-dialogue lock and retrigger memory
        self.dialogue_steps: int = 0
        self.dialogue_cooldown: int = 0
        self.was_in_dialogue: bool = False

        # Tactical combat flee step tracker
        self.combat_flee_step: int = 0
        self.brock_step: int = 0
        self.was_in_battle: bool = False
        self.battle_cooldown: int = 0
        self.battle_stall_counter: int = 0
        self.last_battle_state: Optional[Tuple[int, int, int, int]] = None

    def _load_v2_model(self):
        """Load pretrained Stable-Baselines3 PPO policy weights into CPU memory."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                # Protect against broken tensorflow installation in user roaming packages
                import sys
                sys.modules.setdefault("tensorflow", None)
                from stable_baselines3 import PPO
                # Use custom_objects to prevent Windows PosixPath deserialization errors
                self.v2_model = PPO.load(
                    str(self.checkpoint_path),
                    device=self.device,
                    custom_objects={"lr_schedule": 0, "clip_range": 0, "tensorboard_log": None}
                )
                print(f"[+] Loaded Baseline V2 Pretrained Weights from: {self.checkpoint_path.name}")
            except Exception as e:
                print(f"[!] Warning: Could not load V2 PPO checkpoint: {e}")
                self.v2_model = None
        else:
            print("[!] Warning: Baseline V2 checkpoint not found; running in pure affordance mode.")

    def compute_v2_logits(self, obs: Any) -> torch.Tensor:
        """Extract raw action logits from the pretrained V2 neural network."""
        if self.v2_model is None or obs is None:
            return torch.zeros(len(ACTION_NAMES), device=self.device)

        try:
            obs_tensor, _ = self.v2_model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self.v2_model.policy.get_distribution(obs_tensor)
                logits = dist.distribution.logits.squeeze(0)
                return logits.to(self.device)
        except Exception:
            return torch.zeros(len(ACTION_NAMES), device=self.device)

    def formulate_affordance_bias(
        self,
        snapshot: Dict[str, Any],
        collision_matrix: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Formulate semantic affordance bias vector z_affordance in R^7 based on live WRAM.
        Returns: (affordance_logits, situation_description, suggested_alpha)
        """
        z_aff = torch.zeros(len(ACTION_NAMES), device=self.device)
        in_battle = snapshot.get("in_battle", False)

        # Precise dialogue detection
        if "text_box_active" in snapshot:
            text_active = bool(snapshot["text_box_active"])
        else:
            text_active = snapshot.get("text_box_id", 0) != 0

        map_id = snapshot.get("map_id", 0)
        x = snapshot.get("x", 0)
        y = snapshot.get("y", 0)
        badges = snapshot.get("badges", 0)

        # Detect transition out of dialogue -> trigger anti-retrigger shield
        if self.was_in_dialogue and not text_active:
            self.dialogue_cooldown = 4
        self.was_in_dialogue = text_active

        # Detect transition out of battle -> battle transition cooldown
        if self.was_in_battle and not in_battle:
            self.battle_cooldown = 2
        self.was_in_battle = in_battle

        # Track spatial stagnation and local oscillation limit cycles
        cur_pos = (map_id, x, y)
        in_transition = (self.dialogue_cooldown > 0) or (self.battle_cooldown > 0)
        if in_transition:
            self.stuck_counter = 0
            self.blocked_dirs.clear()
            self.pos_history.clear()
        elif not in_battle and not text_active:
            self.pos_history.append(cur_pos)
            is_stalled = (cur_pos == self.last_pos)
            is_oscillating = (len(self.pos_history) >= 8 and len(set(self.pos_history)) <= 2)

            if is_stalled or is_oscillating:
                self.stuck_counter += 1
                if self.stuck_counter >= 2 and self.last_action_name in ("down", "left", "right", "up"):
                    self.blocked_dirs.add(self.last_action_name)
            else:
                self.stuck_counter = 0
                self.blocked_dirs.clear()
        else:
            self.stuck_counter = 0
            self.blocked_dirs.clear()
            self.pos_history.clear()
        self.last_pos = cur_pos

        if self.battle_cooldown > 0:
            self.battle_cooldown -= 1

        alpha = self.default_alpha

        # 1. Combat State: Ground in Gen 1 Tactical Battle Mechanics
        if in_battle:
            battle_type = snapshot.get("battle_type", 1)
            active_mon = snapshot.get("active_mon", {})
            hp = active_mon.get("hp", 20)
            max_hp = max(1, active_mon.get("max_hp", 20))
            opp_hp = snapshot.get("opponent_hp", 0)
            cur_item = snapshot.get("battle_cur_item", 0)
            menu_y = snapshot.get("battle_menu_y", 14)
            menu_x = snapshot.get("battle_menu_x", 1)
            text_active = snapshot.get("text_box_active", False)
            move_pps = snapshot.get("move_pps", [35, 40, 25, 30])
            moves = active_mon.get("moves", [0, 0, 0, 0])

            # Menus strictly illegal in battle
            z_aff[ACTION_TO_IDX["start"]] -= 8.0

            # 1. Battle Introduction / Trainer Challenge Phase (wFirstMonsNotOutYet != 0)
            first_mons_not_out = snapshot.get("first_mons_not_out", False)
            if first_mons_not_out:
                self.battle_stall_counter = 0
                z_aff[ACTION_TO_IDX["a"]] += 5.0
                z_aff[ACTION_TO_IDX["b"]] -= 3.0
                situation = "Combat Intro: Both Pokémon not yet sent out -> Advance with Button A"
                return z_aff, situation, 2.0

            # 1b. Check if interactive arrow cursor (0xED) is actively present on screen
            menu_cursor_active = snapshot.get("menu_cursor_active", True)
            if not menu_cursor_active:
                z_aff[ACTION_TO_IDX["a"]] += 5.0
                z_aff[ACTION_TO_IDX["b"]] -= 3.0
                situation = "Combat: Advancing battle dialogue/damage text with Button A"
                return z_aff, situation, 2.0

            # Track battle menu stall to prevent any infinite menu/PP/disable deadlocks
            battle_state = (hp, opp_hp, cur_item, menu_y, battle_type)
            if battle_state == self.last_battle_state:
                self.battle_stall_counter += 1
            else:
                self.battle_stall_counter = 0
                self.last_battle_state = battle_state

            # If stalled in battle for >= 4 steps, unjam with menu-cycling / alternative move
            if self.battle_stall_counter >= 4:
                unjam_pattern = ["down", "a", "b", "down", "a", "b", "up", "a"]
                unjam_act = unjam_pattern[self.battle_stall_counter % len(unjam_pattern)]
                z_aff[ACTION_TO_IDX[unjam_act]] += 5.0
                situation = f"Combat: Menu/PP stall detected ({self.battle_stall_counter} steps) -> Unjamming via {unjam_act.upper()}"
                return z_aff, situation, 2.5

            # If active mon fainted (HP == 0)
            if hp <= 0:
                self.combat_flee_step = 0
                party_hps = snapshot.get("party_hps", [])
                living_slots = [i for i, h in enumerate(party_hps) if h > 0]

                # If all party members are fainted, blackout is imminent
                if not living_slots:
                    z_aff[ACTION_TO_IDX["a"]] += 5.0
                    situation = "Combat: Entire party fainted -> Advance blackout dialogue with Button A"
                    return z_aff, situation, 2.0

                target_slot = living_slots[0]

                # If in the party switch menu (menu_y < 4): navigate to living slot and select!
                if menu_y < 4:
                    if cur_item < target_slot:
                        z_aff[ACTION_TO_IDX["down"]] += 5.0
                        z_aff[ACTION_TO_IDX["a"]] -= 3.0
                        situation = f"Combat: Party menu -> Moving cursor DOWN to Slot {target_slot+1}"
                        return z_aff, situation, 2.0
                    elif cur_item > target_slot:
                        z_aff[ACTION_TO_IDX["up"]] += 5.0
                        z_aff[ACTION_TO_IDX["a"]] -= 3.0
                        situation = f"Combat: Party menu -> Moving cursor UP to Slot {target_slot+1}"
                        return z_aff, situation, 2.0
                    else:
                        z_aff[ACTION_TO_IDX["a"]] += 5.0
                        situation = f"Combat: Party menu -> Confirm Slot {target_slot+1} with Button A"
                        return z_aff, situation, 2.0

                # Otherwise text box is still displaying ('<Mon> fainted! ▼'): advance with Button A
                z_aff[ACTION_TO_IDX["a"]] += 5.0
                situation = "Combat: Active mon fainted -> Advancing battle text with Button A"
                return z_aff, situation, 2.0

            # Active Mon is ALIVE (HP > 0):
            # 2. Inside Sub-menu (ITEM bag or PKMN party, menu_y < 4): cancel back with Button B
            if menu_y < 4:
                z_aff[ACTION_TO_IDX["b"]] += 5.0
                z_aff[ACTION_TO_IDX["a"]] -= 3.0
                situation = f"Combat: In sub-menu (menu_y={menu_y}) -> Press Button B to cancel back to battle menu"
                return z_aff, situation, 2.0

            # 3. Move Selection Menu (menu_y in (7, 12)): choose optimal damaging move!
            disabled_slot = snapshot.get("disabled_move_slot", -1)
            valid_moves = [
                (idx, move_pps[idx] if idx < len(move_pps) else 0, moves[idx] if idx < len(moves) else 0)
                for idx in range(min(len(moves), len(move_pps)))
                if (move_pps[idx] if idx < len(move_pps) else 0) > 0
                and (moves[idx] if idx < len(moves) else 0) > 0
                and idx != disabled_slot
            ]

            if menu_y in (7, 12) and valid_moves:
                opp_species = snapshot.get("opponent_species", 0)
                user_species = snapshot.get("user_species", 0)
                if opp_species == 0:
                    if map_id == 54:
                        opp_species = 0x22  # Onix (Brock Gym)
                    elif map_id == 65:
                        opp_species = 0x98  # Starmie (Misty Gym)
                if user_species == 0:
                    active_moves = snapshot.get("active_mon", {}).get("moves", [])
                    if 145 in active_moves or 55 in active_moves:
                        user_species = 0xB1  # Squirtle
                    elif 52 in active_moves:
                        user_species = 0xB0  # Charmander
                    elif 22 in active_moves:
                        user_species = 0x99  # Bulbasaur

                opp_def_mod = snapshot.get("opp_def_mod", 7)
                user_atk_mod = snapshot.get("user_atk_mod", 7)

                def move_evaluator(item):
                    idx, pp, m_id = item
                    if pp <= 0:
                        return -100.0
                    return self.combat_controller.evaluate_move_by_id(
                        move_id=m_id,
                        opp_species_id=opp_species,
                        user_species_id=user_species,
                        opp_def_mod=opp_def_mod,
                        user_atk_mod=user_atk_mod,
                    )

                best_target = max(valid_moves, key=move_evaluator)[0]
                target_menu_item = best_target + 1  # In Gen 1 MoveSelectionMenu, wCurrentMenuItem is strictly 1-based (1..4)
                if cur_item < target_menu_item:
                    z_aff[ACTION_TO_IDX["down"]] += 4.5
                    z_aff[ACTION_TO_IDX["a"]] -= 3.0
                    situation = f"Trainer Combat: Selecting Move {best_target+1} -> Move DOWN"
                    return z_aff, situation, 2.0
                elif cur_item > target_menu_item:
                    z_aff[ACTION_TO_IDX["up"]] += 4.5
                    z_aff[ACTION_TO_IDX["a"]] -= 3.0
                    situation = f"Trainer Combat: Selecting Move {best_target+1} -> Move UP"
                    return z_aff, situation, 2.0
                else:
                    z_aff[ACTION_TO_IDX["a"]] += 5.0
                    z_aff[ACTION_TO_IDX["b"]] -= 3.0
                    situation = f"Trainer Combat: Cursor on valid Move {best_target+1} -> Executing attack with Button A"
                    return z_aff, situation, 2.0

            # 4. Main Battle Menu (wTopMenuItemY == 14 in Gen 1 disassembly): select FIGHT
            if menu_y == 14:
                if battle_type == 2:
                    # In Trainer Combat: Flee is illegal. Always navigate 2x2 grid to FIGHT (left column, row 0)
                    if cur_item >= 2:
                        # wCurrentMenuItem >= 2 indicates right-column confirmation or sub-selection
                        z_aff[ACTION_TO_IDX["b"]] += 4.0
                        z_aff[ACTION_TO_IDX["left"]] += 4.0
                        situation = "Trainer Combat: Main battle menu (cur_item >= 2) -> Cancel/Reset LEFT to FIGHT"
                        return z_aff, situation, 2.0
                    elif menu_x >= 10:
                        # In right column (PKMN/RUN): pressing LEFT moves to left column (FIGHT/ITEM)
                        z_aff[ACTION_TO_IDX["left"]] += 5.0
                        z_aff[ACTION_TO_IDX["a"]] -= 4.0
                        situation = "Trainer Combat: In right column (menu_x >= 10) -> Navigate LEFT to FIGHT"
                        return z_aff, situation, 2.0
                    elif cur_item == 0:
                        z_aff[ACTION_TO_IDX["a"]] += 5.0
                        z_aff[ACTION_TO_IDX["b"]] -= 3.0
                        situation = "Trainer Combat: Main battle menu -> Choose FIGHT with Button A"
                        return z_aff, situation, 2.0
                    else:
                        # Item 1 is ITEM (bottom-left): only UP moves cursor to FIGHT (top-left)
                        z_aff[ACTION_TO_IDX["up"]] += 5.0
                        z_aff[ACTION_TO_IDX["a"]] -= 4.0
                        situation = "Trainer Combat: Main battle menu (item 1/ITEM) -> Navigate UP to FIGHT"
                        return z_aff, situation, 2.0
                else:
                    # Wild Battle: in dungeons/caves (Mt Moon 59, 60, 61) or Route 3/4 (14, 15) when level >= 10,
                    # or low HP, always flee to conserve HP, PP, and time for dungeon traversal!
                    should_flee = (
                        (map_id in (59, 60, 61, 14, 15) and active_mon.get("level", 1) >= 10)
                        or ((hp / max_hp) < 0.60)
                        or (hp <= 14)
                    )
                    if should_flee:
                        if cur_item >= 2:
                            z_aff[ACTION_TO_IDX["b"]] += 4.0
                            situation = "Wild Combat: Main menu cursor >= 2 -> Cancel with Button B"
                            return z_aff, situation, 2.0
                        elif menu_x < 10:
                            z_aff[ACTION_TO_IDX["right"]] += 5.0
                            situation = "Wild Combat: In left column -> Navigate RIGHT towards RUN"
                            return z_aff, situation, 2.0
                        elif cur_item == 0:
                            z_aff[ACTION_TO_IDX["down"]] += 5.0
                            situation = "Wild Combat: In right column (PKMN) -> Navigate DOWN to RUN"
                            return z_aff, situation, 2.0
                        elif cur_item == 1:
                            z_aff[ACTION_TO_IDX["a"]] += 5.0
                            situation = "Wild Combat: Main menu cursor at RUN -> Flee with Button A"
                            return z_aff, situation, 2.0
                        else:
                            z_aff[ACTION_TO_IDX["b"]] += 4.0
                            situation = "Wild Combat: Main menu cursor >= 2 -> Cancel with Button B"
                            return z_aff, situation, 2.0
                    else:
                        if cur_item >= 2:
                            z_aff[ACTION_TO_IDX["b"]] += 4.0
                            z_aff[ACTION_TO_IDX["left"]] += 4.0
                            situation = "Wild Combat: Main battle menu (cur_item >= 2) -> Cancel/Reset LEFT to FIGHT"
                            return z_aff, situation, 2.0
                        elif menu_x >= 10:
                            z_aff[ACTION_TO_IDX["left"]] += 5.0
                            situation = "Wild Combat: In right column -> Navigate LEFT to FIGHT"
                            return z_aff, situation, 2.0
                        elif cur_item == 0:
                            z_aff[ACTION_TO_IDX["a"]] += 5.0
                            situation = "Wild Combat: Main battle menu -> Choose FIGHT with Button A"
                            return z_aff, situation, 2.0
                        else:
                            z_aff[ACTION_TO_IDX["up"]] += 5.0
                            situation = "Wild Combat: Main battle menu (ITEM) -> Navigate UP to FIGHT"
                            return z_aff, situation, 2.0

            # 5. Dialogue / Damage Text Active (moves executing, damage dealt, dialogue)
            z_aff[ACTION_TO_IDX["a"]] += 5.0
            z_aff[ACTION_TO_IDX["b"]] -= 3.0
            situation = "Combat: Advancing battle dialogue/damage text with Button A"
            return z_aff, situation, 2.0
        else:
            self.combat_flee_step = 0
            self.battle_stall_counter = 0
            self.last_battle_state = None

        # 2. Dialogue / Menu State: Advance Text with Button A / B and break loops
        if text_active:
            self.dialogue_steps += 1
            # Inside Pokecenter (Viridian 41, Pewter 58, Cerulean 64, Mt. Moon 68), Nurse Joy asks "Shall we heal your POKeMON?" -> Always select YES with Button A!
            if map_id in (41, 58, 64, 68):
                situation = f"Pokécenter Dialogue: Healing party with Nurse Joy (step {self.dialogue_steps}) -> Button A"
                z_aff[ACTION_TO_IDX["a"]] += 5.0
                z_aff[ACTION_TO_IDX["b"]] -= 3.0
                return z_aff, situation, 2.5

            # If a Pokemon is actively evolving, always press Button A to confirm (never press B to cancel)
            if snapshot.get("is_evolving", False):
                situation = f"Evolution Dialogue: Mon evolving (step {self.dialogue_steps}) -> Confirm evolution with Button A"
                z_aff[ACTION_TO_IDX["a"]] += 5.0
                z_aff[ACTION_TO_IDX["b"]] -= 8.0
                return z_aff, situation, 2.5

            # On Mt Moon B2F around the fossil: always advance/confirm with Button A (confirm "YES" on prompt)
            if map_id == 61 and (x, y) in ((11, 6), (12, 6), (13, 6), (12, 7), (13, 7)):
                situation = f"Mt Moon B2F Fossil Dialogue: Confirming fossil with Button A (step {self.dialogue_steps})"
                z_aff[ACTION_TO_IDX["a"]] += 5.5
                z_aff[ACTION_TO_IDX["b"]] -= 4.0
                return z_aff, situation, 2.5

            if self.dialogue_steps <= 3:
                # Early dialogue: advance text rapidly with Button A
                situation = f"Dialogue: Advancing text with Button A (step {self.dialogue_steps})"
                z_aff[ACTION_TO_IDX["a"]] += 4.5
                z_aff[ACTION_TO_IDX["start"]] -= 6.0
            else:
                # Prolonged dialogue: alternate B and A to safely dismiss without re-triggering NPC interaction
                if self.dialogue_steps % 2 == 0:
                    situation = f"Dialogue: Anti-loop dismiss active (step {self.dialogue_steps}) -> Dismiss with Button B"
                    z_aff[ACTION_TO_IDX["b"]] += 4.5
                    z_aff[ACTION_TO_IDX["a"]] -= 4.0
                else:
                    situation = f"Dialogue: Advancing text with Button A (step {self.dialogue_steps})"
                    z_aff[ACTION_TO_IDX["a"]] += 4.5
                    z_aff[ACTION_TO_IDX["b"]] -= 4.0
                z_aff[ACTION_TO_IDX["start"]] -= 6.0
            return z_aff, situation, 2.5

        # Reset dialogue steps when dialogue is not active
        self.dialogue_steps = 0

        active_mon = snapshot.get("active_mon", {})
        hp = active_mon.get("hp", 20)
        max_hp = max(1, active_mon.get("max_hp", 20))

        # 3. Overworld Navigation State: Ground in Topological World Graph
        subgoal_obj: MapSubgoal = self.world_navigator.get_subgoal(
            map_id, x, y, badges=badges, hp=hp, max_hp=max_hp, has_fossil=snapshot.get("has_fossil", False)
        )
        tg_x, tg_y = subgoal_obj.target_coord

        situation = f"Overworld Navigation: Map {map_id} ({x},{y}) -> {subgoal_obj.goal_name}"

        # Unified Orthogonal Interaction Law (Applies to all NPC / item subgoals)
        if subgoal_obj.is_interaction or subgoal_obj.goal_name in ("claim_dome_fossil", "interact_brock", "challenge_misty", "challenge_misty_from_right", "interact_nurse_joy"):
            dx = tg_x - x
            dy = tg_y - y
            # In Gen 1 WRAM (wPlayerFacingDirection at 0xC109):
            # 0: DOWN, 4: UP, 8: LEFT, 12: RIGHT
            dir_map = {
                (1, 0): (12, "right"),
                (-1, 0): (8, "left"),
                (0, 1): (0, "down"),
                (0, -1): (4, "up"),
            }
            req = dir_map.get((dx, dy))
            if req is not None:
                req_dir, req_act = req
                player_dir = snapshot.get("player_dir", 0)
                if player_dir == req_dir:
                    z_aff[ACTION_TO_IDX["a"]] += 6.0
                    situation = f"Interaction: Facing target at ({tg_x},{tg_y}) -> Press Button A"
                    return z_aff, situation, 2.8
                else:
                    z_aff[ACTION_TO_IDX[req_act]] += 6.0
                    situation = f"Interaction: Adjacent to ({tg_x},{tg_y}) -> Turn {req_act.upper()} to face target"
                    return z_aff, situation, 2.8

        # Direct doorway / exit mat steps for small indoor maps
        dx = tg_x - x
        dy = tg_y - y
        if subgoal_obj.is_warp and subgoal_obj.goal_name == "step_outside_pokecenter":
            if x < 3:
                step_dir = "right"
            elif x > 4:
                step_dir = "left"
            else:
                step_dir = "down"
        elif subgoal_obj.is_warp and (abs(dx) + abs(dy) == 1):
            if dy == 1:
                step_dir = "down"
            elif dy == -1:
                step_dir = "up"
            elif dx == 1:
                step_dir = "right"
            else:
                step_dir = "left"
        elif subgoal_obj.is_warp and dx == 0 and dy == 0:
            if "exit" in subgoal_obj.goal_name or "outside" in subgoal_obj.goal_name:
                step_dir = "down"
            elif "east" in subgoal_obj.goal_name or "cerulean" in subgoal_obj.goal_name:
                step_dir = "right"
            else:
                step_dir = "up"
        elif abs(dx) + abs(dy) == 1:
            # Geodesic BFS step: directly adjacent target tile on authentic ROM grid
            if dy == 1:
                direct_dir = "down"
            elif dy == -1:
                direct_dir = "up"
            elif dx == 1:
                direct_dir = "right"
            else:
                direct_dir = "left"
            if direct_dir not in self.blocked_dirs:
                step_dir = direct_dir
            else:
                # Direct step blocked by dynamic obstacle -> detour via OccupancyGridRouter
                coll = collision_matrix if collision_matrix is not None else [[1] * 20] * 18
                step_dir = self.router.plan_path(coll, (dx, dy), blocked_directions=self.blocked_dirs)
        else:
            # Dynamic pathfinding via OccupancyGridRouter
            coll = collision_matrix if collision_matrix is not None else [[1] * 20] * 18
            step_dir = self.router.plan_path(coll, (dx, dy), blocked_directions=self.blocked_dirs)

            if not step_dir:
                # Vector fallback with blocked direction avoidance
                candidates = []
                if abs(dx) >= abs(dy):
                    candidates = ["right" if dx > 0 else "left", "down" if dy > 0 else "up"]
                else:
                    candidates = ["down" if dy > 0 else "up", "right" if dx > 0 else "left"]

                # Select first unblocked candidate
                step_dir = next((d for d in candidates if d not in self.blocked_dirs), candidates[0])

        opposite_map = {"up": "down", "down": "up", "left": "right", "right": "left"}
        if step_dir and step_dir in ACTION_TO_IDX:
            z_aff[ACTION_TO_IDX[step_dir]] += 5.5
            opp = opposite_map.get(step_dir)
            if opp and opp in ACTION_TO_IDX:
                z_aff[ACTION_TO_IDX[opp]] -= 4.5

        # In caves and dungeons, ensure sufficient steering authority to follow BFS waypoints
        if map_id in (59, 60, 61, 14, 15):
            alpha = max(alpha, 1.8)

        # In overworld navigation: suppress non-directional buttons (A, B, START) to avoid accidental NPC dialogue
        z_aff[ACTION_TO_IDX["a"]] -= 4.0
        z_aff[ACTION_TO_IDX["b"]] -= 4.0
        z_aff[ACTION_TO_IDX["start"]] -= 8.0

        # Critical Warp Doorway Transition:
        # When stepping into doors, gatehouses, or cave ladders, prevent prior drift from veering off-target
        if subgoal_obj.is_warp:
            alpha = max(alpha, 2.8)
            if step_dir and step_dir in ACTION_TO_IDX:
                z_aff[ACTION_TO_IDX[step_dir]] += 5.0
            for d in ("up", "down", "left", "right"):
                if d != step_dir:
                    z_aff[ACTION_TO_IDX[d]] -= 4.0
            z_aff[ACTION_TO_IDX["a"]] -= 6.0
            z_aff[ACTION_TO_IDX["b"]] -= 6.0
            z_aff[ACTION_TO_IDX["start"]] -= 8.0
            situation += f" [Warp Doorway Transition: Steering '{step_dir.upper()}', alpha={alpha:.1f}]"

        # Suppress any currently blocked directions
        for b_dir in self.blocked_dirs:
            if b_dir in ACTION_TO_IDX:
                z_aff[ACTION_TO_IDX[b_dir]] -= 6.0

        # Post-Dialogue Anti-Retrigger Shield:
        # After conversation closes, heavily suppress Button A to prevent re-opening dialogue with NPC
        if self.dialogue_cooldown > 0:
            self.dialogue_cooldown -= 1
            z_aff[ACTION_TO_IDX["a"]] -= 8.0
            situation += f" [Anti-Retrigger Active: Suppressing 'A' for {self.dialogue_cooldown + 1} steps]"

        # In Pewter Gym before Brock is defeated: prevent premature exit towards southern doorway
        if map_id == 54 and (badges & 1) == 0:
            if y >= 6:
                z_aff[ACTION_TO_IDX["down"]] -= 8.0
            alpha = max(alpha, 2.5)

        # Anti-Stagnation Circuit Breaker: If stuck for >2 frames, scale up steering
        if self.stuck_counter >= 2:
            alpha = min(3.5, self.default_alpha + self.stuck_counter * 0.5)
            situation += f" [Anti-Stagnation Active: Stalled {self.stuck_counter} steps, alpha={alpha:.1f}]"

        return z_aff, situation, alpha

    def decide(
        self,
        snapshot: Dict[str, Any],
        obs: Optional[Any] = None,
        collision_matrix: Optional[List[List[int]]] = None,
        temperature: float = 1.0
    ) -> HybridDecision:
        """
        Produce a steered action decision fusing Baseline V2 weights with Jev affordances:
          pi(a | s) = Softmax( (z_v2 + alpha * z_aff) / T )
        """
        import time
        t0 = time.perf_counter()

        # 1. Raw V2 Logits
        z_v2 = self.compute_v2_logits(obs)
        v2_probs_tensor = F.softmax(z_v2, dim=-1)
        v2_probs = {name: float(v2_probs_tensor[i].item()) for i, name in enumerate(ACTION_NAMES)}

        # 2. Jev Affordance Biases
        z_aff, situation, alpha = self.formulate_affordance_bias(snapshot, collision_matrix)

        # 3. Policy Blending (Residual Steering)
        z_total = z_v2 + alpha * z_aff
        scaled_logits = z_total / max(temperature, 1e-4)
        steered_probs_tensor = F.softmax(scaled_logits, dim=-1)
        steered_probs = {name: float(steered_probs_tensor[i].item()) for i, name in enumerate(ACTION_NAMES)}

        # 4. Action Selection (Greedy Argmax or Sampling)
        best_idx = int(torch.argmax(steered_probs_tensor).item())
        best_name = ACTION_NAMES[best_idx]

        # Check if affordance steering altered V2's raw argmax
        v2_raw_idx = int(torch.argmax(z_v2).item())
        is_overridden = (best_idx != v2_raw_idx) and (self.v2_model is not None)

        # Record chosen action for obstacle bump detection
        self.last_action_name = best_name.lower()

        latency = (time.perf_counter() - t0) * 1000.0

        return HybridDecision(
            action_idx=best_idx,
            action_name=best_name,
            v2_probs=v2_probs,
            steered_probs=steered_probs,
            affordance_situation=situation,
            steering_weight=alpha,
            is_overridden=is_overridden,
            latency_ms=latency
        )
