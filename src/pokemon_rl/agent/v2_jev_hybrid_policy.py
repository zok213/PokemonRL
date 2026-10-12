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
        self.world_navigator = WorldGraphNavigator()
        self.router = OccupancyGridRouter()

        # Anti-stagnation memory
        self.last_pos: Optional[Tuple[int, int, int]] = None
        self.stuck_counter: int = 0
        self.blocked_dirs: set[str] = set()

    def _load_v2_model(self):
        """Load pretrained Stable-Baselines3 PPO policy weights into CPU memory."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
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
        text_active = snapshot.get("text_box_id", 0) != 0
        map_id = snapshot.get("map_id", 0)
        x = snapshot.get("x", 0)
        y = snapshot.get("y", 0)
        badges = snapshot.get("badges", 0)

        # Track spatial stagnation
        cur_pos = (map_id, x, y)
        if cur_pos == self.last_pos and not in_battle and not text_active:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.blocked_dirs.clear()
        self.last_pos = cur_pos

        alpha = self.default_alpha

        # 1. Combat State: Ground in Gen 1 Tactical Battle Math
        if in_battle:
            questions, button_map, situation = self.battle_controller.evaluate_combat_affordances(snapshot)
            battle_type = snapshot.get("battle_type", 1)

            # In battle, heavily bias towards selecting FIGHT / Attack Move (Button A)
            z_aff[ACTION_TO_IDX["a"]] += 3.5
            z_aff[ACTION_TO_IDX["start"]] -= 8.0  # Menus illegal in battle

            # Gen 1 Trainer Rule: Flee is strictly illegal in trainer battles (wIsInBattle == 2)
            if battle_type == 2:
                z_aff[ACTION_TO_IDX["b"]] -= 4.0
                z_aff[ACTION_TO_IDX["down"]] -= 2.0  # Suppress moving to RUN option
            else:
                # Wild battle: if low HP, allow/encourage RUN (DOWN, RIGHT, A)
                active_mon = snapshot.get("active_mon", {})
                hp = active_mon.get("hp", 20)
                max_hp = max(1, active_mon.get("max_hp", 20))
                if (hp / max_hp) < 0.25:
                    z_aff[ACTION_TO_IDX["down"]] += 1.5
                    z_aff[ACTION_TO_IDX["right"]] += 1.5

            return z_aff, situation, 2.0  # High steering weight in combat

        # 2. Dialogue / Menu State: Advance Text with Button A
        if text_active:
            situation = "Dialogue: Text box active -> Advance with Button A"
            z_aff[ACTION_TO_IDX["a"]] += 4.5
            z_aff[ACTION_TO_IDX["start"]] -= 6.0
            return z_aff, situation, 2.5

        # 3. Overworld Navigation State: Ground in Topological World Graph
        subgoal_obj: MapSubgoal = self.world_navigator.get_subgoal(map_id, x, y, badges=badges)
        tg_x, tg_y = subgoal_obj.target_coord

        situation = f"Overworld Navigation: Map {map_id} ({x},{y}) -> {subgoal_obj.goal_name}"

        # Dynamic pathfinding via OccupancyGridRouter
        coll = collision_matrix or [[1] * 20] * 18
        step_dir = self.router.plan_path(coll, (tg_x - x, tg_y - y), blocked_directions=self.blocked_dirs)

        if not step_dir:
            # Vector fallback
            dx = tg_x - x
            dy = tg_y - y
            if abs(dx) >= abs(dy):
                step_dir = "right" if dx > 0 else "left"
            else:
                step_dir = "down" if dy > 0 else "up"

        if step_dir and step_dir in ACTION_TO_IDX:
            z_aff[ACTION_TO_IDX[step_dir]] += 3.0

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
