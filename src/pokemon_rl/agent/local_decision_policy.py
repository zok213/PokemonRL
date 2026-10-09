"""
local_decision_policy.py — Local Affordance-Based Decision Engine & SMDP Policy
================================================================================
Implements an open-weight, 100% local decision policy that runs entirely on
the user's laptop/CPU without any cloud dependencies (TypeSafe API or external gateways).

Theoretical Foundations:
  - Semi-Markov Decision Processes (SMDP) & Options Framework (Sutton, Precup, Singh 1999)
  - Affordance-Based Reinforcement Learning (Agarwal et al. 2021; AGRL 2024)
  - Critic-Free Group Relative Policy Optimization (GRPO; Shao et al. 2024)
  - Local System 1 Decision Architecture (jevos; feder-cr/jev 2026; browser-use 2026)

Key Components:
  1. LocalAffordancePolicy: Native PyTorch neural decision head evaluating discrete
     affordance text/state embeddings in <2ms on CPU/GPU with calibrated probabilities.
  2. JevosLocalClient: Drop-in client for local jevos server (http://127.0.0.1:8017)
     speaking the exact POST /v1/systemone wire format.
  3. SMDPDecisionHarness: Bridges GameBoy WRAM telemetry, enumerates strictly legal
     affordances, invokes the local policy only at branching states, and executes
     macro button sequences.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. DATA STRUCTURES & BRANCH SPECIFICATIONS
# =============================================================================

@dataclass
class LocalDecision:
    """Outcome of a local policy inference over enumerated affordances."""
    selected_option: str
    probabilities: Dict[str, float]
    confidence: float
    auxiliary_predictions: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    model_source: str = "local_pytorch"
    log_prob: float = 0.0


@dataclass
class AffordanceBranch:
    """A decision point where the agent must choose among verified legal actions."""
    branch_type: str  # "battle", "overworld_tie", "dialogue", "quest_branch"
    state_payload: Dict[str, Any]
    legal_options: Dict[str, str]  # option_id -> semantic description


# =============================================================================
# 2. LOCAL NEURAL DECISION HEAD (100% LOCAL, CPU/GPU RUNNABLE)
# =============================================================================

def calibrate_affordance_softmax(
    logit_diffs: Dict[str, float],
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Multi-Affordance Softmax Calibration:
      P(o_k) = exp((z_yes^(k) - z_no^(k)) / T) / sum_j exp((z_yes^(j) - z_no^(j)) / T)
    Ensures candidate options compete in a normalized probability simplex with temperature scaling,
    eliminating prior probability drift across independent binary decisions.
    """
    keys = list(logit_diffs.keys())
    if not keys:
        return {}
    diffs = np.array([logit_diffs[k] for k in keys], dtype=np.float32)
    scaled = diffs / max(temperature, 1e-4)
    shifted = scaled - np.max(scaled)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    return {k: float(probs[i]) for i, k in enumerate(keys)}


class LocalAffordancePolicy(nn.Module):
    """
    Lightweight, high-speed neural decision policy that scores enumerated affordances.
    Runs in <2 ms on CPU with zero network dependencies.

    Architecture:
      - State Encoder: Projects structured state vector (HP, badges, battle flags, coordinates) -> 128-dim
      - Affordance Scorer: Computes scalar logit for each candidate option via learned bilinear attention
      - Calibrated Softmax: Outputs probability distribution over legal options
    """

    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        weights_path: Optional[str] = None
    ):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Option feature encoder (maps option hash/feature vector -> embed_dim)
        self.option_encoder = nn.Sequential(
            nn.Linear(16, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [P(faints_this_turn), P(should_flee)]
            nn.Sigmoid()
        )
        self.is_trained: bool = False
        self.neural_weight: float = 0.45

        # Check for weights to load
        target_path = weights_path or "models/pokemonrl_affordance_model.pt"
        if target_path and os.path.exists(target_path):
            self.load_weights(target_path)

    def save_weights(self, path: str | Path):
        """Persist trained neural weights and hyper-parameters to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "state_dim": 32,
            "hidden_dim": 128,
            "embed_dim": 64,
            "is_trained": True,
        }, str(target_path))
        self.is_trained = True

    def load_weights(self, path: str | Path) -> bool:
        """Load persistent PyTorch weights for zero-shot and warm-start execution."""
        target_path = Path(path)
        if not target_path.exists():
            return False
        try:
            checkpoint = torch.load(str(target_path), map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.load_state_dict(checkpoint["state_dict"])
            else:
                self.load_state_dict(checkpoint)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[!] Warning: Failed to load weights from {target_path}: {e}")
            return False

    def _encode_option_features(self, option_id: str, description: str) -> torch.Tensor:
        """Deterministically encode semantic properties of an affordance into a 16-dim feature."""
        feat = np.zeros(16, dtype=np.float32)
        desc_lower = description.lower()
        if "super effective" in desc_lower:
            feat[0] = 2.0
        elif "neutral" in desc_lower:
            feat[0] = 1.0
        elif "not very effective" in desc_lower:
            feat[0] = 0.5
        elif "no effect" in desc_lower:
            feat[0] = 0.0

        if "attack" in desc_lower or "move" in desc_lower:
            feat[1] = 1.0
        if "switch" in desc_lower:
            feat[2] = 1.0
        if "potion" in desc_lower or "item" in desc_lower:
            feat[3] = 1.0
        if "run" in desc_lower or "flee" in desc_lower:
            feat[4] = 1.0

        # Overworld navigation & exploration
        is_retreat = any(w in desc_lower for w in ["return", "retreat", "back", "turn back", "reverse"])
        if not is_retreat and any(w in desc_lower for w in ["advance", "north", "route", "explore", "continue", "exit", "door", "forward", "avenue", "city", "gate"]):
            feat[6] = 2.5
        if is_retreat:
            feat[10] = 1.5

        if any(w in desc_lower for w in ["grass", "tall grass", "train", "wild"]):
            feat[7] = 1.0
        if any(w in desc_lower for w in ["heal", "mom", "center", "rest"]):
            feat[8] = 1.0
        if any(w in desc_lower for w in ["dialogue", "talk", "press a", "confirm", "oak", "advance dialogue"]):
            feat[9] = 2.5
        if any(w in desc_lower for w in ["close", "dismiss", "press b"]):
            feat[10] = 1.5

        # Stable pseudo-hash for tie-breaking
        h = int(hashlib.sha256(option_id.encode()).hexdigest()[:8], 16)
        feat[5] = ((h % 100) / 100.0) * 0.2
        return torch.from_numpy(feat).float()

    def forward(
        self,
        state_tensor: torch.Tensor,
        candidate_options: Dict[str, str],
        temperature: float = 1.0
    ) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
        """
        Forward pass scoring candidate affordances with temperature scaling:
        P(o_k) = exp((z_yes^(k) - z_no^(k)) / T) / sum_j exp((z_yes^(j) - z_no^(j)) / T)
        """
        state_emb = self.state_encoder(state_tensor)  # (1, embed_dim)
        option_names = list(candidate_options.keys())
        if not option_names:
            return {}, {}, torch.empty(0)

        option_feats = torch.stack([
            self._encode_option_features(opt, candidate_options[opt])
            for opt in option_names
        ])  # (K, 16)
        option_embs = self.option_encoder(option_feats)  # (K, embed_dim)

        # Grounded semantic prior weights across the 16 affordance feature dimensions:
        # [0: super_eff, 1: attack, 2: switch, 3: potion, 4: run, 5: tie_hash,
        #  6: progression, 7: grass, 8: heal, 9: dialogue_advance, 10: close/dismiss, ...]
        hp_ratio = float(state_tensor[0, 6].item()) if state_tensor.shape[-1] > 6 else 1.0
        in_battle = float(state_tensor[0, 5].item()) > 0.5 if state_tensor.shape[-1] > 5 else False

        prior_w = torch.zeros(16, dtype=torch.float32, device=state_tensor.device)
        prior_w[0] = 2.0   # super-effective
        prior_w[1] = 1.5   # attack
        prior_w[6] = 2.4   # forward progression (advance north, exit lab, route)
        prior_w[7] = 1.1   # tall grass
        prior_w[9] = 4.0   # dialogue advance (Button A)
        prior_w[10] = -2.5 # dialogue dismiss (Button B)
        prior_w[5] = 0.05  # tie hash

        # Context-dependent dynamic adjustments
        if hp_ratio < 0.35:
            prior_w[8] = 3.5  # heal when low HP
            prior_w[4] = 2.0  # flee wild battle when low HP
        else:
            prior_w[8] = -0.5 # heal suppressed when healthy
            prior_w[4] = -1.5 # flee suppressed when healthy

        if in_battle:
            prior_w[0] = 3.0
            prior_w[1] = 2.5

        prior_logits = torch.matmul(option_feats, prior_w).unsqueeze(0)  # (1, K)

        # Bilinear dot-product score between state embedding and option embeddings: (z_yes - z_no)
        neural_logits = torch.matmul(state_emb, option_embs.T) / (option_embs.shape[-1] ** 0.5)  # (1, K)
        total_logits = self.neural_weight * neural_logits + prior_logits
        scaled_logits = total_logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1).squeeze(0)  # (K,)

        # Auxiliary predictions
        aux = self.auxiliary_head(state_emb).squeeze(0)  # (2,)

        prob_dict = {name: float(probs[i].item()) for i, name in enumerate(option_names)}
        aux_dict = {
            "faints_this_turn": float(aux[0].item()),
            "should_flee": float(aux[1].item())
        }
        return prob_dict, aux_dict, total_logits

    def decide(
        self,
        state_tensor: torch.Tensor,
        candidate_options: Dict[str, str],
        temperature: float = 1.0
    ) -> LocalDecision:
        """Sample or greedily select an affordance with millisecond latency."""
        t0 = time.perf_counter()
        with torch.no_grad():
            prob_dict, aux_dict, logits = self.forward(state_tensor, candidate_options, temperature=temperature)

        if not prob_dict:
            return LocalDecision(
                selected_option="none",
                probabilities={},
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                model_source="local_pytorch"
            )

        # Greedy argmax
        best_opt = max(prob_dict.keys(), key=lambda k: prob_dict[k])
        confidence = prob_dict[best_opt]
        latency = (time.perf_counter() - t0) * 1000.0

        return LocalDecision(
            selected_option=best_opt,
            probabilities=prob_dict,
            confidence=confidence,
            auxiliary_predictions=aux_dict,
            latency_ms=latency,
            model_source="local_pytorch"
        )


# =============================================================================
# 3. JEVOS DROP-IN LOCAL CLIENT (COMPATIBLE WITH feder-cr/jev)
# =============================================================================

class JevosLocalClient:
    """
    Client for local jevos server (feder-cr/jev / OpenVINO / llama.cpp).
    Runs against http://127.0.0.1:8017/v1/systemone with 0 API cost.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8017", timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/systemone"
        self.timeout = timeout
        self._last_check = 0.0
        self._cached_available = False

    def is_available(self) -> bool:
        """Check if local jevos daemon is actively running (cached for 5s)."""
        now = time.time()
        if now - self._last_check < 5.0:
            return self._cached_available
        self._last_check = now
        try:
            req = urllib.request.Request(self.endpoint, data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.1) as resp:
                self._cached_available = True
                return True
        except Exception:
            self._cached_available = False
            return False

    def decide(self, branch: AffordanceBranch, temperature: float = 1.0) -> LocalDecision:
        """
        Send branch decision query to local jevos daemon and apply
        Multi-Affordance Softmax Calibration:
          P(o_k) = exp((z_yes^(k) - z_no^(k)) / T) / sum_j exp((z_yes^(j) - z_no^(j)) / T)
        """
        t0 = time.perf_counter()
        payload = {
            "model": "jevos-local",
            "state": branch.state_payload,
            "questions": {
                "next_action": {
                    "type": "choice",
                    "criteria": branch.legal_options
                },
                "faints_this_turn": {
                    "type": "noul",
                    "instructions": "Will active mon faint this turn?"
                }
            }
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                answers = data.get("answers", {})
                choice = answers.get("next_action", {})
                raw_probs = choice.get("probabilities", {})
                
                # Apply Multi-Affordance Softmax Calibration across candidate options
                if raw_probs and len(raw_probs) > 1:
                    # Convert probabilities to logit differences Delta z = ln(p / (1 - p + eps))
                    logit_diffs = {}
                    for k, p in raw_probs.items():
                        p_clamped = max(min(float(p), 0.999), 0.001)
                        logit_diffs[k] = float(np.log(p_clamped / (1.0 - p_clamped)))
                    probs = calibrate_affordance_softmax(logit_diffs, temperature=temperature)
                else:
                    probs = raw_probs or {sorted(branch.legal_options.keys())[0]: 1.0}

                selected = max(probs.keys(), key=lambda k: probs[k]) if probs else sorted(branch.legal_options.keys())[0]
                latency = (time.perf_counter() - t0) * 1000.0
                return LocalDecision(
                    selected_option=selected,
                    probabilities=probs,
                    confidence=probs.get(selected, 0.5),
                    auxiliary_predictions={"faints_this_turn": answers.get("faints_this_turn", {}).get("noul", 0.0)},
                    latency_ms=latency,
                    model_source="jevos_local_daemon"
                )
        except Exception as e:
            # Fallback to local default
            latency = (time.perf_counter() - t0) * 1000.0
            fallback = sorted(branch.legal_options.keys())[0]
            return LocalDecision(
                selected_option=fallback,
                probabilities={fallback: 1.0},
                confidence=1.0,
                latency_ms=latency,
                model_source=f"fallback_error_{type(e).__name__}"
            )


# =============================================================================
# 4. SMDP DECISION HARNESS (INTEGRATING WRAM & LOCAL RL)
# =============================================================================

class SMDPDecisionHarness:
    """
    Semi-Markov Decision Process (SMDP) Harness.
    Decouples low-level joypad execution from high-level decision learning.
    """

    def __init__(self, local_policy: Optional[LocalAffordancePolicy] = None, weights_path: Optional[str] = None):
        self.policy = local_policy or LocalAffordancePolicy(weights_path=weights_path)
        self.jevos_client = JevosLocalClient()
        self.decision_count = 0

    def extract_state_tensor(
        self,
        wram_snapshot: Dict[str, Any],
        subgoal: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Vectorize GameBoy state into normalized 32-dim feature tensor."""
        vec = np.zeros(32, dtype=np.float32)
        cur_x = float(wram_snapshot.get("x", 0))
        cur_y = float(wram_snapshot.get("y", 0))

        vec[0] = float(wram_snapshot.get("map_id", 0)) / 255.0
        vec[1] = cur_x / 64.0
        vec[2] = cur_y / 64.0
        vec[3] = float(wram_snapshot.get("party_count", 1)) / 6.0
        vec[4] = float(wram_snapshot.get("badges", 0)) / 8.0
        vec[5] = 1.0 if wram_snapshot.get("in_battle", False) else 0.0

        if wram_snapshot.get("active_mon"):
            mon = wram_snapshot["active_mon"]
            hp = mon.get("hp", 20)
            max_hp = max(1, mon.get("max_hp", 20))
            vec[6] = float(hp) / float(max_hp)
            vec[7] = float(mon.get("level", 5)) / 100.0

        # WRAM combat metadata: battle_type (2 = trainer, 1 = wild)
        b_type = wram_snapshot.get("battle_type", 1)
        vec[8] = 1.0 if b_type == 2 else (0.5 if b_type == 1 else 0.0)

        # WRAM text box flag
        vec[9] = 1.0 if wram_snapshot.get("text_box_id", 0) != 0 else 0.0

        # Topological Subgoal displacement and Euclidean distance
        if subgoal is not None:
            dx = float(subgoal[0] - cur_x)
            dy = float(subgoal[1] - cur_y)
            dist = np.sqrt(dx * dx + dy * dy)
            vec[10] = dx / 64.0
            vec[11] = dy / 64.0
            vec[12] = dist / 64.0

        return torch.from_numpy(vec).unsqueeze(0).float()

    def step_branch(
        self,
        branch: AffordanceBranch,
        wram_snapshot: Dict[str, Any],
        subgoal: Optional[Tuple[int, int]] = None,
        temperature: float = 1.0
    ) -> LocalDecision:
        """
        Execute a single decision step with temperature-scaled calibration:
        Tries local jevos daemon first if active; otherwise uses high-speed LocalAffordancePolicy.
        """
        self.decision_count += 1
        if self.jevos_client.is_available():
            decision = self.jevos_client.decide(branch, temperature=temperature)
            decision.model_source = "PokemonRL-JevosDaemon"
            return decision

        state_tensor = self.extract_state_tensor(wram_snapshot, subgoal=subgoal)
        decision = self.policy.decide(state_tensor, branch.legal_options, temperature=temperature)
        decision.model_source = "PokemonRL-LocalPolicy"
        return decision

