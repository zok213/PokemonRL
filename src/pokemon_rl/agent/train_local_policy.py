"""
train_local_policy.py — Neural Policy Training & Distillation for PokemonRL
=============================================================================
Trains LocalAffordancePolicy on representative Kanto affordance decisions
using PyTorch with cross-entropy and auxiliary multi-task loss.
Saves persistent trained weights to models/pokemonrl_affordance_model.pt.
"""

from __future__ import annotations
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root and src are on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pokemon_rl.agent.local_decision_policy import LocalAffordancePolicy


def generate_training_samples() -> List[Tuple[np.ndarray, Dict[str, str], str, float, float]]:
    """
    Generate representative grounded decision states across Kanto gameplay.
    Each sample: (state_vector_32, candidate_options_dict, target_option, target_faint, target_flee)
    """
    samples = []

    # 1. Overworld Navigation Samples (Route 1, Viridian, Forest, Pewter)
    # Target should be 'advance' or directional forward progress
    for map_id in [0, 1, 12, 13, 50, 51, 47, 2]:
        for hp_ratio in [1.0, 0.8, 0.5]:
            s = np.zeros(32, dtype=np.float32)
            s[0] = map_id / 255.0
            s[1] = random.uniform(5, 25) / 64.0
            s[2] = random.uniform(5, 45) / 64.0
            s[3] = 1.0 / 6.0
            s[4] = 0.0
            s[5] = 0.0  # Not in battle
            s[6] = hp_ratio
            s[7] = 0.07  # Level 7
            s[10] = 0.0  # dx
            s[11] = -0.5 # dy (north target)
            s[12] = 0.3  # distance

            options = {
                "advance": "Should Red advance north along the open path towards destination?",
                "explore_grass": "Should Red step into roadside grass to train Pokémon?",
                "retreat": "Should Red return south towards Pallet Town?"
            }
            samples.append((s, options, "advance", 0.0, 0.0))

            # Dialogue state
            s_diag = s.copy()
            s_diag[9] = 1.0  # Dialogue active
            diag_opts = {
                "advance_dialogue": "Should Red press A to advance conversation text?",
                "close_menu": "Should Red press B to close menu?"
            }
            samples.append((s_diag, diag_opts, "advance_dialogue", 0.0, 0.0))

    # 2. Combat Samples: Super-Effective vs Neutral vs Ineffective Attacks
    for level in [5, 7, 10]:
        # Weedle/Caterpie (Bug) vs Charmander (Fire) -> Ember is super-effective
        s_combat = np.zeros(32, dtype=np.float32)
        s_combat[0] = 51 / 255.0  # Viridian Forest
        s_combat[5] = 1.0         # in_battle
        s_combat[6] = 0.85        # Healthy HP
        s_combat[7] = level / 100.0
        s_combat[8] = 0.5         # Wild battle (1/2)

        options = {
            "attack_ember": "Should Red execute Ember (Power 40, Fire, ExpDmg ~14) [SUPER EFFECTIVE]?",
            "attack_scratch": "Should Red execute Scratch (Power 40, Normal, ExpDmg ~6)?",
            "run": "Should Red attempt to flee from this wild encounter?"
        }
        samples.append((s_combat, options, "attack_ember", 0.0, 0.0))

        # Geodude (Rock) vs Charmander -> Scratch is Not Very Effective
        options_rock = {
            "attack_scratch": "Should Red execute Scratch (Power 40, Normal, ExpDmg ~2) [NOT VERY EFFECTIVE]?",
            "attack_growl": "Should Red execute Growl to lower enemy attack?",
            "item_potion": "Should Red use a healing potion?"
        }
        samples.append((s_combat, options_rock, "attack_scratch", 0.0, 0.0))

    # 3. Combat Samples: Critical Low HP in Wild vs Trainer Battles
    # Wild encounter at low HP -> Flee
    s_low_wild = np.zeros(32, dtype=np.float32)
    s_low_wild[0] = 51 / 255.0
    s_low_wild[5] = 1.0
    s_low_wild[6] = 0.20  # 20% HP remaining
    s_low_wild[8] = 0.5   # Wild battle
    wild_flee_opts = {
        "attack_scratch": "Should Red order Charmander to attack with Scratch?",
        "run": "Should Red attempt to FLEE from this wild encounter to preserve HP?",
        "item_potion": "Should Red use Potion to restore 20 HP?"
    }
    samples.append((s_low_wild, wild_flee_opts, "run", 0.8, 1.0))

    # Trainer battle at low HP -> MUST NOT FLEE (fleeing illegal in Gen 1)
    s_low_trainer = s_low_wild.copy()
    s_low_trainer[8] = 1.0  # Trainer battle (wIsInBattle == 2)
    trainer_fight_opts = {
        "attack_scratch": "Should Red command an all-out offensive strike to defeat opponent?",
        "attack_ember": "Should Red execute Ember with full offensive power?",
        "item_potion": "Should Red use Potion to restore 20 HP?"
    }
    samples.append((s_low_trainer, trainer_fight_opts, "attack_ember", 0.7, 0.0))

    return samples


def train_policy(epochs: int = 120, lr: float = 3e-3, save_path: str = "models/pokemonrl_affordance_model.pt") -> LocalAffordancePolicy:
    """Train the LocalAffordancePolicy neural weights and save to disk."""
    policy = LocalAffordancePolicy()
    policy.train()
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    raw_data = generate_training_samples()
    print(f"[*] Training LocalAffordancePolicy on {len(raw_data)} representative Kanto state-affordance pairs...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        random.shuffle(raw_data)

        for s_vec, options, target_opt, target_faint, target_flee in raw_data:
            s_tensor = torch.from_numpy(s_vec).unsqueeze(0).float()
            opt_names = list(options.keys())
            target_idx = opt_names.index(target_opt)

            optimizer.zero_grad()
            prob_dict, aux_dict, logits = policy.forward(s_tensor, options, temperature=1.0)

            # Cross-entropy loss on selected affordance
            target_t = torch.tensor([target_idx], dtype=torch.long)
            ce_loss = nn.CrossEntropyLoss()(logits, target_t)

            # Auxiliary loss on predictions
            state_emb = policy.state_encoder(s_tensor)
            aux_pred = policy.auxiliary_head(state_emb)
            aux_target = torch.tensor([[target_faint, target_flee]], dtype=torch.float32)
            aux_loss = nn.BCELoss()(aux_pred, aux_target)

            loss = ce_loss + 0.3 * aux_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_opt = max(prob_dict.keys(), key=lambda k: prob_dict[k])
            if pred_opt == target_opt:
                correct += 1

        acc = (correct / len(raw_data)) * 100.0
        if epoch % 20 == 0 or epoch == epochs:
            avg_loss = total_loss / len(raw_data)
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")

    policy.eval()
    policy.is_trained = True
    policy.save_weights(save_path)
    print(f"[+] Policy successfully trained & weights saved to: {save_path}")
    return policy


if __name__ == "__main__":
    train_policy()
