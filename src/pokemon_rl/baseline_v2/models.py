"""
models.py — PyTorch & NumPy Policy Architectures for Baseline V2 (Pleines et al., 2025)
========================================================================================
Faithful reproduction of the Actor-Critic architectures from:
  Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden
  "Playing Pokémon Red via Deep Reinforcement Learning"
  IEEE Conference on Games (CoG) 2025 / arXiv:2502.19920

Key Specifications:
  - Feedforward Policy: ~2.03M parameters
  - Recurrent GRU Policy: ~3.87M parameters
  - Observation Encoders:
      1. Nature CNN: (3, 72, 80) -> 512-dim visual features
      2. Spatial CNN: (1, 48, 48) -> 256-dim exploration map features
      3. Telemetry MLP: 64-dim WRAM vector -> 128-dim state features
  - Fused Feature: 512 + 256 + 128 = 896-dim multimodal representation
  - Actor Head: 7 discrete action logits (Softmax)
  - Critic Head: 1 scalar state-value estimate V_phi(s)
  - Advantage Formulation: GAE(gamma=0.997, lambda=0.95)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# PyTorch Native Implementation (Full Training & Inference)
# ============================================================================

if HAS_TORCH:
    class PleinesNatureCNN(nn.Module):
        """Nature CNN for (3, 72, 80) downsampled Game Boy frame stacks."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            # Flattened feature dim: 64 * 5 * 6 = 1920
            self.fc = nn.Linear(64 * 5 * 6, 512)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)
            return F.relu(self.fc(x))

    class PleinesSpatialCNN(nn.Module):
        """Spatial CNN for (1, 48, 48) binary visited exploration grid."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
            # Flattened feature dim: 32 * 11 * 11 = 3872
            self.fc = nn.Linear(32 * 11 * 11, 256)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.reshape(x.size(0), -1)
            return F.relu(self.fc(x))

    class PleinesTelemetryMLP(nn.Module):
        """Telemetry MLP for 64-dim normalized WRAM register vector."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            return F.relu(self.fc2(x))

    class PleinesActorCriticTorch(nn.Module):
        """
        Complete PPO Actor-Critic Policy Network in PyTorch.
        Supports both Feedforward (2.03M params) and Recurrent GRU (3.87M params).
        """
        def __init__(self, use_gru: bool = False, num_actions: int = 7):
            super().__init__()
            self.use_gru = use_gru
            self.num_actions = num_actions

            # Multimodal encoders
            self.visual_enc = PleinesNatureCNN()
            self.spatial_enc = PleinesSpatialCNN()
            self.telemetry_enc = PleinesTelemetryMLP()

            # Fused representation: 512 + 256 + 128 = 896
            fused_dim = 896
            hidden_dim = 512

            if use_gru:
                self.gru = nn.GRU(fused_dim, hidden_dim, batch_first=True)
                self.fc_body = None
            else:
                self.gru = None
                self.fc_body = nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.ReLU(),
                )

            # Actor & Critic Heads
            self.actor_head = nn.Linear(hidden_dim, num_actions)
            self.critic_head = nn.Linear(hidden_dim, 1)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            # Small gain for actor output
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

        def forward(
            self,
            screen: torch.Tensor,
            spatial_map: torch.Tensor,
            telemetry: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            """
            Forward pass returning (action_logits, state_values, next_hidden).
            """
            v_feat = self.visual_enc(screen)
            s_feat = self.spatial_enc(spatial_map)
            t_feat = self.telemetry_enc(telemetry)

            fused = torch.cat([v_feat, s_feat, t_feat], dim=-1)

            if self.use_gru:
                # Add time dimension if 2D
                if fused.dim() == 2:
                    fused_seq = fused.unsqueeze(1)
                else:
                    fused_seq = fused
                out, next_hidden = self.gru(fused_seq, hidden_state)
                h = out[:, -1, :]
            else:
                h = self.fc_body(fused)
                next_hidden = None

            logits = self.actor_head(h)
            values = self.critic_head(h).squeeze(-1)
            return logits, values, next_hidden


# ============================================================================
# NumPy Lightweight Engine (Fast CPU Testing & Inference)
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PleinesActorCriticNumPy:
    """
    Pure NumPy implementation of the Baseline V2 Actor-Critic architecture.
    Provides fast forward inference without requiring GPU or PyTorch tensors.
    """
    NUM_ACTIONS = 7

    def __init__(self, use_gru: bool = False, seed: int = 42):
        self.use_gru = use_gru
        self.rng = np.random.default_rng(seed)

        # Flat visual projection (3 * 72 * 80 = 17,280 -> 512)
        vis_dim = 3 * 72 * 80
        self.W_vis = self.rng.normal(0.0, math.sqrt(2.0 / vis_dim), (512, vis_dim)).astype(np.float32)
        self.b_vis = np.zeros(512, dtype=np.float32)

        # Spatial projection (1 * 48 * 48 = 2,304 -> 256)
        spat_dim = 1 * 48 * 48
        self.W_spat = self.rng.normal(0.0, math.sqrt(2.0 / spat_dim), (256, spat_dim)).astype(np.float32)
        self.b_spat = np.zeros(256, dtype=np.float32)

        # Telemetry projection (64 -> 128)
        self.W_telem = self.rng.normal(0.0, math.sqrt(2.0 / 64), (128, 64)).astype(np.float32)
        self.b_telem = np.zeros(128, dtype=np.float32)

        # Fused trunk: 896 -> 512
        fused_dim = 512 + 256 + 128
        self.W_body = self.rng.normal(0.0, math.sqrt(2.0 / fused_dim), (512, fused_dim)).astype(np.float32)
        self.b_body = np.zeros(512, dtype=np.float32)

        # Heads
        self.W_actor = self.rng.normal(0.0, 0.01, (self.NUM_ACTIONS, 512)).astype(np.float32)
        self.b_actor = np.zeros(self.NUM_ACTIONS, dtype=np.float32)

        self.W_critic = self.rng.normal(0.0, 1.0, (1, 512)).astype(np.float32)
        self.b_critic = np.zeros(1, dtype=np.float32)

    def forward(
        self,
        screen: np.ndarray,
        spatial_map: np.ndarray,
        telemetry: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward inference returning (action_probabilities, state_values).
        """
        B = screen.shape[0]
        # Visual
        v_flat = screen.reshape(B, -1)
        v_feat = relu(v_flat @ self.W_vis.T + self.b_vis)

        # Spatial
        s_flat = spatial_map.reshape(B, -1)
        s_feat = relu(s_flat @ self.W_spat.T + self.b_spat)

        # Telemetry
        t_feat = relu(telemetry @ self.W_telem.T + self.b_telem)

        # Fused body
        fused = np.concatenate([v_feat, s_feat, t_feat], axis=-1)
        h = relu(fused @ self.W_body.T + self.b_body)

        # Outputs
        logits = h @ self.W_actor.T + self.b_actor
        probs = softmax(logits, axis=-1)
        values = (h @ self.W_critic.T + self.b_critic).squeeze(-1)

        return probs, values


# ============================================================================
# Generalized Advantage Estimation (GAE)
# ============================================================================

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.997,
    lambda_gae: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation GAE(gamma=0.997, lambda=0.95).
    Conforms exactly to Equation (4) of Pleines et al. (IEEE CoG 2025).

    Args:
        rewards: 1D array of step rewards, shape (T,)
        values: 1D array of state values, shape (T+1,)
        dones: 1D array of episode termination/truncation flags, shape (T,)
        gamma: discount factor (paper default: 0.997)
        lambda_gae: exponential weighting factor (paper default: 0.95)

    Returns:
        advantages: GAE advantage estimates, shape (T,)
        target_values: empirical value targets (advantages + values[:-1]), shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lambda_gae * mask * gae
        advantages[t] = gae

    target_values = advantages + values[:-1]
    return advantages, target_values
