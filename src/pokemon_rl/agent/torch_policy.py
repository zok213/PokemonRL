"""
torch_policy.py — PyTorch Implementation of MultiModal Policy Network with Feature Warm-Start
=============================================================================================
Transfers Peter Whidden's pretrained convolutional visual feature extractor
(from external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip)
into our SOTA Critic-Free MultiModal Policy Network.

Architectural Highlights:
  1. Visual Stream (Warm-Started):
     Conv2d(3, 32, k8, s4) -> Conv2d(32, 64, k4, s2) -> Conv2d(64, 64, k3, s1)
     -> AdaptiveAvgPool2d((3, 4)) -> Linear(768, 512) -> LayerNorm(512)
     Transfers all 4 weight & bias tensors from the 439,746,560-step checkpoint!
  2. Spatial Stream: Visited coordinate map (1, 48, 48) -> Linear(256) -> LayerNorm(256)
  3. WRAM Stream: 64 telemetry registers -> Linear(128) -> LayerNorm(128)
  4. Modality Balanced Fusion: Concatenate [512 + 256 + 128 = 896] -> Linear(512) -> 8 action logits
  5. Critic Head: ZERO (Critic-Free GRPO)
  6. Action Masking: Integrated zero-leak suppression via wJoyIgnore (0xCD6B)
"""

from __future__ import annotations
import io
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class WarmStartedVisualEncoder(nn.Module):
    """
    Nature CNN Visual Encoder initialized from Whidden's 439M-step checkpoint.
    Uses AdaptiveAvgPool2d((3, 4)) to bridge 72x80 resolution with Whidden's 768 flat dimension.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 512):
        super().__init__()
        self.cnn0 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.linear = nn.Linear(768, out_dim)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim)
        self.is_warm_started = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.cnn0(x))
        h = self.relu(self.cnn2(h))
        h = self.relu(self.cnn4(h))
        h = self.pool(h)
        h = self.relu(self.linear(h.flatten(1)))
        return self.norm(h)

    def load_whidden_weights(self, checkpoint_zip_path: Path | str) -> int:
        """
        Loads all 4 convolutional and linear projection layers from Whidden's checkpoint.
        Returns the number of transferred parameters.
        """
        zip_path = Path(checkpoint_zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open("policy.pth") as f:
                sd = torch.load(io.BytesIO(f.read()), map_location="cpu")

        # Copy layer weights
        self.cnn0.weight.data.copy_(sd["features_extractor.cnn.0.weight"])
        self.cnn0.bias.data.copy_(sd["features_extractor.cnn.0.bias"])
        self.cnn2.weight.data.copy_(sd["features_extractor.cnn.2.weight"])
        self.cnn2.bias.data.copy_(sd["features_extractor.cnn.2.bias"])
        self.cnn4.weight.data.copy_(sd["features_extractor.cnn.4.weight"])
        self.cnn4.bias.data.copy_(sd["features_extractor.cnn.4.bias"])
        self.linear.weight.data.copy_(sd["features_extractor.linear.0.weight"])
        self.linear.bias.data.copy_(sd["features_extractor.linear.0.bias"])

        self.is_warm_started = True
        total_transferred = sum(p.numel() for p in self.parameters())
        return total_transferred

    def freeze_backbone(self):
        """Freezes convolutional feature extractor for cold-start fine-tuning."""
        for p in [self.cnn0, self.cnn2, self.cnn4]:
            for param in p.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes convolutional layers for joint end-to-end training."""
        for p in self.parameters():
            p.requires_grad = True


class SpatialMapEncoder(nn.Module):
    """Encodes (B, 1, 48, 48) visited coordinate binary grid into 256-dim embedding."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WRAMVectorEncoder(nn.Module):
    """Encodes (B, 64) telemetry registers into 128-dim embedding."""
    def __init__(self, in_features: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WarmStartedMultiModalPolicy(nn.Module):
    """
    Critic-Free MultiModal Policy Network with Pretrained Whidden Visual Backbone.
    Features:
      - Visual stream warm-started from 439M steps
      - Spatial visited map encoder
      - WRAM telemetry encoder
      - LayerNorm balancing across all 3 modalities
      - Zero critic parameters (GRPO)
      - Dynamic action masking support
      - Shannon entropy computation for STAD variance injection
    """

    NUM_ACTIONS = 8

    def __init__(
        self,
        whidden_checkpoint_path: Optional[Path | str] = None,
        freeze_visual: bool = False,
    ):
        super().__init__()
        self.visual_enc = WarmStartedVisualEncoder(in_channels=3, out_dim=512)
        self.spatial_enc = SpatialMapEncoder(out_dim=256)
        self.wram_enc = WRAMVectorEncoder(in_features=64, out_dim=128)

        # Fusion: 512 + 256 + 128 = 896 -> 512 -> 8 logits
        self.fusion = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, self.NUM_ACTIONS),
        )

        if whidden_checkpoint_path is not None:
            self.visual_enc.load_whidden_weights(whidden_checkpoint_path)
            if freeze_visual:
                self.visual_enc.freeze_backbone()

    def forward(
        self,
        screen: torch.Tensor,       # (B, 3, 72, 80)
        spatial_map: torch.Tensor,  # (B, 1, 48, 48)
        wram: torch.Tensor,         # (B, 64)
        action_mask: Optional[torch.Tensor] = None, # (B, 8) bool, False = masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (action_probabilities, logits).
        """
        vis_feat = self.visual_enc(screen)       # (B, 512)
        spat_feat = self.spatial_enc(spatial_map) # (B, 256)
        wram_feat = self.wram_enc(wram)          # (B, 128)

        fused = torch.cat([vis_feat, spat_feat, wram_feat], dim=-1) # (B, 896)
        logits = self.fusion(fused)                                 # (B, 8)

        # Hardware action masking: mask invalid actions with large negative value
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def compute_stad_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes per-step Shannon entropy H(pi(.|s)) for STAD variance injection:
        H = - sum_a p(a) * log(p(a) + eps)
        """
        eps = 1e-8
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (B,)
        return entropy
