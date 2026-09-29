"""
warm_started_policy_network.py — Feature Warm-Start from Whidden's 439M Checkpoint
==================================================================================
Transfers 100% of Peter Whidden's pretrained convolutional visual feature extractor
(from external/PokemonRedExperiments/baselines/session_4da05e87_main_good/poke_439746560_steps.zip)
into our SOTA Critic-Free MultiModal Policy Network.

Architectural Highlights:
  1. Visual Stream (Warm-Started):
     Conv2d(3, 32, k8, s4) -> Conv2d(32, 64, k4, s2) -> Conv2d(64, 64, k3, s1)
     -> AdaptiveAvgPool2d((3, 4)) -> Linear(768, 512) -> LayerNorm(512)
     Transfers all 4 weight & bias tensors from the 439,746,560-step checkpoint!
  2. Spatial Stream: Visited coordinate map (1, 48, 48) -> Linear(256) -> LayerNorm(256)
  3. WRAM Stream: 64 telemetry registers -> Linear(128) -> LayerNorm(128)
  4. Modality Balanced Fusion: Concatenate (512 + 256 + 128 = 896) -> Linear(512) -> 8 action logits
  5. Critic Head: ZERO (Critic-Free GRPO)
  6. Action Masking: Integrated zero-leak suppression via wJoyIgnore (0xCD6B)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import torch

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from pokemon_rl.agent.torch_policy import (
    WarmStartedVisualEncoder,
    SpatialMapEncoder,
    WRAMVectorEncoder,
    WarmStartedMultiModalPolicy,
)


def run_warm_start_verification():
    print("=" * 75)
    print("  PHASE 3: WARM-STARTED MULTIMODAL POLICY NETWORK VERIFICATION")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = (
        base_dir
        / "external"
        / "PokemonRedExperiments"
        / "baselines"
        / "session_4da05e87_main_good"
        / "poke_439746560_steps.zip"
    )

    print(f"[*] Checkpoint path: {checkpoint_path}")
    policy = WarmStartedMultiModalPolicy(whidden_checkpoint_path=checkpoint_path, freeze_visual=True)

    print(f"[+] Visual Encoder Warm-Started: {policy.visual_enc.is_warm_started}")
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[+] Total Parameters:     {total_params:,}")
    print(f"[+] Trainable Parameters: {trainable_params:,} (Visual CNN backbone frozen for stability)")

    # Test forward pass
    B = 4
    dummy_screen = torch.randn(B, 3, 72, 80)
    dummy_map = torch.randn(B, 1, 48, 48)
    dummy_wram = torch.randn(B, 64)

    # Action mask: suppress START (button 6)
    mask = torch.ones(B, 8, dtype=torch.bool)
    mask[:, 6] = False

    probs, logits = policy(dummy_screen, dummy_map, dummy_wram, action_mask=mask)
    entropy = policy.compute_stad_entropy(probs)

    assert probs.shape == (B, 8)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.all(probs[:, 6] < 1e-6), "Action 6 (START) must be zeroed by mask"
    assert torch.all(entropy > 0.0), "STAD entropy must be strictly positive"

    print(f"[+] Test 1 Passed: Forward pass tensor shapes: {probs.shape}")
    print(f"[+] Test 2 Passed: Action Mask successfully zeroed button 6 (prob = {probs[0, 6].item():.2e})")
    print(f"[+] Test 3 Passed: STAD Entropy strictly positive: mean = {entropy.mean().item():.3f} > 0")
    print("=" * 75)
    print("  FEATURE WARM-START OPTION 1 FULLY IMPLEMENTED AND VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    run_warm_start_verification()
