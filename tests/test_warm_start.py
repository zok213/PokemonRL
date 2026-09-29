"""
test_warm_start.py — Unit Tests for Feature Warm-Start Option 1
===============================================================
Verifies:
  1. 100% tensor parity between Whidden's 439M checkpoint and WarmStartedVisualEncoder
  2. MultiModalPolicyNetwork forward shapes: (B, 8) probabilities on the simplex
  3. Action masking suppression via wJoyIgnore
  4. STAD entropy computation strict positivity
  5. Discriminative parameter freezing and unfreezing
"""

import io
from pathlib import Path
import zipfile
import pytest
import torch

from phases.phase3_neuro_symbolic_upgrades.warm_started_policy_network import (
    WarmStartedVisualEncoder,
    WarmStartedMultiModalPolicy,
)


@pytest.fixture
def checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    path = (
        repo_root
        / "external"
        / "PokemonRedExperiments"
        / "baselines"
        / "session_4da05e87_main_good"
        / "poke_439746560_steps.zip"
    )
    assert path.exists(), f"Pretrained checkpoint not found at {path}"
    return path


def test_warm_start_weights_parity(checkpoint_path: Path):
    encoder = WarmStartedVisualEncoder()
    transferred = encoder.load_whidden_weights(checkpoint_path)

    assert encoder.is_warm_started is True
    assert transferred > 0

    # Extract raw checkpoint directly to verify bitwise tensor equality
    with zipfile.ZipFile(checkpoint_path, "r") as z:
        with z.open("policy.pth") as f:
            raw_sd = torch.load(io.BytesIO(f.read()), map_location="cpu")

    # Verify layer 0 conv weights match exactly
    assert torch.equal(encoder.cnn0.weight.data, raw_sd["features_extractor.cnn.0.weight"])
    assert torch.equal(encoder.cnn0.bias.data, raw_sd["features_extractor.cnn.0.bias"])
    # Verify layer 2 conv weights match exactly
    assert torch.equal(encoder.cnn2.weight.data, raw_sd["features_extractor.cnn.2.weight"])
    assert torch.equal(encoder.cnn2.bias.data, raw_sd["features_extractor.cnn.2.bias"])
    # Verify layer 4 conv weights match exactly
    assert torch.equal(encoder.cnn4.weight.data, raw_sd["features_extractor.cnn.4.weight"])
    assert torch.equal(encoder.cnn4.bias.data, raw_sd["features_extractor.cnn.4.bias"])
    # Verify linear projection weights match exactly
    assert torch.equal(encoder.linear.weight.data, raw_sd["features_extractor.linear.0.weight"])
    assert torch.equal(encoder.linear.bias.data, raw_sd["features_extractor.linear.0.bias"])


def test_warm_started_policy_forward_and_masking(checkpoint_path: Path):
    policy = WarmStartedMultiModalPolicy(whidden_checkpoint_path=checkpoint_path, freeze_visual=True)
    B = 4
    screen = torch.randn(B, 3, 72, 80)
    spatial_map = torch.randn(B, 1, 48, 48)
    wram = torch.randn(B, 64)

    # Action mask: suppress START (button index 6)
    mask = torch.ones(B, 8, dtype=torch.bool)
    mask[:, 6] = False

    probs, logits = policy(screen, spatial_map, wram, action_mask=mask)

    assert probs.shape == (B, 8)
    assert logits.shape == (B, 8)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B), atol=1e-5)
    # Masked action probability must be effectively zero (< 1e-6)
    assert torch.all(probs[:, 6] < 1e-6)


def test_stad_entropy_strict_positivity(checkpoint_path: Path):
    policy = WarmStartedMultiModalPolicy(whidden_checkpoint_path=checkpoint_path)
    B = 8
    screen = torch.randn(B, 3, 72, 80)
    spatial_map = torch.randn(B, 1, 48, 48)
    wram = torch.randn(B, 64)

    probs, _ = policy(screen, spatial_map, wram)
    entropy = policy.compute_stad_entropy(probs)

    assert entropy.shape == (B,)
    assert torch.all(entropy > 0.0)


def test_freeze_and_unfreeze_schedule(checkpoint_path: Path):
    policy = WarmStartedMultiModalPolicy(whidden_checkpoint_path=checkpoint_path, freeze_visual=True)

    # In frozen mode: cnn0, cnn2, cnn4 have requires_grad = False
    assert policy.visual_enc.cnn0.weight.requires_grad is False
    assert policy.visual_enc.cnn2.weight.requires_grad is False
    assert policy.visual_enc.cnn4.weight.requires_grad is False
    # Head and other encoders remain trainable
    assert policy.fusion[0].weight.requires_grad is True

    # Unfreeze
    policy.visual_enc.unfreeze_backbone()
    assert policy.visual_enc.cnn0.weight.requires_grad is True
    assert policy.visual_enc.cnn2.weight.requires_grad is True
    assert policy.visual_enc.cnn4.weight.requires_grad is True
