"""
test_grpo.py — Unit Tests for Critic-Free Adaptive Tau-GRPO
===========================================================
Verifies zero-mean group advantage, finite-sample variance bound, and STAD resolution.
"""

import pytest
import numpy as np
from pokemon_rl.systems.grpo import AdaptiveTauGRPO


def test_grpo_advantage_zero_mean():
    """Theorem / Proposition: Sibling normalized advantages sum to zero identically."""
    grpo = AdaptiveTauGRPO(group_size=8)
    returns = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    adv, is_aug = grpo.compute_group_advantages(returns)

    assert not is_aug
    assert adv.shape == (8,)
    assert abs(float(np.mean(adv))) < 1e-5
    assert abs(float(np.std(adv)) - 1.0) < 1e-4


def test_zero_variance_black_hole_stad_resolution():
    """
    Lemma: When all 8 rollouts stall at identical obstacle (R_i = 0.0 for all i),
    STAD entropy injection guarantees non-zero advantage variance.
    """
    grpo = AdaptiveTauGRPO(group_size=8, tau=0.25)
    flat_returns = [0.0] * 8
    sibling_entropies = [1.12, 1.25, 0.98, 1.45, 1.02, 1.30, 1.18, 1.22]

    adv, is_aug = grpo.compute_group_advantages(flat_returns, sibling_entropies)

    assert is_aug is True
    assert float(np.std(adv)) > 0.95
    assert abs(float(np.mean(adv))) < 1e-5


def test_clipped_surrogate_loss():
    grpo = AdaptiveTauGRPO(group_size=8, clip_ratio=0.2, beta_kl=0.04)
    N = 16
    log_probs_new = np.zeros(N, dtype=np.float32)
    log_probs_old = np.zeros(N, dtype=np.float32)
    log_probs_ref = np.zeros(N, dtype=np.float32)
    advantages = np.ones(N, dtype=np.float32)

    total_loss, surr_loss, kl_loss = grpo.evaluate_loss(
        log_probs_new, log_probs_old, log_probs_ref, advantages
    )
    # Ratio = 1.0, surrogate = 1.0, loss = -1.0, KL = 0
    assert abs(surr_loss - (-1.0)) < 1e-5
    assert abs(kl_loss - 0.0) < 1e-5
    assert abs(total_loss - (-1.0)) < 1e-5
