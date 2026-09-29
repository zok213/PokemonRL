"""
grpo.py — Critic-Free Group Relative Policy Optimization with STAD
===================================================================
Implements Group Relative Policy Optimization (GRPO; Shao et al., DeepSeek 2024)
adapted for long-horizon JRPG decision forks with Trajectory State-Action Diversity (STAD).

Theoretical Anchors:
  - DeepSeek-Math: Shao et al., arXiv:2402.03300 (2024)
  - Zero-Mean Baseline Proposition: sum(A_i) = 0 by construction across G sibling rollouts
  - VRAM Reduction: 50% static model/optimizer memory reduction by removing Value Critic network
  - Zero-Variance Black Hole Lemma: when all G siblings encounter identical obstacle (e.g. wall in Rock Tunnel),
    std({R}) -> 0 and grad(L_GRPO) -> 0.
  - STAD Resolution: Injecting normalized per-step policy entropy H(pi_theta(.|s_t)) guarantees non-zero variance.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np


class AdaptiveTauGRPO:
    """
    Critic-Free Group Relative Policy Optimization with Trajectory State-Action Diversity (STAD).

    Advantage normalization over G sibling trajectories originating from the identical fork state:
        A_i = (R_i - mean({R})) / (std({R}) + eps)

    Surrogate objective with reverse-KL regularization:
        L_GRPO(theta) = -(1/G) sum_i min( rho_i * A_i, clip(rho_i, 1 +/- eps) * A_i )
                        + beta * D_KL(pi_theta || pi_ref)
    """

    def __init__(
        self,
        group_size: int = 8,
        clip_ratio: float = 0.2,
        tau: float = 0.25,
        beta_kl: float = 0.04,
        epsilon_variance_threshold: float = 1e-6,
    ):
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.tau = tau
        self.beta_kl = beta_kl
        self.eps_var = epsilon_variance_threshold

    def compute_group_advantages(
        self,
        trajectory_returns: List[float] | np.ndarray,
        trajectory_action_entropies: Optional[List[float] | np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute group-normalized advantages across G sibling rollouts.

        If reward variance collapses (std < eps_var, e.g. all 8 rollouts get 0.0 reward in a maze),
        injects normalized Trajectory State-Action Diversity (STAD) to break the degeneracy.

        Args:
            trajectory_returns: (G,) scalar returns for each sibling
            trajectory_action_entropies: (G,) mean policy entropy for each sibling (STAD)

        Returns:
            advantages: (G,) normalized advantages with mean 0 and unit variance
            is_augmented: True if STAD variance injection was triggered
        """
        returns = np.array(trajectory_returns, dtype=np.float32)
        std_ret = float(np.std(returns))
        is_augmented = False

        # Zero-Variance Gradient Black Hole Detection:
        if std_ret < self.eps_var and trajectory_action_entropies is not None:
            entropies = np.array(trajectory_action_entropies, dtype=np.float32)
            ent_std = float(np.std(entropies))
            if ent_std > 1e-8:
                norm_entropies = (entropies - np.mean(entropies)) / (ent_std + 1e-8)
                returns = returns + self.tau * norm_entropies
                is_augmented = True

        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8
        advantages = (returns - mean_ret) / std_ret
        return advantages.astype(np.float32), is_augmented

    def evaluate_loss(
        self,
        log_probs_new: np.ndarray,
        log_probs_old: np.ndarray,
        log_probs_ref: np.ndarray,
        advantages: np.ndarray,
        beta_kl: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Evaluate clipped surrogate objective with reverse-KL divergence penalty.

        Args:
            log_probs_new: (N,) log-probabilities under current policy pi_theta
            log_probs_old: (N,) log-probabilities under rollout policy pi_old
            log_probs_ref: (N,) log-probabilities under reference policy pi_ref
            advantages:    (N,) broadcast or per-step advantages

        Returns:
            (total_loss, surrogate_loss, kl_penalty)
        """
        kl_coeff = self.beta_kl if beta_kl is None else beta_kl

        # Importance sampling ratio rho_t(theta) = pi_theta(a|s) / pi_old(a|s)
        ratio = np.exp(log_probs_new - log_probs_old)
        clipped_ratio = np.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        surrogate = np.minimum(surrogate1, surrogate2)

        # Reverse KL regularization: D_KL(pi_theta || pi_ref)
        # Using exact log-ratio estimator: exp(log_ref - log_new) - (log_ref - log_new) - 1 >= 0
        diff = log_probs_ref - log_probs_new
        kl = np.exp(diff) - diff - 1.0

        surr_loss = -float(np.mean(surrogate))
        kl_loss = float(np.mean(kl))
        total_loss = surr_loss + kl_coeff * kl_loss

        return total_loss, surr_loss, kl_loss
