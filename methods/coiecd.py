"""COIECD — Conflict-of-Interest Entropy Constrained Decoding.

Piecewise branch-level: uses entropy-based conflict detection to decide
whether to interpolate (conflict branch, τ=α) or extrapolate (non-conflict
branch, τ=1+α) on a per-token basis.

In the unified power-family:
  - Conflict tokens:     τ = α ∈ (0,1)   — interpolation regime
  - Non-conflict tokens: τ = 1+α > 1     — extrapolation regime
"""

import torch
import torch.nn.functional as F
import numpy as np
from methods.base import DecodingMethod


class COIECDDecoding(DecodingMethod):
    name = "coiecd"

    def __init__(self, alpha: float = 1.0, threshold_ratio: float = 4):
        self.alpha = alpha
        self.threshold_ratio = threshold_ratio

    def _cal_constraint_bounds(self, logits_cond, logits_uncond):
        """Reproduce the conflict detection from COIECD/decoding.py."""
        normalized = F.log_softmax(logits_uncond, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        normalized_cond = F.log_softmax(logits_cond, dim=-1)
        shifted_scores = (-normalized_cond) - ent

        scores_normalized = shifted_scores.log_softmax(dim=-1)
        probs_min = torch.min(scores_normalized, dim=-1).values
        probs_thresh = probs_min + np.log(self.threshold_ratio)
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_filter = probs_max - np.log(self.threshold_ratio)
        probs_filter = probs_filter.unsqueeze(-1)
        mask_filter = scores_normalized > probs_filter

        probs_thresh = probs_thresh.unsqueeze(-1)
        mask = scores_normalized >= probs_thresh
        count_mask = scores_normalized < probs_thresh
        if count_mask.sum() == 1:
            mask = torch.ones(logits_cond.shape[-1], dtype=torch.bool,
                              device=logits_cond.device).unsqueeze(0)

        return mask, mask_filter

    def get_next_token_logits(self, logits_ctx, logits_prior):
        logits_cond = logits_ctx
        logits_uncond = logits_prior
        logits_diff = logits_cond - logits_uncond

        typical_mask, mask_filter = self._cal_constraint_bounds(logits_ctx, logits_prior)
        constraint = torch.ones_like(logits_diff)
        alpha_list = torch.ones_like(logits_diff) * self.alpha
        constraint[typical_mask] = 0.0
        constraint[mask_filter] = 1.0
        inv_constraint = 1 - constraint

        merged = (constraint * logits_cond + inv_constraint * logits_uncond
                  + logits_diff * alpha_list)
        return merged

    def get_tau(self, logits_ctx, logits_prior):
        typical_mask, _ = self._cal_constraint_bounds(logits_ctx, logits_prior)
        conflict_ratio = typical_mask.float().mean().item()
        if conflict_ratio > 0.5:
            return self.alpha
        return 1 + self.alpha
