"""ARR (Adaptive Regime Routing): confidence gate + JSD-based conflict strength."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from methods.base import DecodingMethod


class ARRDecoding(DecodingMethod):
    """ARR: confidence gate + JSD/log2 conflict strength."""

    name = "arr"

    def _stats(self, logits_ctx, logits_prior):
        if logits_ctx.dim() == 1:
            logits_ctx = logits_ctx.unsqueeze(0)
        if logits_prior.dim() == 1:
            logits_prior = logits_prior.unsqueeze(0)

        log_p_ctx = F.log_softmax(logits_ctx.float(), dim=-1)
        log_p_prior = F.log_softmax(logits_prior.float(), dim=-1)
        p_ctx = log_p_ctx.exp()
        p_prior = log_p_prior.exp()

        ctx_pmax, _ = p_ctx.max(dim=-1)
        prior_pmax, _ = p_prior.max(dim=-1)

        m = 0.5 * (p_ctx + p_prior)
        log_m = torch.log(m.clamp_min(1e-12))
        kl_ctx = (p_ctx * (log_p_ctx - log_m)).sum(dim=-1)
        kl_prior = (p_prior * (log_p_prior - log_m)).sum(dim=-1)
        jsd = 0.5 * (kl_ctx + kl_prior)
        strength = (jsd / math.log(2.0)).clamp(0.0, 1.0)

        return ctx_pmax, prior_pmax, strength

    def _compute_tau_tensor(self, logits_ctx, logits_prior):
        ctx_pmax, prior_pmax, strength = self._stats(logits_ctx, logits_prior)

        ctx_more_confident = ctx_pmax > prior_pmax
        extrapolate = ctx_more_confident

        tau = torch.where(extrapolate, 1.0 + strength, 1.0 - strength)
        return tau

    def get_next_token_logits(self, logits_ctx, logits_prior):
        tau = self._compute_tau_tensor(logits_ctx, logits_prior).unsqueeze(-1)
        return tau * logits_ctx + (1.0 - tau) * logits_prior

    def get_tau(self, logits_ctx, logits_prior):
        tau = self._compute_tau_tensor(logits_ctx, logits_prior)
        return tau.mean().item()
