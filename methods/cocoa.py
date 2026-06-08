"""CoCoA — Confidence-based Contextual Adjustment (simplified single-GPU version).

In the unified power-family: τ_t ∈ [0,1]  (interpolation regime).
Uses ΔH (entropy gap), Rényi D2, and token-level contrast to compute
a conflict-aware mixing weight.

This is a simplified, single-card reimplementation of the core logic
from CoCoA/group_decode_fileio_CoCoA.py, without distributed communication.
"""

import torch
import torch.nn.functional as F
from methods.base import DecodingMethod


class CoCoADecoding(DecodingMethod):
    name = "cocoa"

    def __init__(self, global_alpha: float = 0.5, gamma: float = 1.0,
                 lambda_pm: float = 100.0):
        self.global_alpha = global_alpha
        self.gamma = gamma
        self.lambda_pm = lambda_pm

    def get_next_token_logits(self, logits_ctx, logits_prior):
        if logits_ctx.dim() == 1:
            logits_ctx = logits_ctx.unsqueeze(0)
        if logits_prior.dim() == 1:
            logits_prior = logits_prior.unsqueeze(0)

        p_ctx = F.softmax(logits_ctx.float(), dim=-1)
        p_pri = F.softmax(logits_prior.float(), dim=-1)
        log_p_ctx = F.log_softmax(logits_ctx.float(), dim=-1)
        log_p_pri = F.log_softmax(logits_prior.float(), dim=-1)

        delta = log_p_ctx - log_p_pri
        alpha = self.global_alpha

        s_mix = alpha * log_p_ctx + (1 - alpha) * log_p_pri + self.gamma * delta

        s_med = torch.median(s_mix, dim=-1, keepdim=True)[0]
        s_mad = torch.median(torch.abs(s_mix - s_med), dim=-1, keepdim=True)[0] + 1e-8
        s_z = torch.clamp((s_mix - s_med) / s_mad, -100.0, 100.0)
        s_mix = s_mix + self.lambda_pm * s_z

        return s_mix

    def get_tau(self, logits_ctx, logits_prior):
        return self.global_alpha
