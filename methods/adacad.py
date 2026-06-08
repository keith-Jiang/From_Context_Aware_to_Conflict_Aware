"""AdaCAD — Adaptive Context-Aware Decoding.

Uses Jensen-Shannon Divergence to adaptively set α per token.
In the unified power-family: τ_t = 1 + α_t^{JSD}  (extrapolation regime).

adjusted_logits = (1 + α_t) * logits_ctx - α_t * logits_prior
where α_t = JSD(p_ctx, p_prior)
"""

import torch
import torch.nn.functional as F
from methods.base import DecodingMethod


class AdaCADDecoding(DecodingMethod):
    name = "adacad"

    def __init__(self, warmup_beta: float = 0.0):
        self.warmup_beta = warmup_beta

    def _compute_jsd(self, logits_ctx, logits_prior):
        p_ctx = F.softmax(logits_ctx.float(), dim=-1)
        p_prior = F.softmax(logits_prior.float(), dim=-1)
        m = 0.5 * (p_ctx + p_prior)

        log_m = torch.log(m + 1e-12)
        kl_ctx = F.kl_div(log_m, p_ctx, reduction="batchmean", log_target=False)
        kl_prior = F.kl_div(log_m, p_prior, reduction="batchmean", log_target=False)
        jsd = 0.5 * (kl_ctx + kl_prior)
        return jsd.item()

    def get_next_token_logits(self, logits_ctx, logits_prior):
        alpha_t = self._compute_jsd(logits_ctx, logits_prior)
        alpha_t = max(alpha_t, self.warmup_beta)
        return (1 + alpha_t) * logits_ctx - alpha_t * logits_prior

    def get_tau(self, logits_ctx, logits_prior):
        alpha_t = self._compute_jsd(logits_ctx, logits_prior)
        alpha_t = max(alpha_t, self.warmup_beta)
        return 1 + alpha_t
