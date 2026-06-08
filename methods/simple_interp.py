"""Simple Interpolation — fixed-coefficient interpolation decoding.

The interpolation counterpart of CAD: instead of extrapolation τ = 1+α,
uses τ = α ∈ [0,1] as a fixed interpolation coefficient.

    q_τ(y) ∝ p_prior(y)^{1-α} · p_ctx(y)^α

In logit space:
    adjusted_logits = α * logits_ctx + (1-α) * logits_prior
"""

from methods.base import DecodingMethod


class SimpleInterpDecoding(DecodingMethod):
    name = "simple_interp"

    def __init__(self, alpha: float = 0.75):
        self.alpha = alpha

    def get_next_token_logits(self, logits_ctx, logits_prior):
        return self.alpha * logits_ctx + (1 - self.alpha) * logits_prior

    def get_tau(self, logits_ctx, logits_prior):
        return self.alpha
