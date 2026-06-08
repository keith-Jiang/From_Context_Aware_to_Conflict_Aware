"""Context-Aware Decoding (CAD) — Shi et al., 2023.

In the unified power-family: τ = 1 + α  (extrapolation regime).

adjusted_logits = (1 + α) * logits_ctx - α * logits_prior
"""

import torch
from methods.base import DecodingMethod


class CADDecoding(DecodingMethod):
    name = "cad"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def get_next_token_logits(self, logits_ctx, logits_prior):
        return (1 + self.alpha) * logits_ctx - self.alpha * logits_prior

    def get_tau(self, logits_ctx, logits_prior):
        return 1 + self.alpha
