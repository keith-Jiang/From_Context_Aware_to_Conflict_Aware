"""Greedy decoding: simply use the context-conditioned logits."""

import torch
from methods.base import DecodingMethod


class GreedyDecoding(DecodingMethod):
    name = "greedy"

    def get_next_token_logits(self, logits_ctx, logits_prior):
        return logits_ctx

    def get_tau(self, logits_ctx, logits_prior):
        return 1.0
