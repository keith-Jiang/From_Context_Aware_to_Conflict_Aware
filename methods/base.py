"""Base class for decoding methods.

All methods receive logits from two forward passes:
  - logits_ctx:   logits conditioned on context + question
  - logits_prior: logits conditioned on question only (no context)

And produce a next-token distribution or directly the token id.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class DecodingMethod(ABC):
    """Abstract base for a single-step decoding method."""

    name: str = "base"

    @abstractmethod
    def get_next_token_logits(
        self,
        logits_ctx: torch.Tensor,
        logits_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Combine ctx and prior logits into adjusted logits.

        Args:
            logits_ctx:   [1, vocab_size] logits with context
            logits_prior: [1, vocab_size] logits without context

        Returns:
            adjusted logits [1, vocab_size]
        """
        ...

    def decode_greedy(
        self,
        logits_ctx: torch.Tensor,
        logits_prior: torch.Tensor,
    ) -> int:
        adjusted = self.get_next_token_logits(logits_ctx, logits_prior)
        return torch.argmax(adjusted, dim=-1).item()

    def get_tau(
        self,
        logits_ctx: torch.Tensor,
        logits_prior: torch.Tensor,
    ) -> float:
        """Return the effective τ value for this step (for analysis).

        Default returns None (methods that don't use τ).
        """
        return None
