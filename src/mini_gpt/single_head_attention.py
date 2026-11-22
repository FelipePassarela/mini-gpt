from math import sqrt

import torch
from torch import Tensor, nn


class SingleHeadAttention(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()

        self._model_dim = model_dim
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        queries: Tensor = self.w_q(x)
        keys: Tensor = self.w_k(x)
        values: Tensor = self.w_v(x)

        scores = queries @ keys.mT / sqrt(self.model_dim)
        masked_scores = self._mask_future_tokens(scores)
        attention_weights = torch.softmax(masked_scores, dim=-1)

        return attention_weights @ values

    @staticmethod
    def _mask_future_tokens(scores: Tensor) -> Tensor:
        sequence_length = scores.size(-1)
        mask_shape = (sequence_length, sequence_length)
        token_pairs = torch.ones(mask_shape, dtype=torch.bool, device=scores.device)
        future_tokens = token_pairs.triu_(diagonal=1)
        return scores.masked_fill(future_tokens, value=float("-inf"))

    @property
    def model_dim(self) -> int:
        return self._model_dim
