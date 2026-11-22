from pydantic import (
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    validate_call,
)
from torch import Tensor, nn

from mini_gpt.token_and_position_embeddings import TokenAndPositionEmbeddings
from mini_gpt.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """Single-head GPT with defaults based on minGPT's Tiny Shakespeare example."""

    @validate_call(config=ConfigDict(strict=True, validate_default=True))
    def __init__(
        self,
        vocab_size: PositiveInt = 65,
        context_length: PositiveInt = 128,
        embedding_dim: PositiveInt = 192,
        num_layers: NonNegativeInt = 6,
    ) -> None:
        super().__init__()

        self.embeddings = TokenAndPositionEmbeddings(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(embedding_dim) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        y = self.embeddings(tokens)
        for block in self.transformer_blocks:
            y = block(y)
        y = self.norm(y)
        return self.lm_head(y)
