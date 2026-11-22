import torch
from torch import Tensor, nn


class TokenAndPositionEmbeddings(nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, embedding_dim: int
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=context_length, embedding_dim=embedding_dim
        )

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[-1]
        self._validate_sequence_length(sequence_length)
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def _validate_sequence_length(self, sequence_length: int) -> None:
        context_length = self.position_embedding.num_embeddings
        if sequence_length > context_length:
            raise IndexError(
                f"Sequence length {sequence_length} exceeds context length {context_length}"
            )
