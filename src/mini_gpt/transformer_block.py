from torch import Tensor, nn

from mini_gpt.single_head_attention import SingleHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = SingleHeadAttention(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x + self.attention(self.norm1(x))
        y = y + self.mlp(self.norm2(y))
        return y
