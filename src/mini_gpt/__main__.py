import torch
from torch import nn

from mini_gpt.token_and_position_embeddings import TokenAndPositionEmbeddings
from mini_gpt.transformer_block import TransformerBlock


def main() -> None:
    token_ids = torch.tensor([1, 0, 6, 5], dtype=torch.long)
    model_dim = 3
    model = nn.Sequential(
        TokenAndPositionEmbeddings(
            vocab_size=16, context_length=16, embedding_dim=model_dim
        ),
        TransformerBlock(model_dim=model_dim),
    )
    model.eval()
    with torch.inference_mode():
        hidden_states = model(token_ids)

    print("Input token IDs", token_ids.tolist())
    print("Hidden states with shape", tuple(hidden_states.shape))
    print(hidden_states)


if __name__ == "__main__":
    main()
