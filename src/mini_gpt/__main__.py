import torch

from mini_gpt.transformer_block import TransformerBlock


def main() -> None:
    token_vectors = torch.tensor(
        [
            [1, 1, 1],
            [0, 1, 2],
            [3, 7, 2],
            [0, 0, 0],
        ],
        dtype=torch.get_default_dtype(),
    )
    block = TransformerBlock(model_dim=3)
    hidden_states = block(token_vectors)
    print(hidden_states)


if __name__ == "__main__":
    main()
