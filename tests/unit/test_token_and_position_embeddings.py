import pytest
import torch

from mini_gpt.token_and_position_embeddings import TokenAndPositionEmbeddings


@pytest.fixture
def embeddings_with_known_weights() -> TokenAndPositionEmbeddings:
    embeddings = TokenAndPositionEmbeddings(
        vocab_size=3, context_length=4, embedding_dim=2
    )
    token_weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    position_weights = torch.tensor(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]
    )
    with torch.no_grad():
        embeddings.token_embedding.weight.copy_(token_weights)
        embeddings.position_embedding.weight.copy_(position_weights)
    return embeddings


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_embeddings_add_token_and_position_vectors(
    embeddings_with_known_weights: TokenAndPositionEmbeddings, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    embeddings = embeddings_with_known_weights.to(device)
    tokens = torch.tensor([[2, 0, 2], [1, 2, 0]], device=device)

    output = embeddings(tokens)

    expected = torch.tensor(
        [
            [[15.0, 26.0], [31.0, 42.0], [55.0, 66.0]],
            [[13.0, 24.0], [35.0, 46.0], [51.0, 62.0]],
        ],
        device=device,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("token_ids", "expected_values"),
    [
        pytest.param([2], [[15.0, 26.0]], id="single-token"),
        pytest.param(
            [0, 1, 0, 2],
            [[11.0, 22.0], [33.0, 44.0], [51.0, 62.0], [75.0, 86.0]],
            id="full-context",
        ),
    ],
)
@torch.no_grad()
def test_embeddings_accept_unbatched_sequences(
    embeddings_with_known_weights: TokenAndPositionEmbeddings,
    token_ids: list[int],
    expected_values: list[list[float]],
) -> None:
    tokens = torch.tensor(token_ids)

    output = embeddings_with_known_weights(tokens)

    expected = torch.tensor(expected_values)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_embeddings_remain_usable_after_rejecting_an_overlong_sequence(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    embeddings = TokenAndPositionEmbeddings(
        vocab_size=3, context_length=2, embedding_dim=2
    ).to(device)
    valid_sequence = torch.tensor([0, 1], device=device)
    overlong_sequence = torch.tensor([0, 1, 2], device=device)
    expected_output = embeddings(valid_sequence)

    with pytest.raises(IndexError, match="Sequence length 3 exceeds context length 2"):
        embeddings(overlong_sequence)

    output = embeddings(valid_sequence)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.parametrize("invalid_token_id", [-1, 3], ids=["negative", "above-vocab"])
def test_embeddings_reject_tokens_outside_the_vocabulary(
    invalid_token_id: int,
) -> None:
    embeddings = TokenAndPositionEmbeddings(
        vocab_size=3, context_length=4, embedding_dim=2
    )
    tokens = torch.tensor([0, invalid_token_id])

    with pytest.raises(IndexError):
        embeddings(tokens)


def test_embeddings_accumulate_gradients_for_shared_tokens_and_positions() -> None:
    embeddings = TokenAndPositionEmbeddings(
        vocab_size=4, context_length=4, embedding_dim=2
    )
    tokens = torch.tensor([[2, 0, 2], [1, 2, 0]])
    output_gradient = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    )

    output = embeddings(tokens)
    weights = (embeddings.token_embedding.weight, embeddings.position_embedding.weight)
    token_gradient, position_gradient = torch.autograd.grad(
        output, weights, grad_outputs=output_gradient
    )

    expected_token_gradient = torch.tensor(
        [[14.0, 16.0], [7.0, 8.0], [15.0, 18.0], [0.0, 0.0]]
    )
    expected_position_gradient = torch.tensor(
        [[8.0, 10.0], [12.0, 14.0], [16.0, 18.0], [0.0, 0.0]]
    )
    torch.testing.assert_close(token_gradient, expected_token_gradient)
    torch.testing.assert_close(position_gradient, expected_position_gradient)
