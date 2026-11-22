from math import exp, sqrt

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from mini_gpt.single_head_attention import SingleHeadAttention


def test_two_token_causal_attention_matches_hand_calculated_output() -> None:
    model_dim = 3
    attention = SingleHeadAttention(model_dim=model_dim)
    identity = torch.eye(model_dim)
    tokens = identity[:2]
    with torch.no_grad():
        for projection in (attention.w_q, attention.w_k, attention.w_v):
            projection.weight.copy_(identity)
            projection.bias.zero_()

    output = attention(tokens)

    matching_weight = exp(1 / sqrt(model_dim))
    total_weight = 1.0 + matching_weight
    previous_token_weight = 1.0 / total_weight
    current_token_weight = matching_weight / total_weight
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [previous_token_weight, current_token_weight, 0.0],
        ]
    )
    torch.testing.assert_close(output, expected)


def test_single_token_returns_its_value_projection() -> None:
    attention = SingleHeadAttention(model_dim=2)
    token = torch.tensor([[2.0, -1.0]])
    with torch.no_grad():
        attention.w_v.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        attention.w_v.bias.copy_(torch.tensor([0.5, -0.5]))

    output = attention(token)

    expected = torch.tensor([[0.5, 1.5]])
    torch.testing.assert_close(output, expected)


def test_appending_tokens_preserves_prefix_outputs() -> None:
    attention = SingleHeadAttention(model_dim=2)
    prefix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    suffix = torch.tensor([[3.0, 1.0], [-2.0, 4.0]])
    full_sequence = torch.cat((prefix, suffix))

    prefix_output = attention(prefix)
    full_output = attention(full_sequence)

    torch.testing.assert_close(full_output[: len(prefix)], prefix_output)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_attention_matches_pytorch_causal_attention(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    attention = SingleHeadAttention(model_dim=2).to(device)
    sequences = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]],
            [[0.0, -1.0], [2.0, 1.0], [1.0, 3.0]],
        ],
        device=device,
    )

    output = attention(sequences)

    head_axis = 1
    queries = attention.w_q(sequences).unsqueeze(head_axis)
    keys = attention.w_k(sequences).unsqueeze(head_axis)
    values = attention.w_v(sequences).unsqueeze(head_axis)
    reference = scaled_dot_product_attention(queries, keys, values, is_causal=True)
    expected = reference.squeeze(head_axis)
    torch.testing.assert_close(output, expected)


def test_attention_backpropagates_to_input_and_parameters() -> None:
    attention = SingleHeadAttention(model_dim=2)
    tokens = torch.tensor([[1.0, -1.0], [0.0, 2.0]], requires_grad=True)
    input_and_parameters = (tokens, *attention.parameters())

    output = attention(tokens)
    loss = output.square().sum()
    gradients = torch.autograd.grad(loss, input_and_parameters)

    for gradient in gradients:
        assert torch.isfinite(gradient).all()
