import torch

from mini_gpt.transformer_block import TransformerBlock


@torch.no_grad()
def test_transformer_block_applies_pre_normalized_residual_connections() -> None:
    block = TransformerBlock(model_dim=3)
    tokens = torch.tensor([[1.0, -1.0, 2.0], [0.0, 2.0, -3.0]])

    attention_output = block.attention(block.norm1(tokens))
    after_attention = tokens + attention_output
    mlp_output = block.mlp(block.norm2(after_attention))
    expected = after_attention + mlp_output

    output = block(tokens.clone())

    torch.testing.assert_close(output, expected)


@torch.no_grad()
def test_transformer_block_preserves_prefix_outputs_when_tokens_are_appended() -> None:
    block = TransformerBlock(model_dim=3)
    prefix = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    suffix = torch.tensor([[3.0, -2.0, 0.0], [-1.0, 0.0, 4.0]])
    full_sequence = torch.cat((prefix, suffix))

    prefix_output = block(prefix)
    full_output = block(full_sequence)

    torch.testing.assert_close(full_output[: len(prefix)], prefix_output)


@torch.no_grad()
def test_transformer_block_processes_batch_sequences_independently() -> None:
    block = TransformerBlock(model_dim=4)
    sequences = torch.tensor(
        [
            [[1.0, 0.0, -1.0, 2.0], [0.0, 2.0, 1.0, -1.0], [3.0, -2.0, 0.0, 1.0]],
            [[0.0, -1.0, 2.0, 1.0], [2.0, 1.0, -1.0, 0.0], [1.0, 3.0, 0.0, -2.0]],
        ]
    )

    batch_output = block(sequences)

    individual_outputs = torch.stack([block(sequence) for sequence in sequences])
    torch.testing.assert_close(batch_output, individual_outputs)


@torch.no_grad()
def test_transformer_block_does_not_modify_input_tokens() -> None:
    block = TransformerBlock(model_dim=3)
    tokens = torch.tensor([[1.0, -1.0, 2.0], [0.0, 2.0, -3.0]])
    original_tokens = tokens.clone()

    block(tokens)

    torch.testing.assert_close(tokens, original_tokens, rtol=0, atol=0)


def test_transformer_block_backpropagates_to_its_parameters() -> None:
    block = TransformerBlock(model_dim=3)
    tokens = torch.tensor([[1.0, -1.0, 2.0], [0.0, 2.0, -3.0]])

    output = block(tokens)
    loss = output.square().sum()
    loss.backward()

    for name, parameter in block.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(gradient).all(), f"{name} has non-finite gradients"


def test_transformer_block_passes_input_gradients_through_residuals() -> None:
    block = TransformerBlock(model_dim=3)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()

    tokens = torch.tensor([[1.0, -1.0, 2.0], [0.0, 2.0, -3.0]], requires_grad=True)

    output = block(tokens)
    loss = output.square().sum()
    (input_gradient,) = torch.autograd.grad(loss, tokens)

    expected_gradient = 2 * tokens.detach()
    torch.testing.assert_close(input_gradient, expected_gradient)
