from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.nn.functional import cross_entropy

from mini_gpt.mini_gpt import MiniGPT

VOCAB_SIZE = 7
CONTEXT_LENGTH = 5


def create_model(*, num_layers: int = 2) -> MiniGPT:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return MiniGPT(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            embedding_dim=4,
            num_layers=num_layers,
        )


@pytest.fixture
def inference_model() -> MiniGPT:
    return create_model().eval()


@torch.no_grad()
def test_mini_gpt_computes_logits_with_default_configuration() -> None:
    model = MiniGPT().eval()
    context_length = 128
    vocab_size = 65
    tokens = torch.arange(context_length) % vocab_size

    logits = model(tokens)

    assert logits.shape == (context_length, vocab_size)


@pytest.mark.parametrize("dimension", ["vocab_size", "context_length", "embedding_dim"])
@pytest.mark.parametrize("value", [0, -1])
def test_mini_gpt_rejects_non_positive_dimensions(dimension: str, value: int) -> None:
    with pytest.raises(ValidationError) as error:
        MiniGPT(**{dimension: value})

    (issue,) = error.value.errors()
    assert issue["loc"] == (dimension,)
    assert issue["type"] == "greater_than"
    assert issue["input"] == value


def test_mini_gpt_rejects_negative_layer_count() -> None:
    with pytest.raises(ValidationError) as error:
        MiniGPT(num_layers=-1)

    (issue,) = error.value.errors()
    assert issue["loc"] == ("num_layers",)
    assert issue["type"] == "greater_than_equal"
    assert issue["input"] == -1


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("4", id="string"),
        pytest.param(4.0, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param(np.int64(4), id="numpy-integer"),
    ],
)
def test_mini_gpt_requires_python_integers_for_dimensions(value: Any) -> None:
    with pytest.raises(ValidationError) as error:
        MiniGPT(context_length=value)

    (issue,) = error.value.errors()
    assert issue["loc"] == ("context_length",)
    assert issue["type"] == "int_type"


@pytest.mark.parametrize(
    "token_ids",
    [
        pytest.param([2], id="single-token"),
        pytest.param([0, 2, 6, 1, 5], id="full-context"),
        pytest.param([[0, 2, 6], [4, 1, 3]], id="batch"),
    ],
)
@torch.no_grad()
def test_mini_gpt_returns_vocabulary_logits_for_each_token(
    inference_model: MiniGPT, token_ids: list[int] | list[list[int]]
) -> None:
    tokens = torch.tensor(token_ids)

    logits = inference_model(tokens)

    assert logits.shape == (*tokens.shape, VOCAB_SIZE)


@pytest.mark.parametrize("num_layers", [0, 1, 3])
def test_mini_gpt_registers_the_requested_number_of_distinct_blocks(
    num_layers: int,
) -> None:
    model = create_model(num_layers=num_layers)

    blocks = tuple(model.transformer_blocks)
    distinct_blocks = set(blocks)
    registered_modules = set(model.modules())

    assert len(blocks) == num_layers
    assert len(distinct_blocks) == num_layers
    assert distinct_blocks <= registered_modules


@pytest.mark.parametrize("num_layers", [0, 1, 3])
@torch.no_grad()
def test_mini_gpt_applies_blocks_in_order_before_normalizing_and_projecting(
    num_layers: int,
) -> None:
    model = create_model(num_layers=num_layers).eval()
    tokens = torch.tensor([0, 2, 6])
    reference = torch.nn.Sequential(
        model.embeddings, *model.transformer_blocks, model.norm, model.lm_head
    )

    logits = model(tokens)

    expected_logits = reference(tokens)
    torch.testing.assert_close(logits, expected_logits)


@torch.no_grad()
def test_mini_gpt_preserves_prefix_logits_when_future_tokens_change(
    inference_model: MiniGPT,
) -> None:
    sequences = torch.tensor([[0, 2, 6, 1], [0, 2, 3, 5]])
    prefix_length = 2

    logits = inference_model(sequences)

    first_prefix_logits, second_prefix_logits = logits[:, :prefix_length]
    torch.testing.assert_close(first_prefix_logits, second_prefix_logits)


@torch.no_grad()
def test_mini_gpt_uses_previous_tokens_to_compute_current_logits(
    inference_model: MiniGPT,
) -> None:
    sequences = torch.tensor([[0, 2, 6], [4, 2, 6]])

    logits = inference_model(sequences)

    original_logits, modified_logits = logits[:, -1]
    assert not torch.allclose(original_logits, modified_logits)


@torch.no_grad()
def test_mini_gpt_processes_batch_sequences_independently(
    inference_model: MiniGPT,
) -> None:
    sequences = torch.tensor([[0, 2], [4, 1]])

    batch_logits = inference_model(sequences)

    individual_logits = torch.stack([inference_model(tokens) for tokens in sequences])
    torch.testing.assert_close(batch_logits, individual_logits)


@torch.no_grad()
def test_mini_gpt_rejects_sequences_longer_than_its_context(
    inference_model: MiniGPT,
) -> None:
    tokens = torch.zeros(CONTEXT_LENGTH + 1, dtype=torch.long)

    with pytest.raises(IndexError, match="exceeds context length"):
        inference_model(tokens)


def test_mini_gpt_backpropagates_next_token_loss_to_all_parameters() -> None:
    model = create_model()
    sequence = torch.tensor([0, 2, 6, 3])
    tokens = sequence[:-1]
    next_tokens = sequence[1:]

    logits = model(tokens)
    loss = cross_entropy(logits, next_tokens)
    loss.backward()

    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(gradient).all(), f"{name} has non-finite gradients"
