import pytest
import torch

from mini_gpt.demo import rank_next_token_candidates


def test_next_token_candidates_rank_last_position_across_the_vocabulary() -> None:
    token_weights = torch.tensor([[8.0, 1.0, 1.0], [1.0, 2.0, 7.0]])
    logits = token_weights.log()

    candidates = rank_next_token_candidates(logits, count=2)

    token_ids, probabilities = zip(*candidates, strict=True)
    assert token_ids == (2, 1)
    assert probabilities == pytest.approx((0.7, 0.2))
