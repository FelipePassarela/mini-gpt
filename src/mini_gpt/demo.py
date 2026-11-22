import torch
from torch import Tensor

from mini_gpt.mini_gpt import MiniGPT


def demo() -> None:
    token_ids = torch.tensor([1, 0, 6, 5], dtype=torch.long)
    model = MiniGPT().eval()
    with torch.inference_mode():
        logits = model(token_ids)
        candidates = rank_next_token_candidates(logits, count=5)
    _print_predictions(token_ids, logits.shape, candidates)


def rank_next_token_candidates(
    logits: Tensor, *, count: int
) -> list[tuple[int, float]]:
    next_token_logits = logits[-1]
    probabilities = next_token_logits.softmax(dim=-1)
    top_probabilities, top_token_ids = probabilities.topk(count)
    return list(zip(top_token_ids.tolist(), top_probabilities.tolist(), strict=True))


def _print_predictions(
    token_ids: Tensor, logits_shape: torch.Size, candidates: list[tuple[int, float]]
) -> None:
    print(f"Prompt token IDs {token_ids.tolist()}")
    print(f"Logits shape (sequence length, vocabulary size) {tuple(logits_shape)}")
    print(f"\nTop {len(candidates)} next-token candidates (randomly initialized model)")
    print(f"{'Token ID':>8}  {'Probability':>11}")
    for token_id, probability in candidates:
        print(f"{token_id:>8}  {probability:>11.2%}")
