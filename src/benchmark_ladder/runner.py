"""Small public task runner primitives using synthetic-safe interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_ladder.adapters import ModelAdapter
from benchmark_ladder.scoring import accuracy, chance_normalized_accuracy, mean, pairwise_margin


@dataclass(frozen=True, slots=True)
class PairwiseItem:
    example_id: str
    context: bytes
    candidates: tuple[bytes, bytes]
    gold_index: int

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if self.gold_index not in (0, 1):
            raise ValueError("gold_index must be 0 or 1")


@dataclass(frozen=True, slots=True)
class PairwiseObservation:
    example_id: str
    candidate_logprobs: tuple[float, float]
    gold_index: int
    margin: float
    correct: bool


def evaluate_pairwise(
    adapter: ModelAdapter, items: tuple[PairwiseItem, ...]
) -> tuple[PairwiseObservation, ...]:
    observations: list[PairwiseObservation] = []
    for item in items:
        scores = (
            adapter.continuation_logprob(item.context, item.candidates[0]),
            adapter.continuation_logprob(item.context, item.candidates[1]),
        )
        other_index = 1 - item.gold_index
        margin = pairwise_margin(scores[item.gold_index], scores[other_index])
        observations.append(
            PairwiseObservation(
                example_id=item.example_id,
                candidate_logprobs=scores,
                gold_index=item.gold_index,
                margin=margin,
                correct=margin > 0.0,
            )
        )
    return tuple(observations)


def aggregate_pairwise(
    observations: tuple[PairwiseObservation, ...], chance: float = 0.5
) -> dict[str, int | float]:
    if not observations:
        raise ValueError("cannot aggregate an empty observation set")
    acc = accuracy(tuple(observation.correct for observation in observations))
    return {
        "count": len(observations),
        "accuracy": acc,
        "mean_margin": mean(tuple(observation.margin for observation in observations)),
        "chance_normalized_accuracy": chance_normalized_accuracy(acc, chance),
    }
