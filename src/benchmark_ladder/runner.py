"""Small public task runner primitives using synthetic-safe interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_ladder.adapters import ModelAdapter
from benchmark_ladder.scoring import chance_normalized_accuracy, mean, pairwise_margin


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
        if any(not candidate for candidate in self.candidates):
            raise ValueError("pairwise candidates must be non-empty")


@dataclass(frozen=True, slots=True)
class PairwiseObservation:
    example_id: str
    candidate_logprobs: tuple[float, float]
    candidate_byte_lengths: tuple[int, int]
    gold_index: int
    margin: float
    correct: bool | None
    byte_length_normalized_margin: float
    byte_length_normalized_correct: bool | None


def _decision_from_margin(margin: float) -> bool | None:
    if margin > 0.0:
        return True
    if margin < 0.0:
        return False
    return None


def _decision_score(decision: bool | None) -> float:
    if decision is None:
        return 0.5
    return 1.0 if decision else 0.0


def evaluate_pairwise(
    adapter: ModelAdapter, items: tuple[PairwiseItem, ...]
) -> tuple[PairwiseObservation, ...]:
    observations: list[PairwiseObservation] = []
    for item in items:
        scores = (
            adapter.continuation_logprob(item.context, item.candidates[0]),
            adapter.continuation_logprob(item.context, item.candidates[1]),
        )
        byte_lengths = (len(item.candidates[0]), len(item.candidates[1]))
        normalized_scores = (
            scores[0] / byte_lengths[0],
            scores[1] / byte_lengths[1],
        )
        other_index = 1 - item.gold_index
        margin = pairwise_margin(scores[item.gold_index], scores[other_index])
        normalized_margin = pairwise_margin(
            normalized_scores[item.gold_index], normalized_scores[other_index]
        )
        observations.append(
            PairwiseObservation(
                example_id=item.example_id,
                candidate_logprobs=scores,
                candidate_byte_lengths=byte_lengths,
                gold_index=item.gold_index,
                margin=margin,
                correct=_decision_from_margin(margin),
                byte_length_normalized_margin=normalized_margin,
                byte_length_normalized_correct=_decision_from_margin(normalized_margin),
            )
        )
    return tuple(observations)


def aggregate_pairwise(
    observations: tuple[PairwiseObservation, ...], chance: float = 0.5
) -> dict[str, int | float]:
    if not observations:
        raise ValueError("cannot aggregate an empty observation set")

    raw_scores = tuple(_decision_score(observation.correct) for observation in observations)
    normalized_scores = tuple(
        _decision_score(observation.byte_length_normalized_correct) for observation in observations
    )
    raw_accuracy = mean(raw_scores)
    normalized_accuracy = mean(normalized_scores)

    return {
        "count": len(observations),
        "tie_count": sum(observation.correct is None for observation in observations),
        "accuracy": raw_accuracy,
        "mean_margin": mean(tuple(observation.margin for observation in observations)),
        "chance_normalized_accuracy": chance_normalized_accuracy(raw_accuracy, chance),
        "byte_length_normalized_tie_count": sum(
            observation.byte_length_normalized_correct is None for observation in observations
        ),
        "byte_length_normalized_accuracy": normalized_accuracy,
        "mean_byte_length_normalized_margin": mean(
            tuple(observation.byte_length_normalized_margin for observation in observations)
        ),
        "chance_normalized_byte_length_accuracy": chance_normalized_accuracy(
            normalized_accuracy, chance
        ),
    }
