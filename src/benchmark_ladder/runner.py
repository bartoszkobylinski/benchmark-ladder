"""Small public task runner primitives using synthetic-safe interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark_ladder.adapters import GenerationUnit, ModelAdapter
from benchmark_ladder.scoring import chance_normalized_accuracy, mean, pairwise_margin


@dataclass(frozen=True, slots=True)
class PairwiseScoringPolicy:
    """Versioned scorer semantics for binary discrimination tasks.

    ``tie_epsilon_per_unit`` defines the numerical tie band in log-probability units per
    normalization unit. Raw margins scale the band by the longer candidate; normalized margins
    use the per-unit value directly. Changing this value changes published scores and therefore
    requires a new ``version``.
    """

    version: str
    normalization_unit: GenerationUnit
    tie_epsilon_per_unit: float

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if not math.isfinite(self.tie_epsilon_per_unit) or self.tie_epsilon_per_unit < 0.0:
            raise ValueError("tie_epsilon_per_unit must be finite and >= 0")


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
    scorer_version: str
    normalization_unit: GenerationUnit
    candidate_logprobs: tuple[float, float]
    candidate_byte_lengths: tuple[int, int]
    candidate_unit_counts: tuple[int, int]
    gold_index: int
    margin: float
    tie_epsilon: float
    correct: bool | None
    unit_normalized_margin: float
    unit_normalized_tie_epsilon: float
    unit_normalized_correct: bool | None


def _decision_from_margin(margin: float, tie_epsilon: float) -> bool | None:
    if abs(margin) <= tie_epsilon:
        return None
    return margin > 0.0


def _decision_score(decision: bool | None) -> float:
    if decision is None:
        return 0.5
    return 1.0 if decision else 0.0


def evaluate_pairwise(
    adapter: ModelAdapter,
    items: tuple[PairwiseItem, ...],
    policy: PairwiseScoringPolicy,
) -> tuple[PairwiseObservation, ...]:
    observations: list[PairwiseObservation] = []
    for item in items:
        scores = (
            adapter.continuation_logprob(item.context, item.candidates[0]),
            adapter.continuation_logprob(item.context, item.candidates[1]),
        )
        byte_lengths = (len(item.candidates[0]), len(item.candidates[1]))
        unit_counts = (
            adapter.count_units(item.candidates[0], policy.normalization_unit),
            adapter.count_units(item.candidates[1], policy.normalization_unit),
        )
        if any(count <= 0 for count in unit_counts):
            raise ValueError("pairwise candidate unit counts must be > 0")

        normalized_scores = (
            scores[0] / unit_counts[0],
            scores[1] / unit_counts[1],
        )
        other_index = 1 - item.gold_index
        margin = pairwise_margin(scores[item.gold_index], scores[other_index])
        normalized_margin = pairwise_margin(
            normalized_scores[item.gold_index], normalized_scores[other_index]
        )
        tie_epsilon = policy.tie_epsilon_per_unit * max(unit_counts)
        normalized_tie_epsilon = policy.tie_epsilon_per_unit

        observations.append(
            PairwiseObservation(
                example_id=item.example_id,
                scorer_version=policy.version,
                normalization_unit=policy.normalization_unit,
                candidate_logprobs=scores,
                candidate_byte_lengths=byte_lengths,
                candidate_unit_counts=unit_counts,
                gold_index=item.gold_index,
                margin=margin,
                tie_epsilon=tie_epsilon,
                correct=_decision_from_margin(margin, tie_epsilon),
                unit_normalized_margin=normalized_margin,
                unit_normalized_tie_epsilon=normalized_tie_epsilon,
                unit_normalized_correct=_decision_from_margin(
                    normalized_margin, normalized_tie_epsilon
                ),
            )
        )
    return tuple(observations)


def aggregate_pairwise(
    observations: tuple[PairwiseObservation, ...], chance: float = 0.5
) -> dict[str, int | float]:
    if not observations:
        raise ValueError("cannot aggregate an empty observation set")

    scorer_versions = {observation.scorer_version for observation in observations}
    normalization_units = {observation.normalization_unit for observation in observations}
    if len(scorer_versions) != 1:
        raise ValueError("cannot aggregate observations from different scorer versions")
    if len(normalization_units) != 1:
        raise ValueError("cannot aggregate observations with different normalization units")

    raw_scores = tuple(_decision_score(observation.correct) for observation in observations)
    normalized_scores = tuple(
        _decision_score(observation.unit_normalized_correct) for observation in observations
    )
    raw_accuracy = mean(raw_scores)
    normalized_accuracy = mean(normalized_scores)

    return {
        "count": len(observations),
        "tie_count": sum(observation.correct is None for observation in observations),
        "accuracy": raw_accuracy,
        "mean_margin": mean(tuple(observation.margin for observation in observations)),
        "chance_normalized_accuracy": chance_normalized_accuracy(raw_accuracy, chance),
        "unit_normalized_tie_count": sum(
            observation.unit_normalized_correct is None for observation in observations
        ),
        "unit_normalized_accuracy": normalized_accuracy,
        "mean_unit_normalized_margin": mean(
            tuple(observation.unit_normalized_margin for observation in observations)
        ),
        "chance_normalized_unit_accuracy": chance_normalized_accuracy(
            normalized_accuracy, chance
        ),
    }
