"""End-to-end orchestration for public evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_ladder.adapters import ModelAdapter
from benchmark_ladder.results import (
    EvaluationResult,
    ExecutionMetadata,
    ExecutionStatus,
    ModelMetadata,
    TrainingMetadata,
    evaluation_metadata_from_components,
)
from benchmark_ladder.runner import (
    PairwiseItem,
    PairwiseObservation,
    PairwiseScoringPolicy,
    aggregate_pairwise,
    evaluate_pairwise,
)
from benchmark_ladder.taskio import canonical_pairwise_items_digest


@dataclass(frozen=True, slots=True)
class PairwiseEvaluationRequest:
    """Inputs needed to execute one measured pairwise evaluation."""

    adapter: ModelAdapter
    items: tuple[PairwiseItem, ...]
    scoring_policy: PairwiseScoringPolicy
    model: ModelMetadata
    training: TrainingMetadata
    benchmark_id: str
    benchmark_version: str
    taxonomy_version: str
    runner_git_sha: str
    release_commitment: str
    reference_pool_id: str | None = None


def run_pairwise_evaluation(
    request: PairwiseEvaluationRequest,
) -> tuple[EvaluationResult, tuple[PairwiseObservation, ...]]:
    """Run inference, aggregate metrics, and bind them to public provenance."""

    task_items_digest = canonical_pairwise_items_digest(request.items)
    observations = evaluate_pairwise(
        request.adapter,
        request.items,
        request.scoring_policy,
    )
    metrics = aggregate_pairwise(observations)
    evaluation = evaluation_metadata_from_components(
        benchmark_id=request.benchmark_id,
        benchmark_version=request.benchmark_version,
        scorer=request.scoring_policy,
        taxonomy_version=request.taxonomy_version,
        runner_git_sha=request.runner_git_sha,
        reference_pool_id=request.reference_pool_id,
        release_commitment=request.release_commitment,
        task_items_digest=task_items_digest,
        scorer_config_digest=request.scoring_policy.config_digest,
    )
    result = EvaluationResult(
        model=request.model,
        training=request.training,
        evaluation=evaluation,
        execution=ExecutionMetadata(status=ExecutionStatus.MEASURED),
        metrics=metrics,
    )
    return result, observations
