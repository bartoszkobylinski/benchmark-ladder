"""End-to-end orchestration for public evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

from benchmark_ladder.adapters import ModelAdapter
from benchmark_ladder.language_model import (
    BitsPerByteScoringPolicy,
    LanguageModelItem,
    LanguageModelObservation,
    aggregate_language_model,
    evaluate_language_model,
)
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


@dataclass(frozen=True, slots=True)
class LanguageModelEvaluationRequest:
    """Inputs needed to execute one measured held-out BPB evaluation."""

    adapter: ModelAdapter
    items: tuple[LanguageModelItem, ...]
    scoring_policy: BitsPerByteScoringPolicy
    model: ModelMetadata
    training: TrainingMetadata
    benchmark_id: str
    benchmark_version: str
    taxonomy_version: str
    runner_git_sha: str
    release_commitment: str
    reference_pool_id: str | None = None


def _bind_sequence_start_semantics(adapter: ModelAdapter, model: ModelMetadata) -> ModelMetadata:
    """Bind adapter-owned sequence-start semantics into public model provenance."""

    sequence_start_semantics = adapter.sequence_start_semantics
    if not isinstance(sequence_start_semantics, str) or not sequence_start_semantics.strip():
        raise ValueError("adapter sequence_start_semantics must be a non-empty string")
    if (
        model.sequence_start_semantics is not None
        and model.sequence_start_semantics != sequence_start_semantics
    ):
        raise ValueError("model provenance conflicts with adapter sequence_start_semantics")
    return replace(model, sequence_start_semantics=sequence_start_semantics)


def run_pairwise_evaluation(
    request: PairwiseEvaluationRequest,
) -> tuple[EvaluationResult, tuple[PairwiseObservation, ...]]:
    """Run inference, aggregate metrics, and bind them to public provenance."""

    model = _bind_sequence_start_semantics(request.adapter, request.model)
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
        scorer_config_digest=request.scoring_policy.config_digest,
    )
    result = EvaluationResult(
        model=model,
        training=request.training,
        evaluation=evaluation,
        execution=ExecutionMetadata(status=ExecutionStatus.MEASURED),
        metrics=metrics,
    )
    return result, observations


def run_language_model_evaluation(
    request: LanguageModelEvaluationRequest,
) -> tuple[EvaluationResult, tuple[LanguageModelObservation, ...]]:
    """Run held-out likelihood scoring and bind corpus BPB to public provenance."""

    model = _bind_sequence_start_semantics(request.adapter, request.model)
    observations = evaluate_language_model(
        request.adapter,
        request.items,
        request.scoring_policy,
    )
    metrics = aggregate_language_model(observations)
    evaluation = evaluation_metadata_from_components(
        benchmark_id=request.benchmark_id,
        benchmark_version=request.benchmark_version,
        scorer=request.scoring_policy,
        taxonomy_version=request.taxonomy_version,
        runner_git_sha=request.runner_git_sha,
        reference_pool_id=request.reference_pool_id,
        release_commitment=request.release_commitment,
        scorer_config_digest=request.scoring_policy.config_digest,
    )
    result = EvaluationResult(
        model=model,
        training=request.training,
        evaluation=evaluation,
        execution=ExecutionMetadata(status=ExecutionStatus.MEASURED),
        metrics=metrics,
    )
    return result, observations
