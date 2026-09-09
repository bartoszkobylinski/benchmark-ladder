import json

import pytest

from benchmark_ladder.adapters import DecodingConfig, DecodingMode, GenerationUnit
from benchmark_ladder.results import (
    EvaluationMetadata,
    EvaluationResult,
    ExecutionMetadata,
    ExecutionStatus,
    ModelMetadata,
    TrainingMetadata,
    UnsupportedSchemaVersion,
)

COMMITMENT = "sha256:" + ("a" * 64)


def make_result() -> EvaluationResult:
    return EvaluationResult(
        model=ModelMetadata(parameters=8_160_256, architecture="toy-transformer", tokenizer="byte"),
        training=TrainingMetadata(tokens=31_334_400, checkpoint_step=7250),
        evaluation=EvaluationMetadata(
            benchmark_id="toy-public-task",
            benchmark_version="0-test",
            scorer_version="1",
            taxonomy_version="1",
            runner_git_sha="deadbeef",
            seed=None,
            calibration_rule_version="toy-rule-1",
            reference_pool_id="toy-pool-1",
            release_commitment=COMMITMENT,
        ),
        execution=ExecutionMetadata(
            status=ExecutionStatus.MEASURED,
            decoding=DecodingConfig(
                mode=DecodingMode.GREEDY,
                max_new_units=32,
                unit=GenerationUnit.BYTE,
                stop_policy_id="toy-stop-policy-1",
                tie_break_policy_id="lowest-byte-v1",
            ),
        ),
        metrics={"accuracy": 0.75, "count": 4},
    )


def test_result_round_trip_and_canonical_json() -> None:
    result = make_result()
    payload = result.to_dict()
    assert EvaluationResult.from_dict(payload) == result
    assert result.canonical_json() == EvaluationResult.from_dict(payload).canonical_json()
    assert json.loads(result.canonical_json())["evaluation"]["release_commitment"] == COMMITMENT


def test_future_schema_is_rejected() -> None:
    payload = make_result().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(UnsupportedSchemaVersion):
        EvaluationResult.from_dict(payload)


def test_skipped_execution_requires_reason_and_rule_version() -> None:
    with pytest.raises(ValueError):
        ExecutionMetadata(status=ExecutionStatus.SKIPPED)
    with pytest.raises(ValueError):
        ExecutionMetadata(status=ExecutionStatus.SKIPPED, skip_reason="budget")


def test_skipped_result_rejects_metrics() -> None:
    with pytest.raises(ValueError):
        EvaluationResult(
            model=ModelMetadata(parameters=1, architecture="toy", tokenizer="byte"),
            training=TrainingMetadata(tokens=0),
            evaluation=EvaluationMetadata(
                benchmark_id="toy",
                benchmark_version="0-test",
                scorer_version="1",
                taxonomy_version="1",
                runner_git_sha="deadbeef",
            ),
            execution=ExecutionMetadata(
                status=ExecutionStatus.SKIPPED,
                skip_reason="budget",
                skip_rule_version="budget-rule-v1",
            ),
            metrics={"accuracy": 0.0},
        )


def test_release_commitment_shape_is_validated() -> None:
    with pytest.raises(ValueError):
        EvaluationMetadata(
            benchmark_id="toy",
            benchmark_version="0-test",
            scorer_version="1",
            taxonomy_version="1",
            runner_git_sha="deadbeef",
            release_commitment="sha256:synthetic-only",
        )


def test_sampling_requires_seed_and_temperature() -> None:
    with pytest.raises(ValueError):
        DecodingConfig(
            mode=DecodingMode.SAMPLE,
            max_new_units=8,
            unit=GenerationUnit.BYTE,
            temperature=0.8,
        )
