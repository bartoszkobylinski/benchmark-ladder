"""Versioned public result contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias, cast

from benchmark_ladder.adapters import DecodingConfig, DecodingMode, GenerationUnit

SCHEMA_VERSION = 1
JsonScalar: TypeAlias = str | int | float | bool | None


class UnsupportedSchemaVersion(ValueError):
    pass


class ExecutionStatus(StrEnum):
    MEASURED = "measured"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    parameters: int
    architecture: str
    tokenizer: str

    def __post_init__(self) -> None:
        if self.parameters <= 0:
            raise ValueError("parameters must be > 0")
        if not self.architecture or not self.tokenizer:
            raise ValueError("architecture and tokenizer must be non-empty")


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    tokens: int
    checkpoint_step: int | None = None

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be >= 0")
        if self.checkpoint_step is not None and self.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be >= 0 when set")


@dataclass(frozen=True, slots=True)
class EvaluationMetadata:
    benchmark_id: str
    benchmark_version: str
    scorer_version: str
    taxonomy_version: str
    runner_git_sha: str
    seed: int
    calibration_rule_version: str | None = None
    reference_pool_id: str | None = None
    release_commitment: str | None = None

    def __post_init__(self) -> None:
        required = {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "scorer_version": self.scorer_version,
            "taxonomy_version": self.taxonomy_version,
            "runner_git_sha": self.runner_git_sha,
        }
        for name, value in required.items():
            if not value:
                raise ValueError(f"{name} must be non-empty")
        optional = {
            "calibration_rule_version": self.calibration_rule_version,
            "reference_pool_id": self.reference_pool_id,
            "release_commitment": self.release_commitment,
        }
        for name, value in optional.items():
            if value is not None and not value:
                raise ValueError(f"{name} must be non-empty when set")


@dataclass(frozen=True, slots=True)
class ExecutionMetadata:
    status: ExecutionStatus
    skip_reason: str | None = None
    decoding: DecodingConfig | None = None

    def __post_init__(self) -> None:
        if self.status is ExecutionStatus.SKIPPED and not self.skip_reason:
            raise ValueError("skipped execution requires skip_reason")
        if self.status is ExecutionStatus.MEASURED and self.skip_reason is not None:
            raise ValueError("measured execution must not carry skip_reason")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    model: ModelMetadata
    training: TrainingMetadata
    evaluation: EvaluationMetadata
    execution: ExecutionMetadata
    metrics: Mapping[str, JsonScalar] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(f"unsupported schema_version: {self.schema_version}")
        for key, value in self.metrics.items():
            if not key:
                raise ValueError("metric names must be non-empty")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metric {key!r} must be finite")

    def to_dict(self) -> dict[str, object]:
        decoding = self.execution.decoding
        decoding_data: dict[str, object] | None = None
        if decoding is not None:
            decoding_data = {
                "mode": decoding.mode.value,
                "max_new_units": decoding.max_new_units,
                "unit": decoding.unit.value,
                "temperature": decoding.temperature,
                "top_k": decoding.top_k,
                "top_p": decoding.top_p,
                "stop_policy_id": decoding.stop_policy_id,
                "seed": decoding.seed,
            }

        return {
            "schema_version": self.schema_version,
            "model": {
                "parameters": self.model.parameters,
                "architecture": self.model.architecture,
                "tokenizer": self.model.tokenizer,
            },
            "training": {
                "tokens": self.training.tokens,
                "checkpoint_step": self.training.checkpoint_step,
            },
            "evaluation": {
                "benchmark_id": self.evaluation.benchmark_id,
                "benchmark_version": self.evaluation.benchmark_version,
                "scorer_version": self.evaluation.scorer_version,
                "taxonomy_version": self.evaluation.taxonomy_version,
                "runner_git_sha": self.evaluation.runner_git_sha,
                "seed": self.evaluation.seed,
                "calibration_rule_version": self.evaluation.calibration_rule_version,
                "reference_pool_id": self.evaluation.reference_pool_id,
                "release_commitment": self.evaluation.release_commitment,
            },
            "execution": {
                "status": self.execution.status.value,
                "skip_reason": self.execution.skip_reason,
                "decoding": decoding_data,
            },
            "metrics": dict(self.metrics),
        }

    def canonical_json(self) -> str:
        """Stable JSON representation suitable for golden files and deterministic diffs."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvaluationResult:
        schema_version = _required_int(data, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(f"unsupported schema_version: {schema_version}")

        model_data = _required_mapping(data, "model")
        training_data = _required_mapping(data, "training")
        evaluation_data = _required_mapping(data, "evaluation")
        execution_data = _required_mapping(data, "execution")
        metrics_data = _optional_mapping(data, "metrics")

        decoding_raw = execution_data.get("decoding")
        decoding: DecodingConfig | None
        if decoding_raw is None:
            decoding = None
        else:
            decoding_data = _as_mapping(decoding_raw, "execution.decoding")
            decoding = DecodingConfig(
                mode=DecodingMode(_required_str(decoding_data, "mode")),
                max_new_units=_required_int(decoding_data, "max_new_units"),
                unit=GenerationUnit(_required_str(decoding_data, "unit")),
                temperature=_optional_float(decoding_data, "temperature"),
                top_k=_optional_int(decoding_data, "top_k"),
                top_p=_optional_float(decoding_data, "top_p"),
                stop_policy_id=_optional_str(decoding_data, "stop_policy_id"),
                seed=_optional_int(decoding_data, "seed"),
            )

        metrics: dict[str, JsonScalar] = {}
        for key, value in metrics_data.items():
            if not isinstance(key, str):
                raise TypeError("metric keys must be strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise TypeError(f"metric {key!r} must be a JSON scalar")
            metrics[key] = cast(JsonScalar, value)

        return cls(
            schema_version=schema_version,
            model=ModelMetadata(
                parameters=_required_int(model_data, "parameters"),
                architecture=_required_str(model_data, "architecture"),
                tokenizer=_required_str(model_data, "tokenizer"),
            ),
            training=TrainingMetadata(
                tokens=_required_int(training_data, "tokens"),
                checkpoint_step=_optional_int(training_data, "checkpoint_step"),
            ),
            evaluation=EvaluationMetadata(
                benchmark_id=_required_str(evaluation_data, "benchmark_id"),
                benchmark_version=_required_str(evaluation_data, "benchmark_version"),
                scorer_version=_required_str(evaluation_data, "scorer_version"),
                taxonomy_version=_required_str(evaluation_data, "taxonomy_version"),
                runner_git_sha=_required_str(evaluation_data, "runner_git_sha"),
                seed=_required_int(evaluation_data, "seed"),
                calibration_rule_version=_optional_str(
                    evaluation_data, "calibration_rule_version"
                ),
                reference_pool_id=_optional_str(evaluation_data, "reference_pool_id"),
                release_commitment=_optional_str(evaluation_data, "release_commitment"),
            ),
            execution=ExecutionMetadata(
                status=ExecutionStatus(_required_str(execution_data, "status")),
                skip_reason=_optional_str(execution_data, "skip_reason"),
                decoding=decoding,
            ),
            metrics=metrics,
        )


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in data:
        raise KeyError(key)
    return _as_mapping(data[key], key)


def _optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key, {})
    return _as_mapping(value, key)


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null")
    return value


def _optional_float(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric or null")
    return float(value)
