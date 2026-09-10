"""Helpers that bind result provenance to the versioned objects used in a run."""

from __future__ import annotations

from typing import Protocol

from .schema import EvaluationMetadata


class VersionedComponent(Protocol):
    @property
    def version(self) -> str:
        """Stable identifier for the component behaviour used in a run."""
        ...


def evaluation_metadata_from_components(
    *,
    benchmark_id: str,
    benchmark_version: str,
    scorer: VersionedComponent,
    taxonomy_version: str,
    runner_git_sha: str,
    seed: int | None = None,
    calibration_rule: VersionedComponent | None = None,
    reference_pool_id: str | None = None,
    release_commitment: str | None = None,
    task_items_digest: str | None = None,
    scorer_config_digest: str | None = None,
) -> EvaluationMetadata:
    """Construct provenance without independently retyping component version strings."""

    return EvaluationMetadata(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        scorer_version=scorer.version,
        taxonomy_version=taxonomy_version,
        runner_git_sha=runner_git_sha,
        seed=seed,
        calibration_rule_version=(
            calibration_rule.version if calibration_rule is not None else None
        ),
        reference_pool_id=reference_pool_id,
        release_commitment=release_commitment,
        task_items_digest=task_items_digest,
        scorer_config_digest=scorer_config_digest,
    )
