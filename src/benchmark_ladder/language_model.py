"""Held-out language-model evaluation primitives.

The public harness scores each item as one independent sequence. Item boundaries are therefore
part of benchmark semantics and must stay fixed across model comparisons. Bits-per-byte is
aggregated from total negative log-likelihood over total raw input bytes; per-sequence BPB values
are diagnostics only and are never averaged to form the corpus score.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from benchmark_ladder.adapters import ModelAdapter
from benchmark_ladder.scoring import bits_per_byte

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BitsPerByteScoringPolicy:
    """Versioned scorer semantics for held-out language-model likelihood.

    The scorer fixes independent record boundaries, natural-log probabilities, raw-byte
    denominator and sum-NLL-over-sum-bytes aggregation. A tiny positive log-probability tolerance
    permits floating-point overshoot around zero; accepted overshoots are clamped and counted.
    The tolerance is part of ``config_digest`` so changing it changes public provenance.
    """

    version: str
    positive_logprob_tolerance_nats: float = 1e-6

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if (
            not math.isfinite(self.positive_logprob_tolerance_nats)
            or self.positive_logprob_tolerance_nats < 0.0
        ):
            raise ValueError("positive_logprob_tolerance_nats must be finite and >= 0")

    @property
    def config_digest(self) -> str:
        payload = (
            "bits-per-byte-scoring-policy-v2\n"
            "sequence_boundary=independent_item\n"
            "log_probability_unit=natural_log\n"
            "denominator=raw_input_bytes\n"
            "aggregation=sum_nll_over_sum_bytes\n"
            f"positive_logprob_tolerance_nats={self.positive_logprob_tolerance_nats.hex()}\n"
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LanguageModelItem:
    """One independently scored held-out byte sequence."""

    example_id: str
    data: bytes

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not self.data:
            raise ValueError("language-model item data must be non-empty")


@dataclass(frozen=True, slots=True)
class LanguageModelObservation:
    """Private evidence retained for one held-out sequence."""

    example_id: str
    scorer_version: str
    scorer_config_digest: str
    byte_count: int
    sequence_logprob_nats: float
    positive_logprob_clamped: bool = False

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must be non-empty")
        if not self.scorer_version:
            raise ValueError("scorer_version must be non-empty")
        if not _SHA256_RE.fullmatch(self.scorer_config_digest):
            raise ValueError("scorer_config_digest must be sha256:<64 lowercase hex characters>")
        if self.byte_count <= 0:
            raise ValueError("byte_count must be > 0")
        if not math.isfinite(self.sequence_logprob_nats):
            raise ValueError("sequence_logprob_nats must be finite")
        if self.sequence_logprob_nats > 0.0:
            raise ValueError("sequence_logprob_nats must be <= 0 after tolerance handling")

    @property
    def negative_log_likelihood_nats(self) -> float:
        return -self.sequence_logprob_nats

    @property
    def bits_per_byte(self) -> float:
        return bits_per_byte(self.negative_log_likelihood_nats, self.byte_count)


def evaluate_language_model(
    adapter: ModelAdapter,
    items: tuple[LanguageModelItem, ...],
    policy: BitsPerByteScoringPolicy,
) -> tuple[LanguageModelObservation, ...]:
    """Score held-out sequences without changing benchmark-defined item boundaries."""

    observations: list[LanguageModelObservation] = []
    for item_index, item in enumerate(items):
        logprob = float(adapter.sequence_logprob(item.data))
        if not math.isfinite(logprob):
            raise ValueError(
                f"language-model item index {item_index}: sequence_logprob_nats must be finite"
            )
        if logprob > policy.positive_logprob_tolerance_nats:
            raise ValueError(
                f"language-model item index {item_index}: positive sequence_logprob_nats "
                "exceeds scorer tolerance"
            )
        was_clamped = logprob > 0.0
        if was_clamped:
            logprob = 0.0
        observations.append(
            LanguageModelObservation(
                example_id=item.example_id,
                scorer_version=policy.version,
                scorer_config_digest=policy.config_digest,
                byte_count=len(item.data),
                sequence_logprob_nats=logprob,
                positive_logprob_clamped=was_clamped,
            )
        )
    return tuple(observations)


def aggregate_language_model(
    observations: tuple[LanguageModelObservation, ...],
) -> dict[str, int | float]:
    """Aggregate corpus BPB from total NLL and total raw bytes."""

    if not observations:
        raise ValueError("cannot aggregate an empty observation set")
    scorer_versions = {observation.scorer_version for observation in observations}
    if len(scorer_versions) != 1:
        raise ValueError("cannot aggregate observations from different scorer versions")
    scorer_config_digests = {observation.scorer_config_digest for observation in observations}
    if len(scorer_config_digests) != 1:
        raise ValueError("cannot aggregate observations from different scorer configurations")

    total_bytes = sum(observation.byte_count for observation in observations)
    total_nll = math.fsum(observation.negative_log_likelihood_nats for observation in observations)
    corpus_bpb = bits_per_byte(total_nll, total_bytes)
    return {
        "count": len(observations),
        "byte_count": total_bytes,
        "negative_log_likelihood_nats": total_nll,
        "bits_per_byte": corpus_bpb,
        "positive_logprob_clamp_count": sum(
            observation.positive_logprob_clamped for observation in observations
        ),
    }
