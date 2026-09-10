"""Small, pure scoring functions intended for strong unit/property/mutation testing."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def pairwise_margin(correct_logprob: float, incorrect_logprob: float) -> float:
    _require_finite(correct_logprob, "correct_logprob")
    _require_finite(incorrect_logprob, "incorrect_logprob")
    return correct_logprob - incorrect_logprob


def accuracy(correct: Sequence[bool]) -> float:
    if not correct:
        raise ValueError("accuracy requires at least one observation")
    return sum(correct) / len(correct)


def chance_normalized_accuracy(accuracy_value: float, chance: float) -> float:
    """Map chance to 0 and perfect accuracy to 1, retaining below-chance negatives."""

    _require_finite(accuracy_value, "accuracy_value")
    _require_finite(chance, "chance")
    if not 0.0 <= accuracy_value <= 1.0:
        raise ValueError("accuracy_value must be in [0, 1]")
    if not 0.0 <= chance < 1.0:
        raise ValueError("chance must be in [0, 1)")
    return (accuracy_value - chance) / (1.0 - chance)


def bits_per_byte(total_negative_log_likelihood_nats: float, byte_count: int) -> float:
    """Convert total negative log-likelihood in nats to bits per byte."""

    _require_finite(total_negative_log_likelihood_nats, "total_negative_log_likelihood_nats")
    if total_negative_log_likelihood_nats < 0.0:
        raise ValueError("total_negative_log_likelihood_nats must be >= 0")
    if byte_count <= 0:
        raise ValueError("byte_count must be > 0")
    return total_negative_log_likelihood_nats / (math.log(2.0) * byte_count)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    for value in values:
        _require_finite(value, "value")
    return math.fsum(values) / len(values)
