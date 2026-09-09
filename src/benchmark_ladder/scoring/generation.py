"""Explicit output-normalization primitives for generation-based scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExactMatchPolicy:
    """Deliberately small public normalization policy.

    Hidden tasks may use different private scorers, but any such policy must be versioned and
    tested. Keeping these transforms explicit prevents accidental whitespace/case behaviour.
    """

    version: str
    strip_outer_ascii_whitespace: bool = False
    ascii_case_insensitive: bool = False

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")


def normalize_output(data: bytes, policy: ExactMatchPolicy) -> bytes:
    normalized = data
    if policy.strip_outer_ascii_whitespace:
        normalized = normalized.strip(b" \t\r\n\v\f")
    if policy.ascii_case_insensitive:
        normalized = normalized.lower()
    return normalized


def exact_match(generated: bytes, expected: bytes, policy: ExactMatchPolicy) -> bool:
    return normalize_output(generated, policy) == normalize_output(expected, policy)
