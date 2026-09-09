"""Model-independent adapter contracts.

The public harness defines semantics, not model implementations. Hidden tasks and concrete
production adapters live behind the trusted evaluation boundary described by ADR-0002.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class GenerationUnit(StrEnum):
    """Unit used to enforce a generation budget."""

    BYTE = "byte"
    TOKEN = "token"
    MODEL_UNIT = "model_unit"


class DecodingMode(StrEnum):
    GREEDY = "greedy"
    SAMPLE = "sample"


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    """Versionable decoding inputs that materially affect a generation result."""

    mode: DecodingMode
    max_new_units: int
    unit: GenerationUnit
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    stop_policy_id: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_units < 0:
            raise ValueError("max_new_units must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0 when set")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.stop_policy_id is not None and not self.stop_policy_id:
            raise ValueError("stop_policy_id must be non-empty when set")

        if self.mode is DecodingMode.GREEDY:
            if any(value is not None for value in (self.temperature, self.top_k, self.top_p, self.seed)):
                raise ValueError("greedy decoding must not carry sampling parameters")
        else:
            if self.temperature is None or self.temperature <= 0.0:
                raise ValueError("sampling requires temperature > 0")
            if self.seed is None:
                raise ValueError("sampling requires an explicit seed")


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal interface exposed to public task/scoring machinery.

    ``generate`` returns only the newly generated continuation, not ``prompt + continuation``.
    ``count_units`` makes the unit behind ``max_new_units`` explicit and testable.
    """

    def sequence_logprob(self, data: bytes) -> float:
        """Return log P(data), under the adapter's documented start-of-sequence semantics."""

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        """Return log P(continuation | context)."""

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        """Return generated continuation bytes under ``config``."""

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        """Count generation-budget units represented by ``data``."""


@runtime_checkable
class PositionwiseOracle(Protocol):
    """Independent white-box oracle for byte-adapter continuation verification.

    The returned sequence has one log-probability per target byte in ``data``. Entry ``i`` is
    the log-probability assigned to ``data[i]`` given the prefix ``data[:i]`` under the
    oracle's documented start-of-sequence semantics.

    Implementations used as verification oracles must be independent of production
    ``sequence_logprob`` and ``continuation_logprob`` methods.
    """

    def target_logprobs(self, data: bytes) -> list[float]:
        """Return one independently computed target log-probability per byte."""
