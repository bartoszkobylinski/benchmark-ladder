from __future__ import annotations

from collections.abc import Mapping

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit, ModelAdapter

NOT_CALLABLE = 7


class FixtureAdapter:
    def __init__(self, scores: Mapping[str, float]) -> None:
        self._scores = dict(scores)

    def sequence_logprob(self, data: bytes) -> float:
        return -float(len(data))

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return self._scores[continuation.decode("ascii")]

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        if config.unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        if config.max_new_units == 0:
            return b""
        return (prompt[-1:] or b"x") * config.max_new_units

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        if unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        return len(data)


def create_adapter(config: Mapping[str, object]) -> ModelAdapter:
    raw_scores = config.get("scores")
    if not isinstance(raw_scores, dict):
        raise TypeError("fixture config requires scores object")
    scores: dict[str, float] = {}
    for key, value in raw_scores.items():
        if not isinstance(key, str):
            raise TypeError("score keys must be strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("score values must be numeric")
        scores[key] = float(value)
    return FixtureAdapter(scores)
