from __future__ import annotations

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit


def byte_target_logprob(value: int) -> float:
    return -((value % 17) + 1) / 10.0


class IndependentByteOracle:
    def target_logprobs(self, data: bytes) -> list[float]:
        return [byte_target_logprob(value) for value in data]


class DeterministicByteAdapter:
    def sequence_logprob(self, data: bytes) -> float:
        return sum(byte_target_logprob(value) for value in data)

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return sum(byte_target_logprob(value) for value in continuation)

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        if config.unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        if config.max_new_units == 0:
            return b""
        value = prompt[-1:] or b"x"
        return value * config.max_new_units

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        if unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        return len(data)


class BrokenOffsetAdapter(DeterministicByteAdapter):
    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return sum(byte_target_logprob(value) for value in continuation[1:])


class ScriptedPairwiseAdapter(DeterministicByteAdapter):
    def __init__(self, scores: dict[bytes, float]) -> None:
        self._scores = scores

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return self._scores[continuation]
