from __future__ import annotations

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit


def _adapter_step_logprob(previous: int, current: int) -> float:
    # Production-fake path used by the adapter. Keep separate from the oracle implementation.
    return -((((previous * 31) + current) % 17) + 1) / 10.0


class IndependentByteOracle:
    def target_logprobs(self, data: bytes) -> list[float]:
        # Independent reference path: entry i is log P(data[i] | data[:i]).
        output: list[float] = []
        previous = 0
        for current in data:
            bucket = ((31 * previous + current) % 17) + 1
            output.append(-(bucket / 10.0))
            previous = current
        return output


class DeterministicByteAdapter:
    sequence_start_semantics = "test-byte-prior-zero-v1"

    def sequence_logprob(self, data: bytes) -> float:
        total = 0.0
        previous = 0
        for current in data:
            total += _adapter_step_logprob(previous, current)
            previous = current
        return total

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        total = 0.0
        previous = context[-1] if context else 0
        for current in continuation:
            total += _adapter_step_logprob(previous, current)
            previous = current
        return total

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
        total = 0.0
        previous = context[-1] if context else 0
        for current in continuation[1:]:
            total += _adapter_step_logprob(previous, current)
            previous = current
        return total


class ContextDroppingAdapter(DeterministicByteAdapter):
    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        total = 0.0
        previous = 0
        for current in continuation:
            total += _adapter_step_logprob(previous, current)
            previous = current
        return total


class ScriptedPairwiseAdapter(DeterministicByteAdapter):
    def __init__(self, scores: dict[bytes, float]) -> None:
        self._scores = scores

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return self._scores[continuation]
