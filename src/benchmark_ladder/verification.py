"""Reusable public contract checks for adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit, ModelAdapter, PositionwiseOracle


class ContractViolation(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class ContinuationCase:
    context: bytes
    continuation: bytes


def _assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ContractViolation(f"{label}: values must be finite")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ContractViolation(f"{label}: {actual} != {expected} within {tolerance}")


def verify_byte_continuation_semantics(
    adapter: ModelAdapter,
    oracle: PositionwiseOracle,
    cases: tuple[ContinuationCase, ...],
    *,
    tolerance: float = 1e-9,
) -> None:
    """Verify byte continuation semantics against an independent position-wise oracle."""

    if not cases:
        raise ValueError("at least one continuation case is required")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and >= 0")

    for index, case in enumerate(cases):
        joint = case.context + case.continuation
        joint_targets = oracle.target_logprobs(joint)
        context_targets = oracle.target_logprobs(case.context)
        if len(joint_targets) != len(joint):
            raise ContractViolation(f"case {index}: oracle returned wrong joint target count")
        if len(context_targets) != len(case.context):
            raise ContractViolation(f"case {index}: oracle returned wrong context target count")

        oracle_joint = math.fsum(joint_targets)
        oracle_context = math.fsum(context_targets)
        oracle_conditional = math.fsum(joint_targets[len(case.context) :])

        sequence_joint = adapter.sequence_logprob(joint)
        sequence_context = adapter.sequence_logprob(case.context)
        conditional = adapter.continuation_logprob(case.context, case.continuation)

        _assert_close(sequence_joint, oracle_joint, tolerance, f"case {index} joint sequence")
        _assert_close(sequence_context, oracle_context, tolerance, f"case {index} context sequence")
        _assert_close(conditional, oracle_conditional, tolerance, f"case {index} conditional")
        _assert_close(
            sequence_joint,
            sequence_context + conditional,
            tolerance,
            f"case {index} decomposition",
        )


def verify_generation_contract(
    adapter: ModelAdapter, prompt: bytes, config: DecodingConfig
) -> bytes:
    """Verify generic generation-budget semantics and return the generated continuation."""

    generated = adapter.generate(prompt, config)
    if not isinstance(generated, bytes):
        raise ContractViolation("generate must return bytes")
    unit_count = adapter.count_units(generated, config.unit)
    if unit_count < 0:
        raise ContractViolation("count_units returned a negative count")
    if unit_count > config.max_new_units:
        raise ContractViolation("generate exceeded max_new_units")
    if config.unit is GenerationUnit.BYTE and unit_count != len(generated):
        raise ContractViolation("byte unit count must equal len(generated)")
    if config.max_new_units == 0 and generated:
        raise ContractViolation("zero generation budget must return an empty continuation")
    return generated
