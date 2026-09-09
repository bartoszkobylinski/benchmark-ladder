"""Reusable public contract checks for adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark_ladder.adapters import (
    DecodingConfig,
    DecodingMode,
    GenerationUnit,
    ModelAdapter,
    PositionwiseOracle,
)


class ContractViolation(AssertionError):
    pass


@dataclass(frozen=True, slots=True)
class ContinuationCase:
    context: bytes
    continuation: bytes


def _assert_close(
    actual: float,
    expected: float,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    label: str,
) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ContractViolation(f"{label}: values must be finite")
    if not math.isclose(
        actual,
        expected,
        rel_tol=rel_tolerance,
        abs_tol=abs_tolerance,
    ):
        raise ContractViolation(
            f"{label}: {actual} != {expected} within abs={abs_tolerance}, rel={rel_tolerance}"
        )


def verify_byte_continuation_semantics(
    adapter: ModelAdapter,
    oracle: PositionwiseOracle,
    cases: tuple[ContinuationCase, ...],
    *,
    abs_tolerance: float = 1e-9,
    rel_tolerance: float = 1e-7,
) -> None:
    """Verify byte continuation semantics against an independent position-wise oracle."""

    if not cases:
        raise ValueError("at least one continuation case is required")
    for name, value in (("abs_tolerance", abs_tolerance), ("rel_tolerance", rel_tolerance)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")

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

        close_args = {
            "abs_tolerance": abs_tolerance,
            "rel_tolerance": rel_tolerance,
        }
        _assert_close(
            sequence_joint, oracle_joint, label=f"case {index} joint sequence", **close_args
        )
        _assert_close(
            sequence_context,
            oracle_context,
            label=f"case {index} context sequence",
            **close_args,
        )
        _assert_close(
            conditional, oracle_conditional, label=f"case {index} conditional", **close_args
        )
        _assert_close(
            sequence_joint,
            sequence_context + conditional,
            label=f"case {index} decomposition",
            **close_args,
        )


def verify_generation_contract(
    adapter: ModelAdapter, prompt: bytes, config: DecodingConfig
) -> bytes:
    """Verify generic generation-budget semantics and return the generated continuation.

    This black-box check verifies budget semantics and greedy repeatability. An adapter-specific
    conformance test must still establish that the backend wrapper returns continuation bytes
    only rather than copying the prompt into the returned value.
    """

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

    if config.mode is DecodingMode.GREEDY:
        repeated = adapter.generate(prompt, config)
        if repeated != generated:
            raise ContractViolation("greedy generation must be repeatable for identical inputs")

    return generated
