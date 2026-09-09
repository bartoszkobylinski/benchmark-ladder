import pytest

from benchmark_ladder.adapters import DecodingConfig, DecodingMode, GenerationUnit
from benchmark_ladder.verification import (
    ContinuationCase,
    ContractViolation,
    verify_byte_continuation_semantics,
    verify_generation_contract,
)
from tests.helpers import (
    BrokenOffsetAdapter,
    ContextDroppingAdapter,
    DeterministicByteAdapter,
    IndependentByteOracle,
)

CASES = (
    ContinuationCase(b"", b"a"),
    ContinuationCase(b"ctx", b"x"),
    ContinuationCase(b"white ", b" space"),
    ContinuationCase(b"\x00\xff", b"\x01\xfe"),
)


def test_byte_continuation_contract_passes_with_independent_oracle() -> None:
    verify_byte_continuation_semantics(
        DeterministicByteAdapter(),
        IndependentByteOracle(),
        CASES,
        abs_tolerance=1e-12,
        rel_tolerance=1e-12,
    )


def test_byte_continuation_contract_catches_offset_bug() -> None:
    with pytest.raises(ContractViolation):
        verify_byte_continuation_semantics(
            BrokenOffsetAdapter(),
            IndependentByteOracle(),
            CASES,
            abs_tolerance=1e-12,
            rel_tolerance=1e-12,
        )


def test_byte_continuation_contract_catches_context_drop() -> None:
    with pytest.raises(ContractViolation):
        verify_byte_continuation_semantics(
            ContextDroppingAdapter(),
            IndependentByteOracle(),
            CASES,
            abs_tolerance=1e-12,
            rel_tolerance=1e-12,
        )


def test_generation_contract_enforces_byte_budget() -> None:
    config = DecodingConfig(
        mode=DecodingMode.GREEDY,
        max_new_units=4,
        unit=GenerationUnit.BYTE,
        stop_policy_id="toy-stop-v1",
        tie_break_policy_id="lowest-byte-v1",
    )
    generated = verify_generation_contract(DeterministicByteAdapter(), b"z", config)
    assert generated == b"zzzz"


def test_generation_zero_budget_is_empty() -> None:
    config = DecodingConfig(
        mode=DecodingMode.GREEDY,
        max_new_units=0,
        unit=GenerationUnit.BYTE,
        tie_break_policy_id="lowest-byte-v1",
    )
    assert verify_generation_contract(DeterministicByteAdapter(), b"x", config) == b""
