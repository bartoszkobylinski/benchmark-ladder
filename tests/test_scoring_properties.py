from hypothesis import given
from hypothesis import strategies as st

from benchmark_ladder.scoring import accuracy, chance_normalized_accuracy, pairwise_margin


@given(st.lists(st.booleans(), min_size=1, max_size=200))
def test_accuracy_is_order_invariant(values: list[bool]) -> None:
    assert accuracy(values) == accuracy(tuple(reversed(values)))


@given(
    st.floats(min_value=-1000, max_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000, max_value=0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_pairwise_margin_is_offset_invariant(a: float, b: float, offset: float) -> None:
    assert pairwise_margin(a + offset, b + offset) == pairwise_margin(a, b)


@given(
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_chance_normalized_accuracy_is_monotone(a: float, b: float) -> None:
    low, high = sorted((a, b))
    assert chance_normalized_accuracy(low, 0.5) <= chance_normalized_accuracy(high, 0.5)
