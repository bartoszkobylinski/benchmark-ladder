import math

import pytest

from benchmark_ladder.scoring import (
    accuracy,
    bits_per_byte,
    chance_normalized_accuracy,
    mean,
    pairwise_margin,
)


def test_pairwise_margin() -> None:
    assert pairwise_margin(-2.0, -5.0) == 3.0


def test_accuracy() -> None:
    assert accuracy((True, False, True, True)) == 0.75
    with pytest.raises(ValueError):
        accuracy(())


def test_chance_normalized_accuracy_pins_scale() -> None:
    assert chance_normalized_accuracy(0.5, 0.5) == 0.0
    assert chance_normalized_accuracy(1.0, 0.5) == 1.0
    assert chance_normalized_accuracy(0.25, 0.5) == -0.5
    assert chance_normalized_accuracy(0.625, 0.5) == 0.25
    assert chance_normalized_accuracy(0.75, 0.5) == 0.5
    assert chance_normalized_accuracy(0.875, 0.5) == 0.75


def test_bits_per_byte() -> None:
    nll = 8 * math.log(2.0)
    assert bits_per_byte(nll, 8) == pytest.approx(1.0)


def test_mean_uses_finite_values() -> None:
    assert mean((1.0, 2.0, 3.0)) == 2.0
    with pytest.raises(ValueError):
        mean((1.0, math.inf))
