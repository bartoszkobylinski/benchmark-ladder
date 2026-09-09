import pytest

from benchmark_ladder.runner import PairwiseItem, aggregate_pairwise, evaluate_pairwise
from tests.helpers import ScriptedPairwiseAdapter


def test_pairwise_runner_and_aggregation() -> None:
    adapter = ScriptedPairwiseAdapter({b"a": -1.0, b"b": -2.0, b"c": -5.0, b"d": -1.0})
    items = (
        PairwiseItem("one", b"ctx", (b"a", b"b"), 0),
        PairwiseItem("two", b"ctx", (b"c", b"d"), 0),
    )
    observations = evaluate_pairwise(adapter, items)
    metrics = aggregate_pairwise(observations)

    assert observations[0].margin == 1.0
    assert observations[0].correct is True
    assert observations[1].margin == -4.0
    assert observations[1].correct is False
    assert metrics["count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_margin"] == pytest.approx(-1.5)
    assert metrics["chance_normalized_accuracy"] == 0.0


def test_pairwise_item_rejects_invalid_gold_index() -> None:
    with pytest.raises(ValueError):
        PairwiseItem("bad", b"", (b"a", b"b"), 2)
