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
    assert observations[0].candidate_byte_lengths == (1, 1)
    assert metrics["count"] == 2
    assert metrics["tie_count"] == 0
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_margin"] == pytest.approx(-1.5)
    assert metrics["chance_normalized_accuracy"] == 0.0


def test_uniform_tie_scores_at_chance_not_maximally_wrong() -> None:
    adapter = ScriptedPairwiseAdapter({b"a": -3.0, b"b": -3.0})
    items = tuple(PairwiseItem(f"i{index}", b"ctx", (b"a", b"b"), 0) for index in range(4))
    observations = evaluate_pairwise(adapter, items)
    metrics = aggregate_pairwise(observations)

    assert all(observation.correct is None for observation in observations)
    assert metrics["tie_count"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["chance_normalized_accuracy"] == 0.0


def test_byte_length_normalization_preserves_rescoring_evidence() -> None:
    adapter = ScriptedPairwiseAdapter({b"long": -4.0, b"x": -1.0})
    observation = evaluate_pairwise(
        adapter,
        (PairwiseItem("length", b"ctx", (b"long", b"x"), 0),),
    )[0]
    metrics = aggregate_pairwise((observation,))

    assert observation.candidate_byte_lengths == (4, 1)
    assert observation.margin == -3.0
    assert observation.correct is False
    assert observation.byte_length_normalized_margin == 0.0
    assert observation.byte_length_normalized_correct is None
    assert metrics["accuracy"] == 0.0
    assert metrics["byte_length_normalized_accuracy"] == 0.5
    assert metrics["byte_length_normalized_tie_count"] == 1


def test_pairwise_item_rejects_invalid_gold_index() -> None:
    with pytest.raises(ValueError):
        PairwiseItem("bad", b"", (b"a", b"b"), 2)


def test_pairwise_item_rejects_empty_candidate() -> None:
    with pytest.raises(ValueError):
        PairwiseItem("bad", b"ctx", (b"", b"b"), 0)
