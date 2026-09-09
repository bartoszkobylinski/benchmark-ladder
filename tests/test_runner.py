import pytest

from benchmark_ladder.adapters import GenerationUnit
from benchmark_ladder.runner import (
    PairwiseItem,
    PairwiseScoringPolicy,
    aggregate_pairwise,
    evaluate_pairwise,
)
from tests.helpers import ScriptedPairwiseAdapter

TEST_POLICY = PairwiseScoringPolicy(
    version="pairwise-test-v1",
    normalization_unit=GenerationUnit.BYTE,
    tie_epsilon_per_unit=1e-12,
)


def test_pairwise_runner_and_aggregation() -> None:
    adapter = ScriptedPairwiseAdapter({b"a": -1.0, b"b": -2.0, b"c": -5.0, b"d": -1.0})
    items = (
        PairwiseItem("one", b"ctx", (b"a", b"b"), 0),
        PairwiseItem("two", b"ctx", (b"c", b"d"), 0),
    )
    observations = evaluate_pairwise(adapter, items, TEST_POLICY)
    metrics = aggregate_pairwise(observations)

    assert observations[0].margin == 1.0
    assert observations[0].correct is True
    assert observations[1].margin == -4.0
    assert observations[1].correct is False
    assert observations[0].candidate_byte_lengths == (1, 1)
    assert observations[0].candidate_unit_counts == (1, 1)
    assert observations[0].scorer_version == TEST_POLICY.version
    assert observations[0].normalization_unit is GenerationUnit.BYTE
    assert metrics["count"] == 2
    assert metrics["tie_count"] == 0
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_margin"] == pytest.approx(-1.5)
    assert metrics["chance_normalized_accuracy"] == 0.0


def test_numerical_tie_band_scores_at_chance() -> None:
    adapter = ScriptedPairwiseAdapter({b"a": -3.0, b"b": -3.0 + 5e-13})
    items = tuple(PairwiseItem(f"i{index}", b"ctx", (b"a", b"b"), 0) for index in range(4))
    observations = evaluate_pairwise(adapter, items, TEST_POLICY)
    metrics = aggregate_pairwise(observations)

    assert all(observation.correct is None for observation in observations)
    assert all(observation.tie_epsilon == pytest.approx(1e-12) for observation in observations)
    assert metrics["tie_count"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["chance_normalized_accuracy"] == 0.0


def test_tie_band_scales_with_candidate_unit_count() -> None:
    adapter = ScriptedPairwiseAdapter({b"long": -4.0, b"wide": -4.0 + 3e-12})
    observation = evaluate_pairwise(
        adapter,
        (PairwiseItem("scaled-tie", b"ctx", (b"long", b"wide"), 0),),
        TEST_POLICY,
    )[0]

    assert observation.candidate_unit_counts == (4, 4)
    assert observation.tie_epsilon == pytest.approx(4e-12)
    assert observation.correct is None


def test_unit_normalization_preserves_rescoring_evidence() -> None:
    adapter = ScriptedPairwiseAdapter({b"long": -4.0, b"x": -1.0})
    observation = evaluate_pairwise(
        adapter,
        (PairwiseItem("length", b"ctx", (b"long", b"x"), 0),),
        TEST_POLICY,
    )[0]
    metrics = aggregate_pairwise((observation,))

    assert observation.candidate_byte_lengths == (4, 1)
    assert observation.candidate_unit_counts == (4, 1)
    assert observation.margin == -3.0
    assert observation.correct is False
    assert observation.unit_normalized_margin == 0.0
    assert observation.unit_normalized_correct is None
    assert metrics["accuracy"] == 0.0
    assert metrics["unit_normalized_accuracy"] == 0.5
    assert metrics["unit_normalized_tie_count"] == 1


def test_pairwise_item_rejects_invalid_gold_index() -> None:
    with pytest.raises(ValueError):
        PairwiseItem("bad", b"", (b"a", b"b"), 2)


def test_pairwise_item_rejects_empty_candidate() -> None:
    with pytest.raises(ValueError):
        PairwiseItem("bad", b"ctx", (b"", b"b"), 0)


def test_pairwise_policy_rejects_negative_tie_band() -> None:
    with pytest.raises(ValueError):
        PairwiseScoringPolicy(
            version="bad",
            normalization_unit=GenerationUnit.BYTE,
            tie_epsilon_per_unit=-1.0,
        )
