import pytest

from benchmark_ladder.calibration import TaskState, ThresholdRule


def test_threshold_rule_boundaries() -> None:
    rule = ThresholdRule(floor_max=0.1, ceiling_min=0.9, min_observations=10)
    assert rule.classify(0.1, 10) is TaskState.FLOOR
    assert rule.classify(0.100001, 10) is TaskState.INFORMATIVE
    assert rule.classify(0.899999, 10) is TaskState.INFORMATIVE
    assert rule.classify(0.9, 10) is TaskState.NEAR_CEILING
    assert rule.classify(0.5, 9) is TaskState.INSUFFICIENT


def test_threshold_rule_rejects_invalid_order() -> None:
    with pytest.raises(ValueError):
        ThresholdRule(floor_max=0.9, ceiling_min=0.1, min_observations=10)
