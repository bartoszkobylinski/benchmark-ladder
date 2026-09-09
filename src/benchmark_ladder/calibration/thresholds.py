"""Generic calibration-rule primitive.

This module intentionally does not choose project-wide empirical thresholds. Real threshold
values belong to a versioned calibration rule and reference pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class TaskState(StrEnum):
    FLOOR = "floor"
    INFORMATIVE = "informative"
    NEAR_CEILING = "near_ceiling"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ThresholdRule:
    floor_max: float
    ceiling_min: float
    min_observations: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.floor_max) or not math.isfinite(self.ceiling_min):
            raise ValueError("thresholds must be finite")
        if self.floor_max >= self.ceiling_min:
            raise ValueError("floor_max must be < ceiling_min")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be > 0")

    def classify(self, score: float, observation_count: int) -> TaskState:
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if observation_count < 0:
            raise ValueError("observation_count must be >= 0")
        if observation_count < self.min_observations:
            return TaskState.INSUFFICIENT
        if score <= self.floor_max:
            return TaskState.FLOOR
        if score >= self.ceiling_min:
            return TaskState.NEAR_CEILING
        return TaskState.INFORMATIVE
