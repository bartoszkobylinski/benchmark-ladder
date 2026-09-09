import base64
import json
from pathlib import Path

import pytest

from benchmark_ladder.adapters import GenerationUnit
from benchmark_ladder.runner import PairwiseObservation
from benchmark_ladder.taskio import (
    load_pairwise_jsonl,
    pairwise_observation_json,
    write_pairwise_observations,
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_load_pairwise_jsonl_preserves_exact_bytes(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "example_id": "one",
                "context_b64": _b64(b"ctx\x00\xff"),
                "candidates_b64": [_b64(b"a\n"), _b64(b"b\x00")],
                "gold_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_pairwise_jsonl(task_path)

    assert len(items) == 1
    assert items[0].context == b"ctx\x00\xff"
    assert items[0].candidates == (b"a\n", b"b\x00")
    assert items[0].gold_index == 1


def test_load_pairwise_jsonl_rejects_duplicate_ids(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl"
    row = {
        "example_id": "duplicate",
        "context_b64": _b64(b"ctx"),
        "candidates_b64": [_b64(b"a"), _b64(b"b")],
        "gold_index": 0,
    }
    task_path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate example_id"):
        load_pairwise_jsonl(task_path)


def test_load_pairwise_jsonl_rejects_unknown_fields(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "example_id": "one",
                "context_b64": _b64(b"ctx"),
                "candidates_b64": [_b64(b"a"), _b64(b"b")],
                "gold_index": 0,
                "unexpected": "value",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys mismatch"):
        load_pairwise_jsonl(task_path)


def test_observation_json_contains_no_task_bytes(tmp_path: Path) -> None:
    observation = PairwiseObservation(
        example_id="opaque-id",
        scorer_version="pairwise-v1",
        normalization_unit=GenerationUnit.BYTE,
        candidate_logprobs=(-1.0, -2.0),
        candidate_byte_lengths=(6, 6),
        candidate_unit_counts=(6, 6),
        gold_index=0,
        margin=1.0,
        tie_epsilon=6e-12,
        correct=True,
        unit_normalized_margin=1.0 / 6.0,
        unit_normalized_tie_epsilon=1e-12,
        unit_normalized_correct=True,
    )

    encoded = pairwise_observation_json(observation)
    assert "context" not in encoded
    assert "candidate" in encoded
    assert "secret candidate" not in encoded

    output = tmp_path / "observations.jsonl"
    write_pairwise_observations(output, (observation,))
    assert output.read_text(encoding="utf-8") == encoded + "\n"
