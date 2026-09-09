"""Strict byte-safe task and observation I/O for trusted evaluation runs."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from benchmark_ladder.runner import PairwiseItem, PairwiseObservation

_PAIRWISE_KEYS = {"example_id", "context_b64", "candidates_b64", "gold_index"}


def load_pairwise_jsonl(path: Path) -> tuple[PairwiseItem, ...]:
    """Load binary pairwise task items from a strict JSONL format.

    Text fields carrying model input bytes are base64 encoded so task semantics do not depend on
    a text encoding, newline normalization, or JSON escaping behaviour.
    """

    items: list[PairwiseItem] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        data = _parse_object(raw_line, path=path, line_number=line_number)
        if set(data) != _PAIRWISE_KEYS:
            missing = sorted(_PAIRWISE_KEYS - set(data))
            extra = sorted(set(data) - _PAIRWISE_KEYS)
            raise ValueError(
                f"{path}:{line_number}: pairwise item keys mismatch; "
                f"missing={missing}, extra={extra}"
            )

        example_id = _required_str(data, "example_id", path, line_number)
        if example_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate example_id {example_id!r}")
        seen_ids.add(example_id)

        context = _decode_b64(
            _required_str(data, "context_b64", path, line_number),
            field="context_b64",
            path=path,
            line_number=line_number,
        )
        candidates_raw = data["candidates_b64"]
        if not isinstance(candidates_raw, list) or len(candidates_raw) != 2:
            raise TypeError(f"{path}:{line_number}: candidates_b64 must be a two-element list")
        candidates: list[bytes] = []
        for candidate_index, encoded in enumerate(candidates_raw):
            if not isinstance(encoded, str):
                raise TypeError(
                    f"{path}:{line_number}: candidates_b64[{candidate_index}] must be a string"
                )
            candidates.append(
                _decode_b64(
                    encoded,
                    field=f"candidates_b64[{candidate_index}]",
                    path=path,
                    line_number=line_number,
                )
            )

        gold_index = data["gold_index"]
        if isinstance(gold_index, bool) or not isinstance(gold_index, int):
            raise TypeError(f"{path}:{line_number}: gold_index must be an integer")

        items.append(
            PairwiseItem(
                example_id=example_id,
                context=context,
                candidates=(candidates[0], candidates[1]),
                gold_index=gold_index,
            )
        )

    if not items:
        raise ValueError(f"{path}: pairwise task file contains no items")
    return tuple(items)


def pairwise_observation_json(observation: PairwiseObservation) -> str:
    """Serialize one observation without task text or candidate bytes."""

    payload = {
        "example_id": observation.example_id,
        "scorer_version": observation.scorer_version,
        "normalization_unit": observation.normalization_unit.value,
        "candidate_logprobs": list(observation.candidate_logprobs),
        "candidate_byte_lengths": list(observation.candidate_byte_lengths),
        "candidate_unit_counts": list(observation.candidate_unit_counts),
        "gold_index": observation.gold_index,
        "margin": observation.margin,
        "tie_epsilon": observation.tie_epsilon,
        "correct": observation.correct,
        "unit_normalized_margin": observation.unit_normalized_margin,
        "unit_normalized_tie_epsilon": observation.unit_normalized_tie_epsilon,
        "unit_normalized_correct": observation.unit_normalized_correct,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_pairwise_observations(path: Path, observations: tuple[PairwiseObservation, ...]) -> None:
    """Write private per-example observations as deterministic JSONL."""

    if not observations:
        raise ValueError("cannot write an empty observation set")
    text = "\n".join(pairwise_observation_json(observation) for observation in observations) + "\n"
    _write_text(path, text)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories when necessary."""

    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_object(raw_line: str, *, path: Path, line_number: int) -> Mapping[str, object]:
    try:
        parsed: object = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{path}:{line_number}: each JSONL line must be an object")
    if any(not isinstance(key, str) for key in parsed):
        raise TypeError(f"{path}:{line_number}: object keys must be strings")
    return cast(Mapping[str, object], parsed)


def _required_str(data: Mapping[str, object], key: str, path: Path, line_number: int) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{path}:{line_number}: {key} must be a string")
    return value


def _decode_b64(value: str, *, field: str, path: Path, line_number: int) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: {field} is not valid base64") from exc
