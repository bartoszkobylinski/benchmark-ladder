"""Strict byte-safe task and observation I/O for trusted evaluation runs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from benchmark_ladder.language_model import LanguageModelItem, LanguageModelObservation
from benchmark_ladder.runner import PairwiseItem, PairwiseObservation

_PAIRWISE_KEYS = {"example_id", "context_b64", "candidates_b64", "gold_index"}
_LANGUAGE_MODEL_KEYS = {"example_id", "data_b64"}
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def load_pairwise_jsonl(path: Path) -> tuple[PairwiseItem, ...]:
    """Load binary pairwise task items from a strict JSONL format.

    Text fields carrying model input bytes are base64 encoded so task semantics do not depend on
    a text encoding, newline normalization, or JSON escaping behaviour. Base64 encodings must be
    canonical: multiple textual encodings of the same bytes are rejected.
    """

    items: list[PairwiseItem] = []
    seen_ids: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"unable to read task input ({type(exc).__name__})") from exc
    for line_number, raw_line in enumerate(text.split("\n"), start=1):
        if raw_line.endswith("\r"):
            raw_line = raw_line[:-1]
        if not raw_line.strip():
            continue
        data = _parse_object(raw_line, line_number=line_number)
        if set(data) != _PAIRWISE_KEYS:
            missing = sorted(_PAIRWISE_KEYS - set(data))
            extra = sorted(set(data) - _PAIRWISE_KEYS)
            raise ValueError(
                f"task input line {line_number}: pairwise item keys mismatch; "
                f"missing={missing}, extra={extra}"
            )

        example_id = _required_str(data, "example_id", line_number)
        if example_id in seen_ids:
            raise ValueError(f"task input line {line_number}: duplicate example_id")
        seen_ids.add(example_id)

        context = _decode_b64(
            _required_str(data, "context_b64", line_number),
            field="context_b64",
            line_number=line_number,
        )
        candidates_raw = data["candidates_b64"]
        if not isinstance(candidates_raw, list) or len(candidates_raw) != 2:
            raise TypeError(
                f"task input line {line_number}: candidates_b64 must be a two-element list"
            )
        candidates: list[bytes] = []
        for candidate_index, encoded in enumerate(candidates_raw):
            if not isinstance(encoded, str):
                raise TypeError(
                    f"task input line {line_number}: candidates_b64[{candidate_index}] "
                    "must be a string"
                )
            candidates.append(
                _decode_b64(
                    encoded,
                    field=f"candidates_b64[{candidate_index}]",
                    line_number=line_number,
                )
            )

        gold_index = data["gold_index"]
        if isinstance(gold_index, bool) or not isinstance(gold_index, int):
            raise TypeError(f"task input line {line_number}: gold_index must be an integer")

        items.append(
            PairwiseItem(
                example_id=example_id,
                context=context,
                candidates=(candidates[0], candidates[1]),
                gold_index=gold_index,
            )
        )

    if not items:
        raise ValueError("task input contains no items")
    return tuple(items)


def load_language_model_jsonl(path: Path) -> tuple[LanguageModelItem, ...]:
    """Load independently scored held-out byte sequences from strict JSONL."""

    items: list[LanguageModelItem] = []
    seen_ids: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"unable to read task input ({type(exc).__name__})") from exc
    for line_number, raw_line in enumerate(text.split("\n"), start=1):
        if raw_line.endswith("\r"):
            raw_line = raw_line[:-1]
        if not raw_line.strip():
            continue
        data = _parse_object(raw_line, line_number=line_number)
        if set(data) != _LANGUAGE_MODEL_KEYS:
            missing = sorted(_LANGUAGE_MODEL_KEYS - set(data))
            extra = sorted(set(data) - _LANGUAGE_MODEL_KEYS)
            raise ValueError(
                f"task input line {line_number}: language-model item keys mismatch; "
                f"missing={missing}, extra={extra}"
            )

        example_id = _required_str(data, "example_id", line_number)
        if example_id in seen_ids:
            raise ValueError(f"task input line {line_number}: duplicate example_id")
        seen_ids.add(example_id)

        item_bytes = _decode_b64(
            _required_str(data, "data_b64", line_number),
            field="data_b64",
            line_number=line_number,
        )
        if not item_bytes:
            raise ValueError(
                f"task input line {line_number}: data_b64 must decode to non-empty bytes"
            )
        items.append(LanguageModelItem(example_id=example_id, data=item_bytes))

    if not items:
        raise ValueError("task input contains no items")
    return tuple(items)


def canonical_pairwise_items_digest(items: tuple[PairwiseItem, ...]) -> str:
    """Hash canonical decoded task semantics for private release provenance.

    The digest belongs only in trusted/private records because a bare content hash can be a
    guessing oracle for low-entropy hidden tasks. JSON whitespace, key order and line endings do
    not change task identity, while decoded bytes, item order, gold labels and release-local ids
    do. Including release-local ids and order is deliberate: rotating ids or ordering defines a
    distinct frozen release identity rather than silently reusing the old digest.
    """

    if not items:
        raise ValueError("cannot digest an empty pairwise item set")
    digest = hashlib.sha256()
    for item in items:
        payload = {
            "candidates_b64": [
                base64.b64encode(item.candidates[0]).decode("ascii"),
                base64.b64encode(item.candidates[1]).decode("ascii"),
            ],
            "context_b64": base64.b64encode(item.context).decode("ascii"),
            "example_id": item.example_id,
            "gold_index": item.gold_index,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest.update(encoded)
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def canonical_language_model_items_digest(items: tuple[LanguageModelItem, ...]) -> str:
    """Hash canonical held-out sequence semantics for private release provenance.

    Item ids, decoded bytes and item order are all part of the frozen release identity. Record
    boundaries are intentionally preserved because each record is scored as an independent
    sequence and re-chunking can change the measured likelihood.
    """

    if not items:
        raise ValueError("cannot digest an empty language-model item set")
    digest = hashlib.sha256()
    for item in items:
        payload = {
            "data_b64": base64.b64encode(item.data).decode("ascii"),
            "example_id": item.example_id,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest.update(encoded)
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def private_run_metadata_json(
    *,
    task_items_digest: str,
    scorer_config_digest: str,
    release_commitment: str,
) -> str:
    """Serialize private run anchors that must not be emitted in the public result."""

    digests = {
        "task_items_digest": task_items_digest,
        "scorer_config_digest": scorer_config_digest,
        "release_commitment": release_commitment,
    }
    for name, value in digests.items():
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} must be sha256:<64 lowercase hex characters>")
    payload = {"record_type": "run_metadata", **digests}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def pairwise_observation_json(observation: PairwiseObservation) -> str:
    """Serialize one observation without task text or candidate bytes."""

    payload = {
        "record_type": "observation",
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


def language_model_observation_json(observation: LanguageModelObservation) -> str:
    """Serialize one likelihood observation without held-out sequence bytes."""

    payload = {
        "record_type": "observation",
        "example_id": observation.example_id,
        "scorer_version": observation.scorer_version,
        "scorer_config_digest": observation.scorer_config_digest,
        "byte_count": observation.byte_count,
        "sequence_logprob_nats": observation.sequence_logprob_nats,
        "positive_logprob_clamped": observation.positive_logprob_clamped,
        "negative_log_likelihood_nats": observation.negative_log_likelihood_nats,
        "bits_per_byte": observation.bits_per_byte,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_pairwise_observations(
    path: Path,
    observations: tuple[PairwiseObservation, ...],
    *,
    task_items_digest: str,
    scorer_config_digest: str,
    release_commitment: str,
    overwrite: bool = False,
) -> None:
    """Write private run metadata plus per-example observations as deterministic JSONL."""

    if not observations:
        raise ValueError("cannot write an empty observation set")
    records = [
        private_run_metadata_json(
            task_items_digest=task_items_digest,
            scorer_config_digest=scorer_config_digest,
            release_commitment=release_commitment,
        ),
        *(pairwise_observation_json(observation) for observation in observations),
    ]
    _write_text(path, "\n".join(records) + "\n", overwrite=overwrite)


def write_language_model_observations(
    path: Path,
    observations: tuple[LanguageModelObservation, ...],
    *,
    task_items_digest: str,
    scorer_config_digest: str,
    release_commitment: str,
    overwrite: bool = False,
) -> None:
    """Write private run metadata plus held-out likelihood observations as JSONL."""

    if not observations:
        raise ValueError("cannot write an empty observation set")
    observation_digests = {observation.scorer_config_digest for observation in observations}
    if observation_digests != {scorer_config_digest}:
        raise ValueError("observation scorer configuration does not match run metadata")
    records = [
        private_run_metadata_json(
            task_items_digest=task_items_digest,
            scorer_config_digest=scorer_config_digest,
            release_commitment=release_commitment,
        ),
        *(language_model_observation_json(observation) for observation in observations),
    ]
    _write_text(path, "\n".join(records) + "\n", overwrite=overwrite)


def write_text(path: Path, text: str, *, overwrite: bool = False) -> None:
    """Atomically write UTF-8 text, creating parent directories when necessary."""

    _write_text(path, text, overwrite=overwrite)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise FileExistsError("output path already exists; use explicit overwrite") from exc
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_object(raw_line: str, *, line_number: int) -> Mapping[str, object]:
    try:
        parsed: object = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"task input line {line_number}: invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"task input line {line_number}: each JSONL line must be an object")
    if any(not isinstance(key, str) for key in parsed):
        raise TypeError(f"task input line {line_number}: object keys must be strings")
    return cast(Mapping[str, object], parsed)


def _required_str(data: Mapping[str, object], key: str, line_number: int) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"task input line {line_number}: {key} must be a string")
    return value


def _decode_b64(value: str, *, field: str, line_number: int) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError(f"task input line {line_number}: {field} is not valid base64") from exc
    canonical = base64.b64encode(decoded).decode("ascii")
    if canonical != value:
        raise ValueError(f"task input line {line_number}: {field} is not canonical base64")
    return decoded
