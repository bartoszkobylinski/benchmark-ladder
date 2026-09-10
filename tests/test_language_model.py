import base64
import json
import math
from pathlib import Path
from typing import cast

import pytest

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit, ModelAdapter
from benchmark_ladder.cli import main
from benchmark_ladder.execution import LanguageModelEvaluationRequest, run_language_model_evaluation
from benchmark_ladder.language_model import (
    BitsPerByteScoringPolicy,
    LanguageModelItem,
    LanguageModelObservation,
    aggregate_language_model,
    evaluate_language_model,
)
from benchmark_ladder.results import EvaluationResult, ModelMetadata, TrainingMetadata
from benchmark_ladder.taskio import (
    canonical_language_model_items_digest,
    language_model_observation_json,
    load_language_model_jsonl,
    write_language_model_observations,
)
from tests.helpers import DeterministicByteAdapter

COMMITMENT = "sha256:" + ("d" * 64)
TASK_DIGEST = "sha256:" + ("e" * 64)
SCORER_DIGEST = "sha256:" + ("f" * 64)
DEFAULT_POLICY_DIGEST = "sha256:de110d548d6c700d63065db17afc3226abd7cdd0374120c99ba0bd2e842d7b96"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _observation(
    example_id: str,
    *,
    scorer_version: str = "bpb-v1",
    scorer_config_digest: str = SCORER_DIGEST,
    byte_count: int = 1,
    sequence_logprob_nats: float = -1.0,
    positive_logprob_clamped: bool = False,
) -> LanguageModelObservation:
    return LanguageModelObservation(
        example_id=example_id,
        scorer_version=scorer_version,
        scorer_config_digest=scorer_config_digest,
        byte_count=byte_count,
        sequence_logprob_nats=sequence_logprob_nats,
        positive_logprob_clamped=positive_logprob_clamped,
    )


class _InvalidSequenceAdapter(DeterministicByteAdapter):
    def __init__(self, score: float) -> None:
        self._score = score

    def sequence_logprob(self, data: bytes) -> float:
        del data
        return self._score


class _LengthAdapter:
    sequence_start_semantics = "scores-all-bytes-fixed-prior-v1"

    def sequence_logprob(self, data: bytes) -> float:
        return -float(len(data))

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return -float(len(continuation))

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        del prompt, config
        return b""

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        if unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        return len(data)


class _AlternateStartLengthAdapter(_LengthAdapter):
    sequence_start_semantics = "conditions-first-byte-on-bos-v1"


class _MissingStartAdapter:
    def sequence_logprob(self, data: bytes) -> float:
        return -float(len(data))

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return -float(len(continuation))

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        del prompt, config
        return b""

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        if unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        return len(data)


def test_bpb_policy_digest_and_weighted_aggregation_pin_semantics() -> None:
    policy = BitsPerByteScoringPolicy(version="bpb-v1")
    observations = (
        _observation("short", byte_count=1, sequence_logprob_nats=-0.0),
        _observation("long", byte_count=9, sequence_logprob_nats=-(18.0 * math.log(2.0))),
    )

    metrics = aggregate_language_model(observations)

    assert policy.config_digest == DEFAULT_POLICY_DIGEST
    assert metrics["count"] == 2
    assert metrics["byte_count"] == 10
    assert metrics["negative_log_likelihood_nats"] == pytest.approx(18.0 * math.log(2.0))
    assert metrics["bits_per_byte"] == pytest.approx(1.8)


def test_bpb_policy_digest_changes_with_numerical_tolerance_not_version_label() -> None:
    first = BitsPerByteScoringPolicy(version="bpb-v1")
    renamed = BitsPerByteScoringPolicy(version="renamed")
    tighter = BitsPerByteScoringPolicy(version="bpb-v1", positive_logprob_tolerance_nats=1e-9)

    assert first.config_digest == renamed.config_digest
    assert first.config_digest != tighter.config_digest


def test_language_model_observation_rejects_impossible_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        _observation("one", sequence_logprob_nats=math.nan)
    with pytest.raises(ValueError, match="after tolerance handling"):
        _observation("one", sequence_logprob_nats=0.01)


def test_evaluate_language_model_clamps_tiny_positive_logprob_and_counts_it() -> None:
    item = LanguageModelItem("one", b"abc")
    policy = BitsPerByteScoringPolicy(version="bpb-v1")

    observations = evaluate_language_model(_InvalidSequenceAdapter(5e-7), (item,), policy)
    metrics = aggregate_language_model(observations)

    assert observations[0].sequence_logprob_nats == 0.0
    assert observations[0].positive_logprob_clamped is True
    assert metrics["positive_logprob_clamp_count"] == 1
    assert metrics["bits_per_byte"] == 0.0


def test_evaluate_language_model_rejects_invalid_logprob_with_item_index() -> None:
    item = LanguageModelItem("private-id", b"abc")
    policy = BitsPerByteScoringPolicy(version="bpb-v1")

    with pytest.raises(ValueError, match="item index 0") as positive_error:
        evaluate_language_model(_InvalidSequenceAdapter(2e-6), (item,), policy)
    assert "private-id" not in str(positive_error.value)

    with pytest.raises(ValueError, match="item index 0") as nonfinite_error:
        evaluate_language_model(_InvalidSequenceAdapter(float("-inf")), (item,), policy)
    assert "private-id" not in str(nonfinite_error.value)


def test_aggregate_language_model_rejects_mixed_scorer_versions_and_configs() -> None:
    with pytest.raises(ValueError, match="different scorer versions"):
        aggregate_language_model(
            (
                _observation("a", scorer_version="bpb-v1"),
                _observation("b", scorer_version="bpb-v2"),
            )
        )

    with pytest.raises(ValueError, match="different scorer configurations"):
        aggregate_language_model(
            (
                _observation("a", scorer_config_digest="sha256:" + ("a" * 64)),
                _observation("b", scorer_config_digest="sha256:" + ("b" * 64)),
            )
        )


def test_sequence_boundaries_are_preserved_and_can_change_bpb() -> None:
    adapter = DeterministicByteAdapter()
    policy = BitsPerByteScoringPolicy(version="bpb-v1")
    joined = (LanguageModelItem("joined", b"ab"),)
    split = (LanguageModelItem("a", b"a"), LanguageModelItem("b", b"b"))

    joined_metrics = aggregate_language_model(evaluate_language_model(adapter, joined, policy))
    split_metrics = aggregate_language_model(evaluate_language_model(adapter, split, policy))

    assert joined_metrics["byte_count"] == split_metrics["byte_count"] == 2
    assert joined_metrics["bits_per_byte"] != split_metrics["bits_per_byte"]
    assert canonical_language_model_items_digest(joined) != canonical_language_model_items_digest(
        split
    )


def test_load_language_model_jsonl_preserves_exact_bytes(tmp_path: Path) -> None:
    task_path = tmp_path / "heldout.jsonl"
    task_path.write_text(
        json.dumps({"example_id": "one", "data_b64": _b64(b"a\x00\xff\n")}) + "\n",
        encoding="utf-8",
    )

    items = load_language_model_jsonl(task_path)

    assert items == (LanguageModelItem("one", b"a\x00\xff\n"),)


def test_load_language_model_jsonl_rejects_ambiguous_or_empty_bytes(tmp_path: Path) -> None:
    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_text(
        json.dumps({"example_id": "one", "data_b64": "YR=="}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical base64"):
        load_language_model_jsonl(noncanonical)

    empty = tmp_path / "empty.jsonl"
    empty.write_text(
        json.dumps({"example_id": "one", "data_b64": ""}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty bytes"):
        load_language_model_jsonl(empty)


def test_language_model_digest_ignores_json_formatting_but_includes_order(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_row = {"example_id": "one", "data_b64": _b64(b"abc")}
    second_row = {"data_b64": _b64(b"abc"), "example_id": "one"}
    first_path.write_text(json.dumps(first_row, separators=(",", ":")) + "\n", encoding="utf-8")
    second_path.write_text(json.dumps(second_row) + "\r\n", encoding="utf-8")

    first_items = load_language_model_jsonl(first_path)
    second_items = load_language_model_jsonl(second_path)

    assert canonical_language_model_items_digest(
        first_items
    ) == canonical_language_model_items_digest(second_items)

    other = LanguageModelItem("two", b"def")
    assert canonical_language_model_items_digest((*first_items, other)) != (
        canonical_language_model_items_digest((other, *first_items))
    )


def test_language_model_private_observations_exclude_sequence_bytes(tmp_path: Path) -> None:
    observation = _observation(
        "opaque",
        byte_count=6,
        sequence_logprob_nats=-3.0,
        positive_logprob_clamped=True,
    )
    encoded = language_model_observation_json(observation)

    assert "data_b64" not in encoded
    assert "secret" not in encoded

    output = tmp_path / "observations.jsonl"
    write_language_model_observations(
        output,
        (observation,),
        task_items_digest=TASK_DIGEST,
        scorer_config_digest=SCORER_DIGEST,
        release_commitment=COMMITMENT,
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["record_type"] == "run_metadata"
    assert records[0]["task_items_digest"] == TASK_DIGEST
    assert records[1]["record_type"] == "observation"
    assert records[1]["scorer_config_digest"] == SCORER_DIGEST
    assert records[1]["positive_logprob_clamped"] is True
    assert records[1]["byte_count"] == 6
    assert records[1]["bits_per_byte"] == pytest.approx(3.0 / (6.0 * math.log(2.0)))


def _request(adapter: ModelAdapter) -> LanguageModelEvaluationRequest:
    return LanguageModelEvaluationRequest(
        adapter=adapter,
        items=(LanguageModelItem("one", b"abc"), LanguageModelItem("two", b"de")),
        scoring_policy=BitsPerByteScoringPolicy(version="bpb-e2e-v1"),
        model=ModelMetadata(parameters=8_160_256, architecture="toy", tokenizer="byte"),
        training=TrainingMetadata(tokens=31_334_400, checkpoint_step=7250),
        benchmark_id="heldout-lm",
        benchmark_version="1",
        taxonomy_version="1",
        runner_git_sha="deadbeef",
        release_commitment=COMMITMENT,
        reference_pool_id="pool-1",
    )


def test_run_language_model_evaluation_binds_adapter_start_semantics() -> None:
    request = _request(_LengthAdapter())
    result, observations = run_language_model_evaluation(request)

    assert result.metrics == aggregate_language_model(observations)
    assert result.metrics["bits_per_byte"] == pytest.approx(1.0 / math.log(2.0))
    assert result.evaluation.scorer_version == request.scoring_policy.version
    assert result.evaluation.scorer_config_digest == request.scoring_policy.config_digest
    assert result.evaluation.release_commitment == COMMITMENT
    assert result.model.sequence_start_semantics == _LengthAdapter.sequence_start_semantics
    assert EvaluationResult.from_dict(result.to_dict()) == result
    evaluation_payload = result.to_dict()["evaluation"]
    assert isinstance(evaluation_payload, dict)
    assert "task_items_digest" not in evaluation_payload


def test_different_adapter_start_semantics_produce_different_public_provenance() -> None:
    first, _ = run_language_model_evaluation(_request(_LengthAdapter()))
    second, _ = run_language_model_evaluation(_request(_AlternateStartLengthAdapter()))

    assert first.metrics == second.metrics
    assert first.model.sequence_start_semantics != second.model.sequence_start_semantics
    assert first.canonical_json() != second.canonical_json()


def test_language_model_evaluation_rejects_missing_start_semantics() -> None:
    request = _request(cast(ModelAdapter, _MissingStartAdapter()))

    with pytest.raises((AttributeError, ValueError), match="sequence_start_semantics"):
        run_language_model_evaluation(request)


def test_evaluate_lm_cli_writes_public_result_and_private_evidence(tmp_path: Path) -> None:
    task_path = tmp_path / "heldout.jsonl"
    rows = (
        {"example_id": "one", "data_b64": _b64(b"a")},
        {"example_id": "two", "data_b64": _b64(b"bcd")},
    )
    task_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    adapter_config = tmp_path / "adapter.json"
    adapter_config.write_text(json.dumps({"scores": {}}), encoding="utf-8")
    result_path = tmp_path / "result.json"
    observations_path = tmp_path / "private" / "observations.jsonl"

    exit_code = main(
        [
            "evaluate-lm",
            "--adapter",
            "tests.fixture_adapter_plugin:create_adapter",
            "--adapter-config",
            str(adapter_config),
            "--task-file",
            str(task_path),
            "--result-out",
            str(result_path),
            "--observations-out",
            str(observations_path),
            "--benchmark-id",
            "heldout-lm",
            "--benchmark-version",
            "1",
            "--scorer-version",
            "bpb-cli-v1",
            "--taxonomy-version",
            "1",
            "--runner-git-sha",
            "deadbeef",
            "--release-commitment",
            COMMITMENT,
            "--parameters",
            "8160256",
            "--architecture",
            "toy-transformer",
            "--tokenizer",
            "byte",
            "--training-tokens",
            "31334400",
            "--checkpoint-step",
            "7250",
        ]
    )

    assert exit_code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["metrics"]["count"] == 2
    assert result["metrics"]["byte_count"] == 4
    assert result["metrics"]["bits_per_byte"] == pytest.approx(1.0 / math.log(2.0))
    assert result["metrics"]["positive_logprob_clamp_count"] == 0
    assert result["model"]["sequence_start_semantics"] == "fixture-independent-sequence-v1"
    assert result["evaluation"]["release_commitment"] == COMMITMENT
    assert result["evaluation"]["scorer_config_digest"].startswith("sha256:")
    assert "task_items_digest" not in result["evaluation"]

    private_records = [
        json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines()
    ]
    assert private_records[0]["task_items_digest"] == canonical_language_model_items_digest(
        load_language_model_jsonl(task_path)
    )
    assert len(private_records) == 3
    assert all("data_b64" not in record for record in private_records)
