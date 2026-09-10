import base64
import json
from pathlib import Path

from benchmark_ladder.cli import main
from benchmark_ladder.taskio import canonical_pairwise_items_digest, load_pairwise_jsonl

COMMITMENT = "sha256:" + ("c" * 64)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _evaluate_args(
    *,
    task_path: Path,
    adapter_config: Path,
    result_path: Path,
    observations_path: Path,
) -> list[str]:
    return [
        "evaluate-pairwise",
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
        "private-test",
        "--benchmark-version",
        "1",
        "--scorer-version",
        "pairwise-cli-v1",
        "--tie-epsilon-per-byte",
        "1e-12",
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


def test_evaluate_pairwise_cli_writes_result_and_private_observations(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "example_id": "one",
                "context_b64": _b64(b"ctx"),
                "candidates_b64": [_b64(b"a"), _b64(b"b")],
                "gold_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_config = tmp_path / "adapter.json"
    adapter_config.write_text(
        json.dumps({"scores": {"a": -1.0, "b": -2.0}}),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    observations_path = tmp_path / "private" / "observations.jsonl"

    exit_code = main(
        _evaluate_args(
            task_path=task_path,
            adapter_config=adapter_config,
            result_path=result_path,
            observations_path=observations_path,
        )
    )

    assert exit_code == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["metrics"]["accuracy"] == 1.0
    assert result["evaluation"]["scorer_version"] == "pairwise-cli-v1"
    assert result["evaluation"]["scorer_config_digest"].startswith("sha256:")
    assert result["evaluation"]["task_items_digest"] == canonical_pairwise_items_digest(
        load_pairwise_jsonl(task_path)
    )
    assert result["evaluation"]["release_commitment"] == COMMITMENT

    observation_text = observations_path.read_text(encoding="utf-8")
    assert "one" in observation_text
    assert "context_b64" not in observation_text
    assert _b64(b"ctx") not in observation_text


def test_evaluate_pairwise_cli_refuses_existing_outputs_without_force(tmp_path: Path) -> None:
    task_path = tmp_path / "task.jsonl"
    task_path.write_text(
        json.dumps(
            {
                "example_id": "one",
                "context_b64": _b64(b"ctx"),
                "candidates_b64": [_b64(b"a"), _b64(b"b")],
                "gold_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_config = tmp_path / "adapter.json"
    adapter_config.write_text(json.dumps({"scores": {"a": -1.0, "b": -2.0}}), encoding="utf-8")
    result_path = tmp_path / "result.json"
    observations_path = tmp_path / "observations.jsonl"
    result_path.write_text("existing-result", encoding="utf-8")

    args = _evaluate_args(
        task_path=task_path,
        adapter_config=adapter_config,
        result_path=result_path,
        observations_path=observations_path,
    )
    exit_code = main(args)

    assert exit_code == 2
    assert result_path.read_text(encoding="utf-8") == "existing-result"
    assert not observations_path.exists()

    forced_exit_code = main([*args, "--force"])
    assert forced_exit_code == 0
    assert observations_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["metrics"]["accuracy"] == 1.0


def test_evaluate_pairwise_cli_rejects_same_output_path(tmp_path: Path) -> None:
    output = tmp_path / "same.json"
    exit_code = main(
        [
            "evaluate-pairwise",
            "--adapter",
            "tests.fixture_adapter_plugin:create_adapter",
            "--task-file",
            str(tmp_path / "missing.jsonl"),
            "--result-out",
            str(output),
            "--observations-out",
            str(output),
            "--benchmark-id",
            "x",
            "--benchmark-version",
            "1",
            "--scorer-version",
            "v1",
            "--tie-epsilon-per-byte",
            "1e-12",
            "--taxonomy-version",
            "1",
            "--runner-git-sha",
            "deadbeef",
            "--release-commitment",
            COMMITMENT,
            "--parameters",
            "1",
            "--architecture",
            "x",
            "--tokenizer",
            "byte",
            "--training-tokens",
            "0",
        ]
    )
    assert exit_code == 2


def test_smoke_cli_runs_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"

    exit_code = main(["smoke", "--output-dir", str(output_dir)])

    assert exit_code == 0
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["evaluation"]["benchmark_id"] == "public-smoke"
    assert result["evaluation"]["task_items_digest"].startswith("sha256:")
    assert result["evaluation"]["scorer_config_digest"].startswith("sha256:")
    assert result["metrics"]["count"] == 3
    assert result["metrics"]["accuracy"] == 0.5
    assert (output_dir / "observations.jsonl").read_text(encoding="utf-8").count("\n") == 3
