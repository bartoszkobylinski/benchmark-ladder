"""Command-line entry point for trusted evaluation runs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark_ladder.adapters import DecodingConfig, GenerationUnit
from benchmark_ladder.execution import PairwiseEvaluationRequest, run_pairwise_evaluation
from benchmark_ladder.plugins import load_adapter
from benchmark_ladder.results import ModelMetadata, TrainingMetadata
from benchmark_ladder.runner import PairwiseItem, PairwiseScoringPolicy
from benchmark_ladder.taskio import (
    canonical_pairwise_items_digest,
    load_pairwise_jsonl,
    write_pairwise_observations,
    write_text,
)


class _SmokeAdapter:
    def __init__(self) -> None:
        self._scores = {
            b"a": -1.0,
            b"b": -2.0,
            b"c": -4.0,
            b"d": -1.0,
            b"e": -3.0,
            b"f": -3.0 + 5e-13,
        }

    def sequence_logprob(self, data: bytes) -> float:
        return -float(len(data))

    def continuation_logprob(self, context: bytes, continuation: bytes) -> float:
        del context
        return self._scores[continuation]

    def generate(self, prompt: bytes, config: DecodingConfig) -> bytes:
        if config.unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        if config.max_new_units == 0:
            return b""
        value = prompt[-1:] or b"x"
        return value * config.max_new_units

    def count_units(self, data: bytes, unit: GenerationUnit) -> int:
        if unit is not GenerationUnit.BYTE:
            raise NotImplementedError
        return len(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-ladder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate-pairwise",
        help="run a private pairwise task through an externally installed adapter",
    )
    evaluate.add_argument("--adapter", required=True, help="adapter factory as module:factory")
    evaluate.add_argument("--adapter-config", type=Path)
    evaluate.add_argument("--task-file", required=True, type=Path)
    evaluate.add_argument("--result-out", required=True, type=Path)
    evaluate.add_argument(
        "--observations-out",
        required=True,
        type=Path,
        help="private destination for per-example observations",
    )
    evaluate.add_argument("--benchmark-id", required=True)
    evaluate.add_argument("--benchmark-version", required=True)
    evaluate.add_argument("--scorer-version", required=True)
    evaluate.add_argument("--tie-epsilon-per-byte", required=True, type=float)
    evaluate.add_argument("--taxonomy-version", required=True)
    evaluate.add_argument("--runner-git-sha", required=True)
    evaluate.add_argument("--release-commitment", required=True)
    evaluate.add_argument("--reference-pool-id")
    evaluate.add_argument("--parameters", required=True, type=int)
    evaluate.add_argument("--architecture", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--training-tokens", required=True, type=int)
    evaluate.add_argument("--checkpoint-step", type=int)
    evaluate.add_argument(
        "--force",
        action="store_true",
        help="replace existing result/observation outputs after explicit operator choice",
    )

    smoke = subparsers.add_parser("smoke", help="run the public synthetic end-to-end fixture")
    smoke.add_argument("--output-dir", required=True, type=Path)
    smoke.add_argument("--force", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate-pairwise":
            return _run_evaluate_pairwise(args)
        if args.command == "smoke":
            return _run_smoke(args)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


def _run_evaluate_pairwise(args: argparse.Namespace) -> int:
    result_path = Path(args.result_out)
    observations_path = Path(args.observations_out)
    if result_path.resolve() == observations_path.resolve():
        raise ValueError("result-out and observations-out must be different paths")
    _ensure_outputs_available((observations_path, result_path), force=bool(args.force))

    adapter = load_adapter(args.adapter, args.adapter_config)
    items = load_pairwise_jsonl(Path(args.task_file))
    task_items_digest = canonical_pairwise_items_digest(items)
    policy = PairwiseScoringPolicy(
        version=args.scorer_version,
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=args.tie_epsilon_per_byte,
    )
    request = PairwiseEvaluationRequest(
        adapter=adapter,
        items=items,
        scoring_policy=policy,
        model=ModelMetadata(
            parameters=args.parameters,
            architecture=args.architecture,
            tokenizer=args.tokenizer,
        ),
        training=TrainingMetadata(
            tokens=args.training_tokens,
            checkpoint_step=args.checkpoint_step,
        ),
        benchmark_id=args.benchmark_id,
        benchmark_version=args.benchmark_version,
        taxonomy_version=args.taxonomy_version,
        runner_git_sha=args.runner_git_sha,
        release_commitment=args.release_commitment,
        reference_pool_id=args.reference_pool_id,
    )
    try:
        result, observations = run_pairwise_evaluation(request)
    except Exception as exc:
        raise RuntimeError(f"adapter evaluation failed ({type(exc).__name__})") from exc

    # Private observations are written first. A failure there must not leave a publishable result
    # that lacks the retained evidence needed for later rescoring and release reconciliation.
    write_pairwise_observations(
        observations_path,
        observations,
        task_items_digest=task_items_digest,
        scorer_config_digest=policy.config_digest,
        release_commitment=args.release_commitment,
        overwrite=bool(args.force),
    )
    write_text(result_path, result.canonical_json() + "\n", overwrite=bool(args.force))
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    observations_path = output_dir / "observations.jsonl"
    result_path = output_dir / "result.json"
    _ensure_outputs_available((observations_path, result_path), force=bool(args.force))

    policy = PairwiseScoringPolicy(
        version="public-smoke-pairwise-v1",
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=1e-12,
    )
    items = (
        PairwiseItem("correct", b"ctx", (b"a", b"b"), 0),
        PairwiseItem("wrong", b"ctx", (b"c", b"d"), 0),
        PairwiseItem("tie", b"ctx", (b"e", b"f"), 0),
    )
    task_items_digest = canonical_pairwise_items_digest(items)
    commitment = "sha256:" + hashlib.sha256(b"benchmark-ladder-public-smoke-v1").hexdigest()
    request = PairwiseEvaluationRequest(
        adapter=_SmokeAdapter(),
        items=items,
        scoring_policy=policy,
        model=ModelMetadata(parameters=1, architecture="public-smoke", tokenizer="byte"),
        training=TrainingMetadata(tokens=0, checkpoint_step=0),
        benchmark_id="public-smoke",
        benchmark_version="1",
        taxonomy_version="1",
        runner_git_sha="0" * 40,
        release_commitment=commitment,
    )
    result, observations = run_pairwise_evaluation(request)
    write_pairwise_observations(
        observations_path,
        observations,
        task_items_digest=task_items_digest,
        scorer_config_digest=policy.config_digest,
        release_commitment=commitment,
        overwrite=bool(args.force),
    )
    write_text(result_path, result.canonical_json() + "\n", overwrite=bool(args.force))
    return 0


def _ensure_outputs_available(paths: tuple[Path, ...], *, force: bool) -> None:
    if force:
        return
    if any(path.exists() for path in paths):
        raise FileExistsError("output path already exists; use --force to replace it")


if __name__ == "__main__":
    raise SystemExit(main())
