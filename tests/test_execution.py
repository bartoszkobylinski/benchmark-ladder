from benchmark_ladder.adapters import GenerationUnit
from benchmark_ladder.execution import PairwiseEvaluationRequest, run_pairwise_evaluation
from benchmark_ladder.results import ModelMetadata, TrainingMetadata
from benchmark_ladder.runner import PairwiseItem, PairwiseScoringPolicy
from tests.helpers import ScriptedPairwiseAdapter

COMMITMENT = "sha256:" + ("b" * 64)


def test_run_pairwise_evaluation_binds_metrics_and_public_provenance() -> None:
    policy = PairwiseScoringPolicy(
        version="pairwise-e2e-v1",
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=1e-12,
    )
    items = (PairwiseItem("one", b"ctx", (b"a", b"b"), 0),)
    request = PairwiseEvaluationRequest(
        adapter=ScriptedPairwiseAdapter({b"a": -1.0, b"b": -2.0}),
        items=items,
        scoring_policy=policy,
        model=ModelMetadata(parameters=8_160_256, architecture="toy", tokenizer="byte"),
        training=TrainingMetadata(tokens=31_334_400, checkpoint_step=7250),
        benchmark_id="private-pairwise",
        benchmark_version="1",
        taxonomy_version="1",
        runner_git_sha="deadbeef",
        release_commitment=COMMITMENT,
        reference_pool_id="pool-1",
    )

    result, observations = run_pairwise_evaluation(request)

    assert len(observations) == 1
    assert result.metrics["accuracy"] == 1.0
    assert result.evaluation.scorer_version == policy.version
    assert result.evaluation.scorer_config_digest == policy.config_digest
    assert result.evaluation.reference_pool_id == "pool-1"
    assert result.evaluation.release_commitment == COMMITMENT
    evaluation_payload = result.to_dict()["evaluation"]
    assert isinstance(evaluation_payload, dict)
    assert "task_items_digest" not in evaluation_payload


def _semantic_request(policy: PairwiseScoringPolicy) -> PairwiseEvaluationRequest:
    return PairwiseEvaluationRequest(
        adapter=ScriptedPairwiseAdapter({b"a": -1.0, b"b": -1.0 + 1e-9}),
        items=(PairwiseItem("one", b"ctx", (b"a", b"b"), 0),),
        scoring_policy=policy,
        model=ModelMetadata(parameters=1, architecture="toy", tokenizer="byte"),
        training=TrainingMetadata(tokens=0),
        benchmark_id="private-pairwise",
        benchmark_version="1",
        taxonomy_version="1",
        runner_git_sha="deadbeef",
        release_commitment=COMMITMENT,
    )


def test_same_version_with_different_scorer_semantics_changes_public_provenance() -> None:
    strict_policy = PairwiseScoringPolicy(
        version="same-label",
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=1e-12,
    )
    loose_policy = PairwiseScoringPolicy(
        version="same-label",
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=1e-6,
    )

    strict_result, _ = run_pairwise_evaluation(_semantic_request(strict_policy))
    loose_result, _ = run_pairwise_evaluation(_semantic_request(loose_policy))

    assert strict_result.evaluation.scorer_version == loose_result.evaluation.scorer_version
    assert (
        strict_result.evaluation.scorer_config_digest
        != loose_result.evaluation.scorer_config_digest
    )
    assert strict_result.metrics != loose_result.metrics
