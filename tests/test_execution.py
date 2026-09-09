from benchmark_ladder.adapters import GenerationUnit
from benchmark_ladder.execution import PairwiseEvaluationRequest, run_pairwise_evaluation
from benchmark_ladder.results import ModelMetadata, TrainingMetadata
from benchmark_ladder.runner import PairwiseItem, PairwiseScoringPolicy
from tests.helpers import ScriptedPairwiseAdapter

COMMITMENT = "sha256:" + ("b" * 64)


def test_run_pairwise_evaluation_binds_metrics_and_provenance() -> None:
    policy = PairwiseScoringPolicy(
        version="pairwise-e2e-v1",
        normalization_unit=GenerationUnit.BYTE,
        tie_epsilon_per_unit=1e-12,
    )
    request = PairwiseEvaluationRequest(
        adapter=ScriptedPairwiseAdapter({b"a": -1.0, b"b": -2.0}),
        items=(PairwiseItem("one", b"ctx", (b"a", b"b"), 0),),
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
    assert result.evaluation.reference_pool_id == "pool-1"
    assert result.evaluation.release_commitment == COMMITMENT
