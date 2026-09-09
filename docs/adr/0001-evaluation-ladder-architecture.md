# ADR-0001: Evaluation Ladder Architecture

- Status: Proposed
- Date: 2026-09-09

## Context

Fabryka users can train language models with different parameter counts, data budgets, tokenizers, architectures, and stopping points. The evaluation system therefore cannot assume that a fixed parameter count implies a fixed capability level.

The project needs an evaluation protocol that remains useful from very small models (around 8M parameters) through tens and hundreds of millions of parameters, while preserving enough raw evidence to improve the scoring methodology later without rerunning expensive inference.

A single benchmark score is insufficient for this purpose. At small scale, many downstream accuracy benchmarks sit at chance even while the underlying model is improving. At larger scale, easier tasks saturate and stop differentiating models.

## Decision

The benchmark ladder is capability-gated, not parameter-gated.

Parameter count, training tokens, estimated FLOPs, architecture, tokenizer, context length, dataset identity, and checkpoint step are recorded as experiment metadata. They may be used for analysis and recommendations, but they do not determine which capability a model is considered to have reached.

The evaluation pipeline is split into five layers:

1. **Model adapter** — exposes model-independent scoring and generation operations.
2. **Task runner** — presents benchmark items to the model and records per-example observations.
3. **Scoring** — converts observations into task metrics.
4. **Ladder engine** — classifies task regions as floor, informative, or saturated and determines whether deeper evaluation is useful.
5. **Reporting** — emits versioned aggregate results and protocol metadata.

The initial ladder is provisional and expected to evolve. A useful starting shape is:

- L0 — training sanity and language modelling signal: held-out loss / bits-per-byte.
- L1 — linguistic competence: pairwise grammatical discrimination and continuous log-probability margin.
- L2 — context mechanics: copy, induction, retrieval and related controlled probes.
- L3 — local comprehension: cloze and simple information extraction.
- L4 — knowledge and commonsense.
- L5 — multi-step reasoning.
- L6 — harder downstream benchmark suites.

These labels describe capability families, not fixed benchmark packages.

### Raw observations are first-class data

The runner MUST retain per-example observations internally rather than only final aggregate scores.

For a pairwise task, an observation should contain enough information to recompute alternative metrics, for example:

```json
{
  "task": "pl-language-l1",
  "task_version": "1.0",
  "example_id": "opaque-id",
  "candidate_a_logprob": -12.91,
  "candidate_b_logprob": -13.44,
  "gold_index": 0,
  "margin": 0.53,
  "correct": true
}
```

Public reporting rules for these observations are defined separately in ADR-0002.

### Model adapter contract

Benchmarks MUST NOT depend directly on a specific model implementation.

The public adapter interface should expose only the primitives required by tasks, such as:

```python
class ModelAdapter(Protocol):
    def sequence_logprob(self, data: bytes) -> float: ...

    def continuation_logprob(
        self,
        context: bytes,
        continuation: bytes,
    ) -> float: ...

    def generate(
        self,
        prompt: bytes,
        max_new_units: int,
    ) -> bytes: ...
```

Byte-oriented models are the initial target, but the contract must allow additional adapters later without changing benchmark logic.

### Result contract

Every evaluation result MUST record protocol provenance. At minimum:

```json
{
  "schema_version": 1,
  "model": {
    "parameters": 8160256,
    "architecture": "...",
    "tokenizer": "byte"
  },
  "training": {
    "tokens": 31334400,
    "checkpoint_step": 7250
  },
  "evaluation": {
    "benchmark_id": "pl-language-l1",
    "benchmark_version": "1.0",
    "runner_git_sha": "...",
    "seed": 42
  }
}
```

The schema is versioned. Changing the meaning of an existing field requires a schema version change.

### Checkpoint-aware evaluation

Where historical checkpoints exist, the ladder SHOULD evaluate the same model family at multiple training-token budgets. This permits capability curves over training progress, not only comparisons between final models.

A final checkpoint cannot be used to reconstruct historical benchmark results unless the relevant historical weights were retained.

### Floor, informative region, and saturation

Every benchmark or probe SHOULD eventually have empirically established operating regions:

- **floor** — the score is indistinguishable from a trivial/random baseline or otherwise fails to differentiate models;
- **informative** — the metric provides useful separation between models/checkpoints;
- **saturated** — additional model improvement produces little or no score separation.

These regions are empirical properties of the benchmark and evaluation protocol. They MUST NOT be hard-coded solely from parameter count.

## Consequences

### Positive

- The same evaluation system can compare unusually strong and weak models at the same parameter count.
- Continuous metrics can expose progress before discrete accuracy metrics move away from chance.
- Raw observations allow new scoring methods to be tested without rerunning model inference.
- The architecture supports byte-level Fabryka models and future model adapters.
- Checkpoint sweeps can reveal when capabilities emerge during training.

### Negative

- The ladder engine requires empirical calibration rather than a static lookup table.
- Retaining raw observations increases storage and handling requirements.
- Result comparability depends on strict benchmark and protocol versioning.

## Rejected alternatives

### Assign benchmarks directly by parameter count

Rejected because parameter count alone does not determine capability. Training data, token budget, architecture, tokenizer, optimization and stopping point materially affect model quality.

### Store only aggregate benchmark scores

Rejected because it prevents later analysis of continuous metrics, category-level behaviour, alternative aggregations and scoring bugs without rerunning inference.

### Use one universal composite score as the primary evaluation

Rejected because a single number hides floor and saturation effects and can mix incomparable capabilities or languages. Composite scores may be reported as secondary summaries, but the ladder remains capability-resolved.
