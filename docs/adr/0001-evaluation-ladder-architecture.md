# ADR-0001: Evaluation Ladder Architecture

- Status: Proposed
- Date: 2026-09-09

## Context

Fabryka users can train language models with different parameter counts, data budgets, tokenizers, architectures, and stopping points. The evaluation system therefore cannot assume that a fixed parameter count implies a fixed capability level.

The project needs an evaluation protocol that remains useful from very small models (around 8M parameters) through tens and hundreds of millions of parameters, while preserving enough evidence to improve scoring methodology later without rerunning expensive inference.

A single benchmark score is insufficient for this purpose. At small scale, many downstream accuracy benchmarks sit at chance even while the underlying model is improving. At larger scale, easier tasks saturate and stop differentiating models.

The evaluation design must also avoid introducing a false total ordering of capabilities. Small models can have spiky profiles: a model may perform above chance on one downstream task while still failing a seemingly more basic probe.

## Decision

The benchmark ladder is **capability-resolved and budget-adaptive**, not parameter-gated and not a strict prerequisite chain.

Parameter count, training tokens, estimated FLOPs, architecture, tokenizer, context length, dataset identity, and checkpoint step are recorded as experiment metadata. They may be used for analysis, calibration and recommendations, but they do not determine which capability a model is considered to have reached.

The term **ladder** describes an empirical difficulty progression and reporting view. It does not mean that passing level N is a logical prerequisite for measuring level N+1.

The evaluation pipeline is split into five layers:

1. **Model adapter** — exposes model-independent scoring and generation operations.
2. **Task execution** — evaluates task items and records per-example observations inside the trusted evaluation boundary.
3. **Scoring** — converts observations into task metrics.
4. **Calibration and budget engine** — estimates whether more evaluation budget is useful for each task or capability family.
5. **Reporting** — emits versioned aggregate results and protocol metadata.

The initial capability families are provisional and expected to evolve. A useful starting shape is:

- L0 — training sanity and language-modelling signal;
- L1 — linguistic competence: pairwise grammatical discrimination and continuous log-probability margin;
- L2 — context mechanics: copy, induction, retrieval and related controlled probes;
- L3 — local comprehension: cloze and simple information extraction;
- L4 — knowledge and commonsense;
- L5 — multi-step reasoning;
- L6 — harder downstream benchmark suites.

These labels describe capability families, not fixed benchmark packages and not hard gates.

### Capability taxonomy is versioned

The mapping from tasks/benchmarks to capability families is a versioned artifact, not an implicit presentation rule.

A taxonomy version MUST identify at least:

- the capability-family definitions;
- the task/benchmark assignments to those families;
- any ordering or display metadata used to render a ladder/profile;
- the rationale or changelog for reassignments.

Moving a task from one family to another, splitting a family, merging families, or materially changing a family definition requires a new `taxonomy_version`.

Historical capability profiles MUST retain the taxonomy version under which they were produced. A later UI MAY re-render historical task-level results under a newer taxonomy, but that is a derived reinterpretation and MUST identify the target taxonomy explicitly rather than silently rewriting the original profile.

## Capability profile before hard gating

Every evaluation result SHOULD represent a capability profile rather than only a "highest level reached" value.

Where cost permits, each applicable capability family SHOULD receive at least a small probe budget. Additional budget can then be allocated to tasks where more observations are expected to reduce uncertainty or improve discrimination.

A lower-level result MUST NOT by itself prevent all measurement of higher-level capabilities.

If a task is not executed, the result MUST distinguish that state from a measured floor result. A skipped task record MUST include at least:

- execution status (`skipped`);
- skip reason;
- budget/calibration rule version;
- benchmark/task version;
- taxonomy version;
- reference pool identifier if calibration data influenced the decision.

Prefer a small sub-sampled probe over a hard skip when a low-cost probe can preserve useful evidence about non-monotone capability profiles.

## Calibration: task state versus operating region

The system distinguishes two concepts that MUST NOT be conflated.

### Per-model task state

A task state describes one model/checkpoint on one task, for example:

- consistent with baseline/floor;
- measurable/informative for this model;
- near ceiling;
- insufficient evidence.

The state is computed from that run's observations and an explicitly versioned rule.

### Benchmark operating region

A benchmark operating region describes how well a benchmark discriminates models across a defined calibration population.

Operating regions are therefore not global properties of a benchmark alone. They are properties of:

```text
benchmark@version
+ scorer_version
+ calibration_rule_version
+ reference_pool_id
```

The `reference_pool_id` identifies the frozen population of models/checkpoints used to estimate floor, discrimination and saturation behaviour. A result that reports an operating-region classification MUST record this identifier.

### Reference-pool lifecycle

A reference pool MUST be an immutable, explicitly created and versioned calibration artifact. It MUST identify at least:

- the model/checkpoint identifiers included in the pool;
- the benchmark and benchmark version evaluated for those checkpoints;
- the scorer version used for the reference observations;
- the calibration rule version for which the pool is intended;
- enough provenance to reproduce which observations belong to the pool.

A reference pool MUST NOT mean "all models observed so far" or grow implicitly as new runs arrive. Once a pool is frozen, adding, removing or replacing a model/checkpoint requires a new `reference_pool_id`.

Where practical, the frozen reference-pool manifest SHOULD receive the same kind of immutable commitment/provenance treatment as other evaluation-release artifacts. Calibration results computed against different reference pools are distinct results and MUST NOT be presented as directly comparable without an explicit bridging analysis.

### Reference-pool bootstrap

A new architecture, tokenizer family or training regime may initially have no valid reference pool. That is an expected state, not an error. Early runs in such a domain MUST report calibration as `unavailable` or `out-of-domain` while still emitting ordinary task-level measurements.

Promotion from uncalibrated runs to the first reference pool for a domain MUST be explicit. The applicable `calibration_rule_version` MUST predeclare, before the pool is frozen:

- the calibration domain it claims to cover;
- the minimum number of distinct model/checkpoint observations required;
- coverage requirements across relevant variation such as training budget, model scale or architecture settings;
- the statistical acceptance criteria for estimating floor/informative/saturation behaviour;
- who or what process is authorized to freeze the pool.

There is no repository-wide hard-coded minimum population because the required sample size depends on the calibration method and claim. A pool MUST NOT be minted until the predeclared minimum and coverage criteria of its calibration rule are satisfied.

The first pool for a domain MAY be built from runs that were themselves evaluated without an operating-region classification. Its manifest MUST record that bootstrap provenance explicitly. After the pool is frozen, subsequent runs may be calibrated against it only when they fall inside the declared calibration domain.

Calibration rules MUST NOT be hard-coded solely from parameter count.

The initial implementation does not require a specific statistical method. Item Response Theory, uncertainty-based budget allocation and related adaptive-testing methods remain compatible with this architecture and may be introduced by a later decision once empirical data exists.

## L0 comparability

Held-out loss and bits-per-byte are related but not interchangeable comparability metrics.

- **Bits-per-byte (BPB)** is the preferred tokenizer-independent language-modelling metric for byte-comparable corpora.
- **Token-level loss/perplexity** may be reported for training diagnostics, but cross-tokenizer comparisons MUST NOT treat it as directly comparable.
- Any held-out corpus used for comparative L0 evaluation MUST have a versioned identity and provenance. If it is hidden, the secrecy and commitment requirements of ADR-0002 apply to it as to any other hidden benchmark.

## Raw observations are first-class private data

The trusted evaluator MUST retain sufficient per-example observations internally to recompute alternative metrics without rerunning model inference, subject to the minimisation and retention controls in ADR-0002.

The following is a **synthetic schema illustration only; it is not a record from a real benchmark**:

```json
{
  "task": "toy-pairwise-task",
  "task_version": "0-test",
  "example_id": "release-random-id",
  "candidate_a_logprob": -12.91,
  "candidate_b_logprob": -13.44,
  "gold_index": 0,
  "margin": 0.53,
  "correct": true
}
```

Hidden-evaluation observations are private research data. They MUST NOT be published through the public repository, CI artifacts or logs. Identifier requirements, retention, access controls and rescoring rules are defined in ADR-0002.

## Model adapter contract

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

Byte-oriented models are the initial target. Additional adapters may be added later only with explicit, tested continuation-boundary semantics as required by ADR-0003.

## Result contract

Every evaluation result MUST record protocol provenance. At minimum, the result schema must be able to represent:

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
    "benchmark_id": "...",
    "benchmark_version": "...",
    "scorer_version": "...",
    "taxonomy_version": "...",
    "runner_git_sha": "...",
    "seed": 42,
    "calibration_rule_version": "...",
    "reference_pool_id": "..."
  },
  "execution": {
    "status": "measured",
    "skip_reason": null
  }
}
```

The example above is schema documentation, not a real benchmark result.

The schema is versioned. Changing the meaning of an existing field requires a schema version change.

A published score is always identified by at least `benchmark_version` and `scorer_version`. A published capability profile is additionally identified by `taxonomy_version`. If old private observations are re-scored under a new scorer, the new score MUST be published as a distinct result and MUST NOT silently replace the earlier published score.

## Checkpoint-aware evaluation

Where historical checkpoints exist, the ladder SHOULD evaluate the same model family at multiple training-token budgets. This permits capability curves over training progress, not only comparisons between final models.

A final checkpoint cannot be used to reconstruct historical benchmark results unless the relevant historical weights were retained.

## Consequences

### Positive

- The same evaluation system can compare unusually strong and weak models at the same parameter count.
- Non-monotone, spiky capability profiles remain observable.
- Continuous metrics can expose progress before discrete accuracy metrics move away from chance.
- Private raw observations allow new scoring methods to be tested without rerunning model inference.
- Calibration claims are tied to an explicit frozen reference population rather than an implicit global region.
- New model domains have an explicit bootstrap path instead of silently inheriting an invalid calibration.
- Taxonomy changes cannot silently rewrite historical capability profiles.
- Checkpoint sweeps can reveal capability changes during training.

### Negative

- The calibration and budget engine requires empirical data rather than a static lookup table.
- Results carry more provenance fields and explicit missingness states.
- Retaining private observations increases storage, access-control and retention requirements.
- Adaptive budget allocation is more complex than a fixed sequential benchmark list.
- Reference-pool maintenance creates an additional versioned artifact lifecycle.
- Capability-taxonomy maintenance creates another versioned presentation/interpretation artifact.

## Rejected alternatives

### Assign benchmarks directly by parameter count

Rejected because parameter count alone does not determine capability. Training data, token budget, architecture, tokenizer, optimization and stopping point materially affect model quality.

### Strict sequential gating

Rejected because it creates non-random missingness and assumes a total order of capabilities that small models need not follow. Budget allocation may still be adaptive, but a weak result on one capability does not erase evidence from all later capabilities.

### Store only aggregate benchmark scores

Rejected because it prevents later analysis of continuous metrics, category-level behaviour, alternative aggregations and scoring bugs without rerunning inference.

### Use one universal composite score as the primary evaluation

Rejected because a single number hides floor and saturation effects and can mix incomparable capabilities or languages. Composite scores may be reported as secondary summaries, but the primary result remains capability-resolved.
