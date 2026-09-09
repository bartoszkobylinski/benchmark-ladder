# ADR-0003: Verification, Reproducibility and CI

- Status: Proposed
- Date: 2026-09-09

## Context

Evaluation code can produce plausible-looking scores while being subtly wrong. Bugs in normalization, answer indexing, batching, masking, byte accounting, aggregation, continuation boundaries or benchmark routing can silently invalidate comparisons.

This project will publish benchmark infrastructure used to compare small language models across model sizes and training budgets. Correctness therefore requires stronger verification than ordinary application code.

The public repository must also remain safe for forks and pull requests without exposing private benchmark material defined in ADR-0002.

## Decision

Verification is layered. Public CI validates the evaluation machinery with toy data and deterministic fixtures. Hidden benchmark execution is a separate trusted operation.

The public test strategy consists of:

1. unit tests;
2. property-based tests;
3. metamorphic and analytic tests;
4. interface/contract tests;
5. deterministic golden integration tests;
6. differential validation against independent public references;
7. mutation testing for correctness-critical logic;
8. static checks and formatting/lint checks.

## 1. Unit tests

Scoring and transformation logic SHOULD be implemented as small pure functions wherever practical.

Examples include:

- bits-per-byte conversion;
- pairwise log-probability margin;
- accuracy;
- chance-normalized accuracy;
- category aggregation;
- task-state / floor / saturation classification;
- budget-rule decisions;
- result schema serialization.

Unit tests MUST cover boundary and degenerate cases, not only typical examples.

Examples:

```text
chance_normalized(accuracy=chance) == 0
chance_normalized(accuracy=1) == 1
accuracy < chance => normalized score < 0
```

and:

```text
correct_logprob = -2
incorrect_logprob = -5
margin = +3
```

## 2. Property-based tests

Property-based testing SHOULD be used for invariants and monotonic relationships that hold over broad input ranges.

Initial properties include:

- reordering examples does not change an aggregate score;
- splitting a deterministic evaluation into batches does not change the final score;
- permuting answer choices together with the gold index preserves correctness;
- serializing and deserializing a valid result preserves its semantic value;
- adding identical offsets to all candidate log-probabilities does not change pairwise ordering or margin differences where mathematically appropriate;
- aggregate counts equal the sum of category counts;
- increasing the gold candidate log-probability while holding competing candidates fixed MUST NOT decrease a monotone score;
- decreasing only an incorrect candidate's log-probability MUST NOT make a previously correct pairwise decision incorrect;
- a calibration classifier must behave monotonically around explicitly defined synthetic threshold cases where the rule is intended to be monotone.

A constant scorer can satisfy many invariance properties, so invariance tests alone are insufficient.

`hypothesis` is the preferred Python property-testing library unless implementation constraints require an alternative.

## 3. Metamorphic and analytic tests

Metamorphic tests MUST validate expected behaviour under controlled transformations when an exact expected output is otherwise inconvenient.

The test suite should include deterministic fake model adapters such as:

- **OracleAdapter** — always assigns the preferred score to the gold answer;
- **UniformAdapter** — assigns equal score to all choices;
- **WrongAdapter** — systematically prefers incorrect answers;
- **ScriptedAdapter** — returns configured scores chosen so the expected aggregate is computed analytically;
- **EchoAdapter** — useful for copy/retrieval protocol tests.

These adapters allow strong end-to-end assertions without a GPU or a real model.

Expected examples:

```text
OracleAdapter => accuracy 1.0
UniformAdapter => pairwise margin 0
WrongAdapter => below-chance score where applicable
ScriptedAdapter => analytically precomputed intermediate and aggregate values
```

For every scoring transform with a non-trivial continuous or normalized scale, tests MUST pin more than the endpoints. The suite MUST either:

- verify at least three analytically computed interior points spanning the useful range; or
- verify an analytic identity for the whole mapping (for example, affine dependence on the underlying correct-count where that is the intended definition).

This guards against monotone implementations that preserve 0, one midpoint and 1 while distorting the rest of the scale.

## 4. Contract tests

Every model adapter implementation MUST pass the same adapter contract suite.

Contract tests cover, at minimum:

- input/output types;
- empty and minimal inputs;
- deterministic behaviour when the adapter declares deterministic mode;
- continuation boundary handling;
- byte/unit accounting;
- batch versus single-item equivalence where supported;
- generation length semantics;
- error behaviour for unsupported operations.

### Continuation decomposition identity

For byte-oriented adapters, the following identity MUST hold within a documented numerical tolerance:

```text
sequence_logprob(context + continuation)
==
sequence_logprob(context)
+ continuation_logprob(context, continuation)
```

This identity is valid evidence only when the two sides are computed through implementation-independent paths. `continuation_logprob` MUST NOT satisfy the contract merely by returning:

```python
sequence_logprob(context + continuation) - sequence_logprob(context)
```

The intended byte-adapter implementation path is a masked/offset conditional-likelihood calculation over the joint sequence, independently checked against whole-sequence decomposition. The contract suite MUST be able to detect or forbid a tautological subtraction implementation rather than treating it as verification.

The contract suite MUST test this across empty, short, whitespace-sensitive and boundary-heavy byte sequences.

Future tokenized adapters MUST define explicit continuation-boundary semantics. If tokenizer retokenization across a text boundary makes the byte-level identity inapplicable, the adapter MUST provide an equivalent differential oracle that verifies its documented conditional-likelihood semantics and MUST include tests for both aligned and boundary-sensitive cases. Silent fallback to naive token slicing is not permitted.

### Task plugin contracts

Hidden task implementations are correctness-critical. Every private task plugin MUST pass an equivalent task contract suite inside the trusted environment.

The public repository may provide structurally safe toy implementations of that contract, but hidden examples, templates and distributions remain governed by ADR-0002.

## 5. Golden integration tests

The public repository MUST contain at least one tiny deterministic end-to-end fixture that is unrelated to hidden benchmark content.

The fixture runs through:

```text
fake model adapter
    -> toy task
    -> raw observations
    -> scoring
    -> task-state / calibration classification
    -> budget decision
    -> result serialization
```

Expected outputs are stored as golden fixtures.

A code change that alters a golden result MUST require an explicit fixture update. The pull request should explain whether the change is a bug fix, a schema change or an intentional metric change.

Floating-point values SHOULD use documented tolerances where byte-identical output is not portable. Stable structured fields SHOULD remain deterministic.

Golden fixtures MUST include cases immediately below, at and immediately above any synthetic classification threshold used by the test calibration rule.

### Determinism and execution-invariance check

Public CI MUST run the deterministic golden pipeline from clean process state more than once and compare the resulting structured outputs.

At least one rerun MUST intentionally vary a factor that the declared semantics say should not change the result. Initial public-CI perturbations SHOULD include one or more of:

- single-item versus batched execution;
- different worker/thread counts such as `OMP_NUM_THREADS=1` versus `4` where the code path supports it;
- different deterministic chunking/accumulation boundaries.

Stable fields must match exactly and floating-point fields must match within declared tolerances. If a runtime or backend is known to make a given perturbation numerically non-equivalent, that limitation must be explicit and the relevant invariant must not be claimed.

## 6. Differential validation against independent public references

Internal tests can agree with each other while sharing the same mistaken assumption. The project therefore MUST establish at least one independent end-to-end differential check using only public material before the first release that claims harness correctness.

The initial intended reference is the EleutherAI `lm-evaluation-harness`, using a public model and public task under a deliberately matched protocol. A concrete first target SHOULD use a small public causal LM such as `EleutherAI/pythia-70m` and a frozen public task slice whose scoring semantics can be reproduced on both sides.

A differential check has to be falsifiable. Its fixture/specification MUST record:

- reference harness version/commit;
- public model identifier and revision;
- public task identifier/version and exact frozen item slice;
- prompt/context construction;
- normalization and answer-scoring semantics;
- numerical tolerance and aggregate pass/fail criterion.

For deterministic log-probability comparisons under an intentionally identical protocol, the initial acceptance target SHOULD require per-example scores to agree within a documented tight tolerance (provisionally `1e-5` absolute unless backend precision requires a justified alternative) and discrete decisions/aggregate counts to agree exactly. If the protocols cannot be made identical, the comparison MUST be labeled diagnostic rather than passing/failing compatibility validation.

This check is not a substitute for unit or contract tests and does not need to run on every pull request if it is expensive. It SHOULD run before releases and after changes to adapter, scoring or benchmark-routing semantics that could affect compatibility.

## 7. Mutation testing

Mutation testing is required for correctness-critical logic, especially:

- scoring;
- normalization;
- aggregation;
- task-state/calibration classification;
- budget decisions;
- result transformations;
- pure adapter boundary/offset logic that can be tested without invoking expensive model inference.

Mutation testing SHOULD NOT initially mutate GPU kernels, third-party framework internals or code whose mutants are dominated by runtime cost rather than correctness value.

The initial target is:

- >= 95% mutation score for core scoring/normalization modules;
- >= 90% mutation score for calibration/budget decision logic.

These thresholds may begin as report-only while the first implementation is established, but once enforced they MUST NOT be lowered without an explicit decision and rationale.

Known equivalent-mutant classes and intentionally excluded numeric-tolerance mutants MUST be documented and excluded explicitly from the denominator rather than hidden by lowering the target.

`mutmut` is the preferred initial mutation runner unless practical limitations require an alternative.

## 8. Static checks

The public CI SHOULD run:

- `ruff` for linting and formatting checks;
- a Python type checker (`mypy` initially unless the codebase standardizes on another);
- `pytest` for unit, property, contract and integration tests.

Dependency and tool versions SHOULD be pinned or otherwise reproducible through the project package configuration and lockfile policy.

## Public GitHub Actions pipeline

Every pull request should run a fast public CI pipeline using only repository-visible fixtures:

```text
lint / format
    -> typecheck
    -> unit tests
    -> property tests
    -> contract tests
    -> golden integration tests
    -> execution-invariance rerun/diff
```

Mutation testing may run on protected-branch pushes and/or a scheduled workflow if its runtime makes it unsuitable for every pull request.

The CI pipeline MUST NOT require access to private benchmark data.

The public repository MUST remain fully testable by an arbitrary fork.

The workflow restrictions in ADR-0002, including the prohibition on exposing secrets through `pull_request_target`, chained untrusted workflows or self-hosted runners, apply to all CI definitions in this repository.

## Trusted hidden evaluation

Real benchmark evaluation is not part of untrusted public PR CI.

A trusted evaluation workflow may:

1. start from a reviewed commit;
2. obtain access to the private evaluator/package;
3. execute hidden tasks under the trusted boundary defined in ADR-0002;
4. retain permitted raw observations privately;
5. publish only allowed aggregate results.

Private task plugins MUST run their contract tests before producing publishable hidden benchmark results.

## Determinism and provenance

Every evaluation run MUST record enough metadata to explain how a score was produced, including where applicable:

- runner commit SHA;
- result schema version;
- benchmark identifier and version;
- scorer version;
- calibration/budget rule version;
- reference pool identifier when calibration data was used;
- model/checkpoint identifier;
- parameter count;
- training token count;
- tokenizer/model adapter identity;
- random seed;
- relevant numerical precision/runtime settings.

A rerun under the same declared deterministic configuration MUST produce the same structured result within documented floating-point tolerances. Any component that is intentionally stochastic MUST record the source of randomness and seed/control policy.

## Schema evolution tests

Versioning a schema is not sufficient unless readers and migrations are tested.

The test suite MUST verify, where applicable:

- a reader accepts supported schema versions;
- unsupported future versions fail explicitly rather than being silently misread;
- migrations preserve documented semantics;
- rescored or migrated results retain original benchmark/scorer provenance;
- changing the meaning of a field requires a schema-version change.

## Test data policy

Public test fixtures MUST be obviously synthetic or independently created for testing the harness. They MUST NOT be sampled, paraphrased or transformed from hidden benchmark material and MUST NOT reproduce hidden benchmark distributions or difficulty schedules closely enough to serve as matched training data.

The same rule applies to examples in documentation.

## Consequences

### Positive

- Core metric errors are likely to be caught before they affect published model comparisons.
- Adapter boundary bugs receive explicit non-tautological differential verification.
- Public contributors and forks can run the entire public verification suite without secrets or GPUs.
- Mutation tests verify that tests assert semantics rather than merely execute code.
- Contract tests make model and task adapters replaceable.
- Golden fixtures make accidental scoring and calibration changes visible in review.
- Independent public differential checks reduce the risk of an internally self-consistent but globally wrong harness.
- Execution-invariance reruns test more than repeated execution under one identical runtime configuration.

### Negative

- The test suite requires more initial engineering than a simple benchmark runner.
- Mutation testing and differential reference checks add runtime.
- Determinism requirements constrain some implementation choices.
- Hidden benchmark integration requires a second trusted test/evaluation path.
- Future non-byte adapters require explicit continuation-boundary semantics rather than assuming byte-model behaviour generalizes.
- Differential compatibility checks require maintaining a frozen external reference specification.

## Rejected alternatives

### Rely only on benchmark smoke tests against real models

Rejected because approximate smoke-test agreement is expensive, weakly diagnostic and may still pass when scoring logic is subtly wrong.

This does **not** reject reproducing a known public result under a matched protocol as an independent differential oracle; that is explicitly part of the verification strategy above.

### Run private benchmark evaluation for every public pull request

Rejected because untrusted code must not receive access to hidden benchmark material or credentials.

### Optimize only for line coverage

Rejected because high line coverage does not demonstrate that tests detect incorrect scoring behaviour. Mutation, property, metamorphic, analytic and differential testing provide stronger evidence for correctness-critical logic.
