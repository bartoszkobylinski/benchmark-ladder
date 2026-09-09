# ADR-0003: Verification, Reproducibility and CI

- Status: Proposed
- Date: 2026-09-09

## Context

Evaluation code can produce plausible-looking scores while being subtly wrong. Bugs in normalization, answer indexing, batching, masking, byte accounting, aggregation or benchmark routing can silently invalidate comparisons.

This project will publish benchmark infrastructure used to compare small language models across model sizes and training budgets. Correctness therefore requires stronger verification than ordinary application code.

The public repository must also remain safe for forks and pull requests without exposing private benchmark material defined in ADR-0002.

## Decision

Verification is layered. Public CI validates the evaluation machinery with toy data and deterministic fixtures. Hidden benchmark execution is a separate trusted operation.

The public test strategy consists of:

1. unit tests;
2. property-based tests;
3. metamorphic tests;
4. interface/contract tests;
5. deterministic golden integration tests;
6. mutation testing for correctness-critical pure logic;
7. static checks and formatting/lint checks.

## 1. Unit tests

Scoring and transformation logic SHOULD be implemented as small pure functions wherever practical.

Examples include:

- bits-per-byte conversion;
- pairwise log-probability margin;
- accuracy;
- chance-normalized accuracy;
- category aggregation;
- floor/informative/saturation classification;
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

Property-based testing SHOULD be used for invariants that hold over broad input ranges.

Initial properties include:

- reordering examples does not change an aggregate score;
- splitting a deterministic evaluation into batches does not change the final score;
- permuting answer choices together with the gold index preserves correctness;
- serializing and deserializing a valid result preserves its semantic value;
- adding identical offsets to all candidate log-probabilities does not change pairwise ordering or margin differences where mathematically appropriate;
- aggregate counts equal the sum of category counts.

`hypothesis` is the preferred Python property-testing library unless implementation constraints require an alternative.

## 3. Metamorphic tests

Metamorphic tests MUST validate expected behaviour under controlled transformations when an exact expected output is otherwise inconvenient.

The test suite should include deterministic fake model adapters such as:

- **OracleAdapter** — always assigns the preferred score to the gold answer;
- **UniformAdapter** — assigns equal score to all choices;
- **WrongAdapter** — systematically prefers incorrect answers;
- **ScriptedAdapter** — returns a configured sequence of scores;
- **EchoAdapter** — useful for copy/retrieval protocol tests.

These adapters allow strong end-to-end assertions without a GPU or a real model.

Expected examples:

```text
OracleAdapter => accuracy 1.0
UniformAdapter => pairwise margin 0
WrongAdapter => below-chance score where applicable
```

## 4. Contract tests

Every model adapter implementation MUST pass the same adapter contract suite.

Contract tests cover, at minimum:

- input/output types;
- empty and minimal inputs;
- deterministic behaviour when the adapter declares deterministic mode;
- continuation boundary handling;
- byte accounting;
- batch versus single-item equivalence where supported;
- generation length semantics;
- error behaviour for unsupported operations.

Task plugins SHOULD have an equivalent public task contract suite so hidden task implementations can be validated privately against the same interface.

## 5. Golden integration tests

The public repository MUST contain at least one tiny deterministic end-to-end fixture that is unrelated to hidden benchmark content.

The fixture runs through:

```text
fake model adapter
    -> toy task
    -> raw observations
    -> scoring
    -> ladder classification
    -> result serialization
```

Expected outputs are stored as golden fixtures.

A code change that alters a golden result MUST require an explicit fixture update. The pull request should explain whether the change is a bug fix, a schema change or an intentional metric change.

Floating-point values SHOULD use documented tolerances where byte-identical output is not portable. Stable structured fields SHOULD remain deterministic.

## 6. Mutation testing

Mutation testing is required for correctness-critical pure logic, especially:

- scoring;
- normalization;
- aggregation;
- ladder classification;
- result transformations.

Mutation testing SHOULD NOT initially cover GPU/model inference code, third-party framework glue or code whose mutants are dominated by runtime cost rather than correctness value.

The initial target is:

- >= 95% mutation score for core scoring/normalization modules;
- >= 90% mutation score for ladder decision logic.

These thresholds may begin as report-only while the first implementation is established, but once enforced they MUST NOT be lowered without an explicit decision and rationale.

`mutmut` is the preferred initial mutation runner unless practical limitations require an alternative.

## 7. Static checks

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
```

Mutation testing may run on protected-branch pushes and/or a scheduled workflow if its runtime makes it unsuitable for every pull request.

The CI pipeline MUST NOT require access to private benchmark data.

The public repository MUST remain fully testable by an arbitrary fork.

## Trusted hidden evaluation

Real benchmark evaluation is not part of untrusted public PR CI.

A trusted evaluation workflow may later:

1. start from a reviewed commit;
2. obtain access to the private benchmark package;
3. execute hidden tasks;
4. retain raw observations privately;
5. publish only allowed aggregate results.

The security constraints of ADR-0002 apply regardless of CI provider.

## Determinism and provenance

Every evaluation run MUST record enough metadata to explain how a score was produced, including where applicable:

- runner commit SHA;
- result schema version;
- benchmark identifier and version;
- model/checkpoint identifier;
- parameter count;
- training token count;
- tokenizer/model adapter identity;
- random seed;
- relevant numerical precision/runtime settings.

A rerun under the same declared deterministic configuration SHOULD produce the same structured result within documented floating-point tolerances.

## Test data policy

Public test fixtures MUST be obviously synthetic or independently created for testing the harness. They MUST NOT be sampled, paraphrased or transformed from hidden benchmark material.

The same rule applies to examples in documentation.

## Consequences

### Positive

- Core metric errors are likely to be caught before they affect published model comparisons.
- Public contributors and forks can run the entire public verification suite without secrets or GPUs.
- Mutation tests verify that tests assert semantics rather than merely execute code.
- Contract tests make model and task adapters replaceable.
- Golden fixtures make accidental scoring changes visible in review.

### Negative

- The test suite requires more initial engineering than a simple benchmark runner.
- Mutation testing adds CI runtime.
- Determinism requirements constrain some implementation choices.
- Hidden benchmark integration requires a second trusted test/evaluation path.

## Rejected alternatives

### Rely on benchmark smoke tests against real models

Rejected because they are expensive, weakly diagnostic and may still pass when scoring logic is subtly wrong.

### Run private benchmark evaluation for every public pull request

Rejected because untrusted code must not receive access to hidden benchmark material or credentials.

### Optimize only for line coverage

Rejected because high line coverage does not demonstrate that tests detect incorrect scoring behaviour. Mutation, property and metamorphic testing provide stronger evidence for correctness-critical logic.
