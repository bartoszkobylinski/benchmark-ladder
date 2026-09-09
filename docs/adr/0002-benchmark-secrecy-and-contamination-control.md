# ADR-0002: Benchmark Secrecy and Contamination Control

- Status: Proposed
- Date: 2026-09-09

## Context

The public repository is intended to expose the evaluation framework, schemas, adapters, scoring logic, tests and documentation. It must not expose the hidden evaluation material used to measure submitted models.

Publishing benchmark items, answer keys, generators, templates, generation seeds or per-example outputs creates a direct contamination path: future training corpora can ingest the material from GitHub, package indexes, mirrors, caches or web crawls. Even without direct item disclosure, repeatedly exposing detailed benchmark feedback can enable adaptive overfitting to the hidden test distribution.

The project therefore needs a hard security and research-integrity boundary between public evaluation infrastructure and private benchmark material.

## Decision

The public repository contains the **evaluation harness**, not the hidden benchmark corpus.

Authoritative hidden benchmark material MUST live outside this public repository in a separate private repository or equivalent access-controlled storage.

The separation is architectural, not merely conventional.

### Public repository MAY contain

- model adapter interfaces and implementations;
- task interfaces;
- scoring primitives;
- aggregation logic;
- ladder logic;
- result schemas;
- CLI and orchestration code;
- synthetic toy fixtures that are not derived from hidden benchmark material;
- contract, unit, property, metamorphic and mutation tests;
- benchmark metadata that is safe to disclose, such as public identifier, version and high-level capability description;
- aggregate benchmark results that satisfy the disclosure policy below.

### Public repository MUST NOT contain

- real hidden benchmark examples;
- answer keys;
- hidden benchmark templates;
- private benchmark generators;
- generator parameters sufficient to reproduce the hidden distribution;
- secret random seeds;
- private split membership;
- raw private benchmark datasets;
- per-example model outputs from hidden evaluation;
- per-example log-probabilities from hidden evaluation;
- credentials or URLs that grant access to private benchmark material.

This prohibition includes source code, test fixtures, examples in documentation, issue bodies, pull-request comments, CI logs, caches and workflow artifacts.

## `.gitignore` is defense in depth, not the security boundary

The public repository SHOULD ignore common local paths used during trusted evaluation, but `.gitignore` MUST NOT be treated as protection for hidden benchmark material.

Ignored files can still be force-added, copied into artifacts, exposed in logs, or committed after an ignore rule changes. Files that were committed once remain recoverable from Git history even if they are deleted later.

Therefore, the canonical hidden benchmark corpus MUST be maintained in a separate private repository or access-controlled store. Local ignored paths exist only to reduce the chance of an accidental add during development.

## Private benchmark package

Hidden evaluation material should be consumed through a private package or plugin boundary. The public runner should require only a public task protocol, for example:

```python
class Task(Protocol):
    id: str
    version: str

    def iter_items(self): ...
    def score_observation(self, observation): ...
```

The public harness MUST be able to execute its full public test suite without access to the private package.

The hidden package may implement real tasks, load private datasets and define private generators. Its internal structure is not part of the public API.

## Trusted evaluation boundary

Real hidden evaluation MUST run only in a trusted environment.

Untrusted pull-request code MUST NOT receive credentials that can access private benchmark material. In particular:

- public fork PR workflows MUST run only public tests and toy fixtures;
- workflows triggered by untrusted code MUST NOT fetch the private benchmark repository;
- private benchmark credentials MUST NOT be available to arbitrary PR jobs;
- real benchmark inputs and per-example outputs MUST NOT be printed to job logs;
- real benchmark datasets and raw observations MUST NOT be uploaded to public artifacts or shared caches.

If private evaluation is later integrated into CI, it must run only against trusted commits under an explicitly reviewed workflow.

## Development set and final holdout

The private benchmark pool SHOULD distinguish at least two roles:

1. **development hidden set** — may be evaluated more frequently and may support iteration;
2. **final holdout** — used sparingly for final comparison or leaderboard decisions.

The final holdout MUST NOT be used for:

- model training;
- hyperparameter selection;
- early stopping;
- repeated interactive debugging;
- tuning benchmark-specific heuristics.

Repeated access to aggregate scores can itself leak information. The project MAY later introduce rate limits, evaluation quotas, rotating hidden sets or delayed leaderboard evaluation if adaptive overfitting becomes material.

## Synthetic hidden benchmarks

A synthetic benchmark is not automatically safe to publish.

If the generator, templates, distribution parameters and seeds allow a user to reproduce the evaluation distribution, models can be trained directly against that distribution. For hidden synthetic probes, the implementation details necessary to reproduce the benchmark MUST remain private.

The public repository may describe the capability at a high level, for example "contextual induction across increasing difficulty", without disclosing the exact hidden construction.

## Public disclosure

Public outputs SHOULD be aggregate and versioned. A public result may contain, for example:

```json
{
  "benchmark": "pl-language-l1",
  "version": "1.0",
  "score": 0.684,
  "region": "informative"
}
```

It MUST NOT reveal hidden examples, answer keys, per-example responses or raw per-example log-probabilities.

Category-level aggregates may be published only when the category granularity does not create a practical reconstruction or adaptive-overfitting risk.

## Benchmark commitments

Each frozen hidden benchmark release SHOULD have a cryptographic commitment calculated over a canonical package manifest or archive.

For example:

```text
pl-language-l1@1.0
sha256:8f26c3...
```

The commitment may be public even while the benchmark contents remain private. This provides evidence that a frozen evaluation set was not silently changed after model results were observed.

The canonicalization procedure used to compute the commitment MUST itself be versioned and documented.

## Incident response

If hidden benchmark material is accidentally disclosed publicly:

1. treat the affected benchmark version as compromised;
2. do not rely on deleting the file or rewriting Git history as sufficient remediation;
3. rotate the affected hidden set or benchmark version;
4. document the compromise and the first affected public timestamp;
5. mark subsequent results against the compromised version accordingly.

Once public, benchmark content must be assumed to have been copied.

## Consequences

### Positive

- Public development can proceed without making hidden evaluation data trainable from the repository.
- Public CI remains fork-safe and requires no benchmark secrets.
- Benchmark releases can be frozen and externally committed without disclosure.
- The design leaves room for stronger trusted-evaluation infrastructure later.

### Negative

- Contributors cannot reproduce the full hidden evaluation from the public repository alone.
- A second private storage location and access policy must be maintained.
- Debugging hidden tasks requires a trusted environment.
- Aggregate score access may still require governance to limit adaptive overfitting.

## Rejected alternatives

### Keep hidden files in the public repository but add them to `.gitignore`

Rejected. `.gitignore` only affects normal Git add behaviour and is not an access-control mechanism.

### Publish synthetic generators but hide only their seeds

Rejected as a general policy because reproducing the benchmark distribution may be sufficient to train against it even without the exact final examples.

### Publish all raw evaluation outputs for transparency

Rejected because per-example outputs can reveal hidden benchmark structure and enable adaptive overfitting. Reproducibility is instead provided through versioned protocols, provenance, public scoring code and cryptographic commitments.
