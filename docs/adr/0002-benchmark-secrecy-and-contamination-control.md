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

## Public repository MAY contain

- model adapter interfaces and evaluator-owned adapter implementations;
- task interfaces;
- scoring primitives that do not reveal hidden task construction;
- aggregation logic;
- calibration and reporting logic;
- result schemas;
- CLI and orchestration code;
- synthetic toy fixtures that do not reproduce hidden benchmark structure or distribution;
- contract, unit, property, metamorphic and mutation tests;
- benchmark metadata that is safe to disclose, such as public identifier, version and high-level capability description;
- aggregate benchmark results that satisfy the disclosure policy below.

## Public repository MUST NOT contain

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
- stable identifiers that permit matching hidden items across releases;
- credentials or URLs that grant access to private benchmark material.

This prohibition includes source code, test fixtures, examples in documentation, issue bodies, pull-request comments, CI logs, caches and workflow artifacts.

## `.gitignore` is defense in depth, not the security boundary

The public repository SHOULD ignore common local paths used during trusted evaluation, but `.gitignore` MUST NOT be treated as protection for hidden benchmark material.

Ignored files can still be force-added, copied into artifacts, exposed in logs, or committed after an ignore rule changes. Files that were committed once remain recoverable from Git history even if they are deleted later.

Therefore, the canonical hidden benchmark corpus MUST be maintained in a separate private repository or access-controlled store. Local ignored paths exist only to reduce the chance of an accidental add during development.

## Trusted evaluator owns hidden plaintext

Hidden items MUST be materialized only inside the trusted evaluation boundary.

The preferred architecture is that a private evaluator process owns:

- benchmark item loading;
- hidden task construction;
- answer keys;
- hidden scorer configuration;
- private per-example observations;
- release-local item identifiers.

The public harness may orchestrate the evaluation and expose adapter contracts, but it MUST NOT require hidden benchmark plaintext to pass through public CI, public logs or untrusted extension code.

Where practical, the private evaluator should communicate with model inference through a narrow evaluator-owned interface and return only private observations or allowed aggregates.

### Submitted model code is not automatically trusted

A hidden item eventually has to be scored by a model. If user-supplied executable code can inspect the item, that code can also log or exfiltrate it.

Therefore:

- evaluator-owned model implementations SHOULD load submitted weights when the model format permits this;
- arbitrary user-supplied executable code MUST NOT receive hidden benchmark plaintext together with unrestricted network or persistent-storage access;
- if arbitrary model code must be executed, it MUST run in an isolated environment with no network egress, no benchmark credentials, constrained filesystem access and ephemeral state;
- the trust decision for a model runtime MUST be explicit and auditable.

This requirement applies independently of whether the surrounding harness is public or private.

## Private benchmark package

Hidden evaluation material should be consumed through a private package, service or worker boundary.

The public repository may define an abstract task/evaluation contract, but the real hidden task implementation MUST remain private. Public task fixtures may exercise the same software interface, but they MUST NOT disclose hidden templates, difficulty schedules, statistical distributions or other construction details sufficient to create matched training data.

The high-level task semantics may be public when that is an intentional part of the benchmark design. For example, the project may disclose that a capability family measures contextual induction without disclosing how hidden instances are generated.

The public harness MUST be able to execute its full public test suite without access to the private package.

## Trusted evaluation boundary and CI footguns

Real hidden evaluation MUST run only in a trusted environment.

Untrusted pull-request code MUST NOT receive credentials that can access private benchmark material. In particular:

- public fork PR workflows MUST run only public tests and toy fixtures;
- workflows triggered by untrusted code MUST NOT fetch the private benchmark repository;
- `pull_request_target` MUST NOT be used to expose benchmark secrets to code from an untrusted pull request;
- `workflow_run` or equivalent chained workflows MUST NOT reintroduce secrets to artifacts or code produced by an untrusted run;
- self-hosted runners that can access private benchmark material MUST NOT execute arbitrary public-PR code;
- private benchmark credentials MUST NOT be available to arbitrary PR jobs;
- real benchmark inputs and per-example outputs MUST NOT be printed to job logs;
- real benchmark datasets and raw observations MUST NOT be uploaded to public artifacts or shared caches.

If private evaluation is integrated into CI, it MUST run only against reviewed trusted commits under an explicitly reviewed workflow.

## Development set and final holdout

The private benchmark pool MUST distinguish at least two roles before public repeated evaluation is enabled:

1. **development hidden set** — may be evaluated more frequently and may support iteration;
2. **final holdout** — used sparingly for final comparison or leaderboard decisions.

The final holdout MUST NOT be used for:

- model training;
- hyperparameter selection;
- early stopping;
- repeated interactive debugging;
- tuning benchmark-specific heuristics.

### Adaptive-overfitting policy

Item secrecy alone is not sufficient protection against adaptive overfitting.

Unrestricted repeated release of exact scores from a fixed final holdout is prohibited.

Before a public leaderboard or repeated final-holdout evaluation is enabled, the project MUST define a versioned feedback policy covering at least:

- per-user/model submission budget or equivalent query control;
- score precision/disclosure granularity;
- when a changed score is released;
- rotation or retirement policy for compromised or exhausted holdouts;
- whether thresholding, rounding, noise or reusable-holdout style mechanisms are used.

The development hidden set may expose more frequent aggregate feedback, but its role and resulting overfitting risk MUST be explicit. Results on the final holdout must not be interpreted as an ordinary development signal.

## Synthetic hidden benchmarks

A synthetic benchmark is not automatically safe to publish.

If the generator, templates, distribution parameters and seeds allow a user to reproduce the evaluation distribution, models can be trained directly against that distribution. For hidden synthetic probes, the implementation details necessary to reproduce the benchmark MUST remain private.

The public repository may describe the capability at a high level without disclosing the exact hidden construction.

## Public disclosure

Public outputs SHOULD be aggregate and versioned. A public result may contain, for example:

```json
{
  "benchmark": "example-benchmark",
  "version": "1.0",
  "scorer_version": "1",
  "score": 0.684,
  "calibration_rule_version": "1",
  "reference_pool_id": "pool-2026-09"
}
```

The example above is illustrative and does not correspond to a hidden benchmark result.

Public results MUST NOT reveal hidden examples, answer keys, per-example responses or raw per-example log-probabilities.

Category-level aggregates may be published only when the category granularity does not create a practical reconstruction or adaptive-overfitting risk.

## Benchmark commitments

Each frozen hidden evaluation release SHOULD have a cryptographic commitment before its first scored result is published.

The commitment MUST bind an **evaluation release manifest**, not only the item bytes. At minimum the committed manifest must identify:

- benchmark version;
- digest of canonical hidden content;
- scorer version;
- calibration/budget rule version where relevant;
- canonicalization procedure version;
- release metadata required to reproduce the interpretation of the score.

For low-entropy or synthetic benchmark material, the public commitment MUST NOT be a bare unsalted content hash. Use a secret random salt:

```text
commitment = H(salt || canonical_release_manifest)
```

The salt remains private while the benchmark is active and is released only when the commitment is intentionally revealed/audited.

The canonicalization procedure used to compute the commitment MUST itself be versioned and documented.

### Time anchoring

A commitment stored only in a mutable repository does not independently prove when it existed.

Before the first public result for a frozen final-holdout release, its commitment SHOULD be anchored in an external append-only or independently timestamped system, such as a transparency log or equivalent immutable timestamping mechanism.

A later reveal SHOULD provide enough material to verify both the commitment and its external anchor.

## Private raw observations

Private observations are sensitive research data, not ordinary debug output.

They MUST use release-local random item identifiers rather than content hashes or stable cross-version identifiers. Item identifiers MUST NOT provide a verification oracle for guessed benchmark content or reveal which items survived a rotation.

The private observation store MUST have a documented policy covering:

- minimisation: retain only fields needed for justified rescoring/research use;
- access control;
- encryption at rest where supported by the storage platform;
- access logging or equivalent auditability;
- retention period or explicit reason for indefinite retention;
- deletion/rotation procedure after compromise.

Where practical, observations from unrelated benchmark releases SHOULD be separable so compromise of one store does not automatically expose every historical release.

A score is a function of observations and a scorer version. Re-scoring old observations under a new scorer MUST create a new versioned result. It MUST NOT silently replace a previously published score.

## Incident response

If hidden benchmark material or the private observation store is accidentally disclosed:

1. identify the full blast radius, including every benchmark version exposed through the compromised store;
2. treat each affected active benchmark release as compromised;
3. do not rely on deleting a file or rewriting Git history as sufficient remediation;
4. rotate or retire affected hidden sets/releases;
5. invalidate or replace affected secrets, salts and credentials;
6. document the compromise and the first known public timestamp;
7. mark subsequent results against compromised versions accordingly.

Once public, benchmark content must be assumed to have been copied.

## Consequences

### Positive

- Public development can proceed without making hidden evaluation data trainable from the repository.
- Public CI remains fork-safe and requires no benchmark secrets.
- Hidden plaintext has an explicit trusted execution boundary.
- Commitment releases bind both hidden content and score interpretation.
- Repeated final-holdout feedback cannot silently become an unrestricted optimization oracle.

### Negative

- Contributors cannot reproduce the full hidden evaluation from the public repository alone.
- A second private storage location and access policy must be maintained.
- Debugging hidden tasks requires a trusted environment.
- Secure execution of arbitrary submitted model code may require sandboxing or a constrained evaluator-owned model format.
- Final-holdout access requires governance in addition to technical secrecy.

## Rejected alternatives

### Keep hidden files in the public repository but add them to `.gitignore`

Rejected. `.gitignore` only affects normal Git add behaviour and is not an access-control mechanism.

### Publish synthetic generators but hide only their seeds

Rejected as a general policy because reproducing the benchmark distribution may be sufficient to train against it even without the exact final examples.

### Pass hidden plaintext through arbitrary plugin code and rely on logging discipline

Rejected because accidental logging, tracebacks, crash dumps or malicious extension code can disclose benchmark material. The trusted evaluation boundary must constrain which code can observe plaintext and what that code can access.

### Publish unrestricted exact final-holdout scores

Rejected because repeated aggregate feedback itself can support adaptive overfitting even when individual items remain secret.

### Publish all raw evaluation outputs for transparency

Rejected because per-example outputs can reveal hidden benchmark structure and enable adaptive overfitting. Reproducibility is instead provided through versioned protocols, provenance, public scoring code where safe, and cryptographic commitments.
