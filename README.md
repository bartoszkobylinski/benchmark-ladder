# benchmark-ladder

Capability-resolved evaluation infrastructure for small language models.

The repository separates public evaluation machinery from canonical hidden benchmark material. The public package contains adapter contracts, scoring, result schemas, verification and CLI orchestration. Hidden task content, answer keys and private per-example observations belong in access-controlled storage described by ADR-0002.

## Quick smoke test

After installing the package:

```bash
benchmark-ladder smoke --output-dir /tmp/benchmark-ladder-smoke
```

This runs a fully public synthetic fixture and writes `result.json` plus `observations.jsonl`. Existing outputs are not replaced unless `--force` is supplied explicitly.

## Pairwise evaluation

Real task bytes stay outside this repository. The CLI consumes a strict JSONL task file where all model-input bytes are canonical base64:

```json
{"example_id":"opaque-001","context_b64":"Y3R4","candidates_b64":["YQ==","Yg=="],"gold_index":0}
```

The loader rejects non-canonical base64 encodings. Task identity is computed from a canonical serialization of the decoded items, so source-file JSON whitespace, key order and line endings do not change the private `task_items_digest`. Release-local ids and item order are deliberately included in that identity: rotating either defines a distinct frozen release.

Concrete model support is supplied by an externally installed adapter factory with the form `module:factory`. The factory receives a JSON object and returns an object implementing the public `ModelAdapter` contract.

```bash
benchmark-ladder evaluate-pairwise \
  --adapter my_adapter.package:create_adapter \
  --adapter-config /private/model.json \
  --task-file /private/benchmark.jsonl \
  --result-out ./result.json \
  --observations-out /private/results/observations.jsonl \
  --benchmark-id example-pairwise \
  --benchmark-version 1 \
  --scorer-version pairwise-byte-v1 \
  --tie-epsilon-per-byte 1e-12 \
  --taxonomy-version 1 \
  --runner-git-sha "$GIT_SHA" \
  --release-commitment "sha256:<64-hex>" \
  --parameters 8160256 \
  --architecture transformer \
  --tokenizer byte \
  --training-tokens 31334400 \
  --checkpoint-step 7250
```

Per-example observations are intentionally written separately from the public aggregate result. The private observation file begins with a `run_metadata` record carrying the canonical `task_items_digest`, scorer configuration digest and release commitment. Observation records follow with scores, lengths, margins and decisions, but not task context or candidate bytes. Both output files are written atomically, and existing evidence is not overwritten unless `--force` is explicitly requested.

The provenance boundary intentionally keeps the unsalted hidden-item content digest private. The public result records `scorer_config_digest`, which identifies the semantic pairwise scorer configuration, and `release_commitment`, the externally supplied commitment for the frozen hidden evaluation release described by ADR-0002. The trusted evaluator reconciles the public commitment with the private frozen release manifest, and that manifest binds the private `task_items_digest`. This avoids publishing a guessing oracle for low-entropy hidden task material while preserving an auditable chain to the exact decoded items that were evaluated.

The current pairwise normalization path is byte-only until the adapter contract gains boundary-aware token/model-unit counting.

External adapter factories execute inside the current trusted evaluator process. This CLI is not a sandbox for arbitrary submitted code. If untrusted model code must execute against hidden plaintext, the isolation requirements in ADR-0002 apply outside this process.

## Held-out bits-per-byte evaluation

The L0 likelihood path consumes held-out byte sequences using the same strict canonical-base64 boundary. Each JSONL record contains one independently scored sequence:

```json
{"example_id":"opaque-001","data_b64":"VGhpcyBpcyBoZWxkIG91dC4="}
```

Each record is passed unchanged to `ModelAdapter.sequence_logprob`. Record boundaries are part of benchmark semantics: splitting or joining records changes the start-of-sequence positions and can change the measured likelihood. A frozen release must therefore keep the same sequence boundaries and order across every checkpoint or model being compared, and its records must fit the supported scoring context of every adapter used for that comparison.

Corpus bits-per-byte is computed as total negative log-likelihood in nats divided by total raw input bytes and `ln(2)`. The harness sums NLL and bytes first; it does not average per-sequence BPB, which would overweight short records.

```bash
benchmark-ladder evaluate-lm \
  --adapter my_adapter.package:create_adapter \
  --adapter-config /private/model.json \
  --task-file /private/heldout.jsonl \
  --result-out ./result.json \
  --observations-out /private/results/observations.jsonl \
  --benchmark-id heldout-lm \
  --benchmark-version 1 \
  --scorer-version bpb-independent-sequence-v1 \
  --taxonomy-version 1 \
  --runner-git-sha "$GIT_SHA" \
  --release-commitment "sha256:<64-hex>" \
  --parameters 8160256 \
  --architecture transformer \
  --tokenizer byte \
  --training-tokens 31334400 \
  --checkpoint-step 7250
```

The public result reports corpus `bits_per_byte`, total `byte_count`, sequence `count` and total `negative_log_likelihood_nats`, together with the normal public provenance anchors. The unsalted canonical digest of the held-out bytes remains only in private run metadata, following the same release-commitment chain as pairwise evaluation. Per-sequence private observations retain byte counts and sequence log-probabilities but never the held-out bytes themselves.

## Architecture decisions

See `docs/adr/` for the evaluation-ladder architecture, hidden-benchmark boundary, and verification/reproducibility policy.

## Background reading

For 8M through tens or hundreds of millions of parameters, useful references include BabyLM Challenge and Findings of BabyLM; GPT-wee; Yam & Paek, “Teaching Tiny Minds”; Kaplan et al., “Scaling Laws for Neural Language Models”; Hoffmann et al., “Training Compute-Optimal Large Language Models”; Wei et al., “Emergent Abilities of Large Language Models”; and Schaeffer et al., “Are Emergent Abilities of Large Language Models a Mirage?”. The recurring design lesson is that continuous metrics are necessary when discrete benchmark accuracy remains at chance or saturates.
