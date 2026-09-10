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

The loader rejects non-canonical base64 encodings. Task identity is computed from a canonical serialization of the decoded items, so source-file JSON whitespace, key order and line endings do not change the recorded `task_items_digest`.

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

Per-example observations are intentionally written separately from the public aggregate result. The observations contain scores, lengths, margins and decisions, but not task context or candidate bytes. Both output files are written atomically, and existing evidence is not overwritten unless `--force` is explicitly requested.

The public result records three distinct provenance anchors:

- `task_items_digest`: SHA-256 over the canonical decoded pairwise items actually evaluated;
- `scorer_config_digest`: SHA-256 over the semantic pairwise scorer configuration, including normalization unit and tie epsilon;
- `release_commitment`: the externally supplied commitment for the frozen hidden evaluation release described by ADR-0002.

These values serve different purposes. `release_commitment` is not the task-file hash: the release commitment is expected to bind a frozen release manifest, while that manifest should include the recorded `task_items_digest`. This preserves the salted release-commitment design while making the exact evaluated item set falsifiable and auditable.

The current pairwise normalization path is byte-only until the adapter contract gains boundary-aware token/model-unit counting.

External adapter factories execute inside the current trusted evaluator process. This CLI is not a sandbox for arbitrary submitted code. If untrusted model code must execute against hidden plaintext, the isolation requirements in ADR-0002 apply outside this process.

## Architecture decisions

See `docs/adr/` for the evaluation-ladder architecture, hidden-benchmark boundary, and verification/reproducibility policy.

## Background reading

For 8M through tens or hundreds of millions of parameters, useful references include BabyLM Challenge and Findings of BabyLM; GPT-wee; Yam & Paek, “Teaching Tiny Minds”; Kaplan et al., “Scaling Laws for Neural Language Models”; Hoffmann et al., “Training Compute-Optimal Large Language Models”; Wei et al., “Emerent Abilities of Large Language Models”; and Schaeffer et al., “Are Emergent Abilities of Large Language Models a Mirage?”. The recurring design lesson is that continuous metrics are necessary when discrete benchmark accuracy remains at chance or saturates.
