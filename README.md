# benchmark-ladder

Capability-resolved evaluation infrastructure for small language models.

The repository separates public evaluation machinery from canonical hidden benchmark material. The public package contains adapter contracts, scoring, result schemas, verification and CLI orchestration. Hidden task content, answer keys and private per-example observations belong in access-controlled storage described by ADR-0002.

## Quick smoke test

After installing the package:

```bash
benchmark-ladder smoke --output-dir /tmp/benchmark-ladder-smoke
```

This runs a fully public synthetic fixture and writes `result.json` plus `observations.jsonl`.

## Pairwise evaluation

Real task bytes stay outside this repository. The CLI consumes a strict JSONL task file where all model-input bytes are base64 encoded:

```json
{"example_id":"opaque-001","context_b64":"Y3R4","candidates_b64":["YQ==","Yg=="],"gold_index":0}
```

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

Per-example observations are intentionally written separately from the public aggregate result. The observations contain scores, lengths, margins and decisions, but not task context or candidate bytes. The current pairwise normalization path is byte-only until the adapter contract gains boundary-aware token/model-unit counting.

## Architecture decisions

See `docs/adr/` for the evaluation-ladder architecture, hidden-benchmark boundary, and verification/reproducibility policy.

## Background reading

For 8M through tens or hundreds of millions of parameters, useful references include BabyLM Challenge and Findings of BabyLM; GPT-wee; Yam & Paek, “Teaching Tiny Minds”; Kaplan et al., “Scaling Laws for Neural Language Models”; Hoffmann et al., “Training Compute-Optimal Large Language Models”; Wei et al., “Emergent Abilities of Large Language Models”; and Schaeffer et al., “Are Emergent Abilities of Large Language Models a Mirage?”. The recurring design lesson is that continuous metrics are necessary when discrete benchmark accuracy remains at chance or saturates.
