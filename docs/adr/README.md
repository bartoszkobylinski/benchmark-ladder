# Architecture Decision Records

This directory records architectural and research-infrastructure decisions for `benchmark-ladder`.

ADRs are written before implementation when a decision affects result comparability, benchmark integrity, security boundaries, reproducibility, or the public/private interface.

## Status values

- **Proposed** — open for review.
- **Accepted** — adopted as the current project decision.
- **Superseded** — replaced by a later ADR.
- **Rejected** — considered and not adopted.

## Acceptance governance

An ADR becomes **Accepted** only after an explicit maintainer decision. Merge alone does not implicitly accept a document marked `Proposed`.

Before an ADR is relied upon by implementation work, one of the following must happen:

1. its status is changed to `Accepted` in the reviewed PR before merge; or
2. a follow-up acceptance commit changes only the status after the maintainer records the decision.

A PR that intentionally merges a `Proposed` ADR for further design discussion must say so explicitly and implementation must not treat that ADR as settled policy.

Accepted ADRs are append-only records of the decision at the time. Material changes should normally be made in a new ADR that supersedes the old one rather than silently rewriting history. Corrections made while an ADR is still `Proposed` remain part of the review process and may edit the document directly.

## Current ADRs

- [ADR-0001: Evaluation Ladder Architecture](0001-evaluation-ladder-architecture.md)
- [ADR-0002: Benchmark Secrecy and Contamination Control](0002-benchmark-secrecy-and-contamination-control.md)
- [ADR-0003: Verification, Reproducibility and CI](0003-verification-reproducibility-and-ci.md)
