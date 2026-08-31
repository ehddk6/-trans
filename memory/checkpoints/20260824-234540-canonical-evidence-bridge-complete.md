# Checkpoint — canonical evidence bridge complete — 2026-08-24 20:10:27

## The story so far

The deep structural improvement is implemented in the uncommitted working tree. The canonical `process-title` path now preserves transcript ASR alternatives as `source_evidence` through Terra, Sol, and the deterministic gate. Critical dual-ASR meaning conflicts force `machine-uncertain`; agreement never promotes quality. Windows prompt-hash reproducibility, legacy schema parity, v2 input normalization/source refs, and lazy optional-ASR initialization were also fixed. The implementation prompt, frozen plan, result report, tests, and engineering evaluation are in the repository.

## Decided

- D-001: perform the structural improvement through verified completion, not as a superficial proposal.
- Evidence bridge is additive and raw-preserving; process schema is v3 and TranslationUnit schema is v2 with v1 read compatibility.
- Claims are limited to deterministic conflict-detection safety. Actual human translation-quality improvement remains `not-demonstrated`.

## Waiting on the user

- None for this goal. No deployment, external model call, human-final promotion, or commit was performed.

## Next first action

Read `docs/CANONICAL_EVIDENCE_BRIDGE_RESULT.md` for the result and reproduction commands. If the user next wants proof of actual subtitle improvement, open a separate sealed, blind human-listening evaluation goal against commit `0dddfe4`; do not reuse the synthetic engineering score as human-quality evidence.

## Tried

- Baseline full pytest had 3 failures: two CRLF/exact-byte prompt provenance failures and one eager optional-ASR construction failure. Root fixes were canonical text hashing/LF policy and lazy backend construction; expectations were not weakened.
- A blanket validation over every `*.manifest.json` incorrectly included the legacy targeted-retranslation manifest, which is not a `prompt_contract` manifest. Final verification explicitly validates the Terra and autonomous manifests supported by that validator.
- Final evidence: 453 tests pass, compileall passes, two prompt contracts pass, frozen 9-case report regenerates exactly, diff check passes, and zero-context rehearsal round 1 is clean.
