# Canonical quality path

## The current working tree now carries a canonical source-evidence bridge — verified 2026-08-24

- Claim label: `observed`.
- Claim: the working tree after the 2026-08-24 implementation builds `source_evidence` from transcript ASR metrics, stores it on `TranslationUnit`, passes it to Terra and Sol, and lets the deterministic audit block observed dual-family meaning conflicts.
- Primary evidence: `src/translation_forensics/evidence_bridge.py`; bridge calls in `integrated_pipeline.py`, `integrated_translation.py`, `process_title.py`, and `automated_quality.py`; `tests/integration/test_process_title.py`; `tests/unit/test_evidence_bridge.py`; `evaluation/engineering/canonical-evidence-bridge-v1.json`.
- Verification: full 453-test suite and compileall exited 0; frozen evaluation regenerated exactly with 6/6 hazards newly blocked, 0 missed, and 0/3 clean-control regressions.
- Limits: this supersedes the earlier “not wired” finding only for the current working tree. It proves deterministic conflict handling on hand-authored cases, not runtime prevalence, actual subtitle improvement, or human-equivalent quality.
- Verification date: 2026-08-24.

## The 2026-08-17 recovery v2 is not statically wired into `process-title` — verified 2026-08-24

- Claim label: `observed`.
- Claim: at commit `0dddfe4`, the repository documents `process-title` as the canonical command for new work, while `asr_fusion`, `graduated_recovery`, and `utterance_routing` are imported and called by `codex_quality.py`/`local_asr.py`, not by `process_title.py` or its Terra/Sol translation and deterministic-audit calls.
- Primary sources opened: `README.md:34-55`; `docs/INTEGRATED_PROCESS_TITLE.md:3-29`; imports and calls in `src/translation_forensics/process_title.py:38-42,495,644-648`; imports and calls in `src/translation_forensics/codex_quality.py:15-27,327,726`; `src/translation_forensics/local_asr.py:15,496,608,901,1409`.
- Sample: one static reachability audit of the checked-out HEAD plus two repository-owned canonical-path documents.
- Limits: this establishes the current static integration boundary, not runtime usage frequency and not any change in human translation quality. External wrappers and untracked code were not evaluated.
- Verification date: 2026-08-24.
