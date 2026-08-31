# PRODUCT TRUTH — translation-forensics

Rule: every entry carries evidence, a date, and the date it was last checked against the code. External claims may be sourced only from the Implemented section. Code states are never blended: implemented, wired, operational, and verified are distinct.

## Implemented

- `process-title` schema v3 builds an additive, raw-preserving `TranslationUnit.source_evidence` record from transcript backend/Qwen alternatives and carries the same record through Terra input, translation decisions, Sol audit input, and deterministic quality audit. Evidence: `src/translation_forensics/evidence_bridge.py`, `integrated_pipeline.py`, `integrated_translation.py`, `process_title.py`; canonical-path integration test and full 453-test suite passed. Implemented and wired; checked 2026-08-24.
- The deterministic gate fail-closes observed dual-ASR meaning conflicts. The frozen synthetic contrast suite newly blocks 6/6 polarity/question/refusal-permission/stop-continue/direction hazards, misses 0, and changes 0/3 clean controls. Evidence: `tests/fixtures/source-evidence-contrast-cases.json`, `evaluation/engineering/canonical-evidence-bridge-v1.json`, exact regeneration assertion. Engineering-safety verified; checked 2026-08-24.
- Prompt provenance hashes canonical UTF-8 text with LF newlines, so CRLF/LF materialization is stable while content changes are rejected; optional conflict-ASR backends are initialized only when a rerun actually occurs. Evidence: prompt-contract, pilot-validation, and codex-quality regression tests; full suite passed. Implemented and verified; checked 2026-08-24.

## Not implemented

- No adjudicated sealed human gold or direct-listening before/after result demonstrates that the current code produces more accurate or more natural Korean subtitles than commit `0dddfe4`. `evaluation/gold/manifest.json` declares `empty-no-gold-answers`; current engineering report sets `human_quality_proven=false`. Checked 2026-08-24.
- No Korean-only independent model backcheck, full-title second-ASR pass, human-final promotion, or deployed subtitle replacement was performed by this change. Checked against the frozen plan and working-tree scope on 2026-08-24.

## Permanently excluded

<!-- Link a user-confirmed decision for each exclusion. -->
