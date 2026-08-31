# Verified `비디오` subtitle improvement — 2026-08-24

- Claim label: `observed`.
- Claim: the hash-guarded improvement run created `C:\Users\ehddk\OneDrive\비디오_개선본_20260824` without changing the original `비디오` SRTs. It copied 175 SRTs, changed 10 files and 305 blocks, and reduced Japanese-regex characters in the 111 playback files mapped from `Videos` from 2,558 to 0.
- Primary evidence: `C:\Users\ehddk\OneDrive\비디오_개선본_20260824\qa_report.json`, `change_ledger.jsonl`, `config/video-subtitle-overrides.json`, and `src/translation_forensics/video_subtitle_improvement.py`.
- Verification: independent inventory/residual/hash audit reported 111 common titles, 175 input/output SRTs, 165 unchanged files with 0 hash mismatches, and 2,558 → 0 residual characters. All 461 repository tests and compileall over `src`, `tools`, and `tests` passed.
- Regression controls: `ABF-196.srt` and `PRED-879.srt` remained byte-identical; no output hit for the blocked periodic viewer-boilerplate phrases.
- Existing defects preserved outside the requested change: `MOON-057.srt`, `PRED-488.srt`, and `SNOS-167.srt` have pre-existing SRT parse/timecode errors and were not modified.
- Limits: this is a text/timing-evidence improvement, not a source-audio listening evaluation. Zero Japanese-regex residue is not proof that every translated meaning is correct. The zero-residual claim applies to the mapped 111 playback files, not auxiliary Japanese source SRTs.
- Verification date: 2026-08-24.
