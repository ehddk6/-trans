# Inferred audio recovery v4.1 — 2026-07-27

## Result

The 60 `targeted-hold-marker` blocks from v3 were re-examined with a fresh
local `large-v3` Whisper transcription cut to each exact subtitle interval.
The v4.1 run recovered 29 blocks and deliberately retained 31 holds.  It does not
represent human review, machine-final approval, or a `final` release.

| Title | Inferred recovery | Still held |
| --- | ---: | ---: |
| ADN-622 | 0 | 2 |
| ADN-746 | 0 | 1 |
| IPX-998 | 1 | 0 |
| JUQ-439 | 1 | 2 |
| JUQ-778 | 1 | 1 |
| MDON-065 | 1 | 1 |
| SONE-785 | 4 | 0 |
| SSIS-400 | 2 | 1 |
| SSIS-575 | 3 | 0 |
| SSIS-642 | 2 | 10 |
| SSIS-652 | 0 | 1 |
| SSIS-908 | 14 | 12 |
| **Total** | **29** | **31** |

JUQ-811 had no v3 hold marker, so its v3 output remains unchanged and needs no
inferred-recovery package.

## Evidence and outputs

- Exact-block audio/ASR: `workspaces/<TITLE>/inferred-recovery-audio-v4/`
- Fixed model inputs: `workspaces/<TITLE>/inferred-recovery-context-v4.json`
- Applied v4 source/viewer SRT, ledger, report and structural check:
  `workspaces/<TITLE>/inferred-recovery-v4.1/`
- Auditable inference decisions:
  `retranslation-runs/2026-07-27/inferred-recovery-audio-inference-v1.json`

Every `infer-replace` has an `asr_same_family` basis and confidence of
`medium` or `low`.  Every retained marker has an explicit reason.  ASR
profiles were treated as alternate views of one Whisper family—not independent
votes.

## Guardrails preserved

Each generated `inferred-recovery-report.json` and decision ledger records:

- `inference: true`
- `human_reviewed: false`
- `machine_final_allowed: false`
- `human_final_allowed: false`
- `final_promotion_allowed: false`

The first external structured-prompt attempt was discarded because the native
stdin pipe converted Japanese text to question marks.  A corrected limited
second pass was used only as a conflict check: it vetoed JUQ-778 #978 because
its otherwise stable ASR contradicted the repeated greeting context.  This is
why v4.1 retains that block as a hold.  The applied decision file is the
explicit block-by-block evidence-based inference record.

## Verification

- All 12 v4 packages passed source/viewer structure validation against the
  original Japanese SRT: block count, number, timecode, ordering and encoding.
- `python -m pytest -q` passed (67 tests).
