# GPT-5.6 direct translation run — 2026-07-26

## Scope and model

- Semantic Korean translation was produced directly from the Japanese SRT with `gpt-5.6-terra`.
- The run did not send subtitle text to an external translation service and did not reuse legacy Korean subtitles as translation text.
- Local `faster-whisper-large-v3-turbo` was used only for explicitly logged corrupted-source windows. It is supporting evidence, not a semantic authority.
- No result in this run has human audio verification, human semantic approval, blinded evaluation, or final-package status.

## Completion ledger

| Title | Japanese blocks | Directly translated blocks | Deliverable state |
| --- | ---: | ---: | --- |
| ADN-622 | 651 | 651 | Two unreviewed intermediate SRT drafts |
| ADN-746 | 386 | 386 | Two unreviewed intermediate SRT drafts |
| IPX-998 | 832 | 832 | Two unreviewed intermediate SRT drafts |
| JUQ-439 | 890 | 887 | Three blocks deliberately blocked; no full SRT emitted |
| JUQ-778 | 1,955 | 1,955 | One unreviewed intermediate Korean SRT; not two independent variants |
| JUQ-811 | 827 | 827 | Two unreviewed intermediate SRT drafts |
| MDON-065 | 533 | 533 | Two unreviewed intermediate SRT drafts |
| SONE-785 | 858 | 858 | Two unreviewed intermediate SRT drafts |
| SSIS-400 | 577 | 577 | Two unreviewed intermediate SRT drafts; decision ledger mechanically reconstructed |
| SSIS-575 | 1,628 | 1,628 | Two unreviewed intermediate SRT drafts |
| SSIS-642 | 641 | 641 | Two unreviewed intermediate SRT drafts |
| SSIS-652 | 304 | 304 | Two unreviewed intermediate SRT drafts |
| SSIS-908 | 298 | 298 | Two unreviewed intermediate SRT drafts |

Total: 10,377 directly translated blocks out of 10,380.

## Revalidation performed

On 2026-07-26, the two-SRT outputs for the eleven titles that have both variants were revalidated against their Japanese structure SRTs. Every checked source-faithful and viewer-natural file had:

- identical block number, timecode, and ordering to its structure reference;
- zero empty-subtitle, Japanese-residue, or structural errors; and
- UTF-8/LF SRT formatting.

The validator emitted readability and repetition warnings. They are review signals, not evidence that the translations are semantically correct.

JUQ-778's one Korean SRT separately passed the same structural, empty-subtitle, and Japanese-residue checks. Its 1,955 checkpoint decisions are continuous, but a single output cannot stand in for the required source-faithful and viewer-natural pair.

## Explicitly blocked source corruption

`JUQ-439` blocks 373–375 remain `blocked` because their Japanese text is corrupted and local ASR candidates conflict or are low-confidence. A larger local context-window retry did not yield a reliable recovery. The decision ledger and blocked report retain the exact reasons. No blank, ellipsis-filled, or invented SRT was emitted for this title.

## Promotion boundary

All artifacts from this run remain under `workspaces/<title>/intermediate/`. `final_promotion_allowed` is false for these direct drafts. Structural checks do not replace source review, audio review, evaluation, or final packaging gates.
