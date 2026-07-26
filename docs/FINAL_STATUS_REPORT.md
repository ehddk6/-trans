# Final status report — 2026-07-26

## Decision

`not-demonstrated`

The repository now has a reproducible, evidence-first implementation baseline,
but the supplied workspace has no adjudicated gold answers or completed blind
human review. It would therefore be false to claim that translation quality is
improved, best, optimal, or state of the art.

## What was audited

- Five supplied reference assets were preserved under `references/` and their
  inventory/checksums are recorded in `references/source-audit.json`.
- Vendor implementations remain preserved under `vendor/`; no vendor source,
  original SRT, audio, captures, previous translation, or existing output was
  edited.
- A Git repository was initialized and the audited baseline was committed as
  `6b4be8a` (`chore: preserve audited baseline`).
- The project test suite was run after the changes: **27 passed**.

## Current-system audit answers

| Question | Verified result | Boundary |
| --- | --- | --- |
| H1: does the priority score find real errors? | Not demonstrated. The bundled model has historical metrics, but no independently adjudicated labels or leave-one-title-out rerun in this workspace. | Do not report its score as quality or generalization. |
| H2: do P1/P2 cover high-risk errors? | Not demonstrated. P3/P4 random-audit labels are absent. | No whole-title recall claim. |
| H3: does ASR similarity hide semantic conflict? | Guard implemented: explicit reviewer-provided confirmed slot conflicts trigger escalation, while empty/unknown slots never become inferred claims. | Existing Whisper profiles remain one `whisper-family`, not independent evidence. |
| H4: are `avg_logprob`/`no_speech_prob` calibrated? | Not demonstrated. They are used only as review signals. | No fixed correctness threshold is claimed. |
| H5: do subtitle boundaries equal real scenes? | Not demonstrated. Alignment is stored as timing/boundary evidence, never as a semantic decision. | Forced-alignment failure is an uncertainty, not a meaning failure. |
| H6: does error memory generalize across titles? | Not demonstrated. | No cross-title automatic rule transfer is enabled. |

## Implemented and verified safeguards

- Semantic-frame and competing-hypothesis validation, including critical-slot
  escalation and source-family de-duplication.
- ASR input compatibility with an explicit default `source_family` and optional
  reviewer-authored `semantic_slots_json`. The code never derives slots from an
  ASR string; declared polarity/question/actor/etc. conflicts escalate instead
  of being resolved by majority voting.
- Reviewer artifact validators and schemas for `speaker-state`,
  `alignment-evidence`, and `backtranslation-check`. A named speaker needs
  identity provenance; invalid timing is rejected; `meaning-flip` and
  `meaning-addition` block a backtranslation check.
- Packaging validates supplied evidence artifacts and still refuses `final`
  without complete review, direct listening, evidence, semantic records, and
  evaluation evidence.

## Research application judgement

The project retains the following bounded uses: Whisper confidence values as
review signals, forced alignment as timing evidence, MQM as an error taxonomy,
and human/blind evaluation rather than automated metric output as the final
quality adjudicator. The supporting links are in `docs/research-findings.md`:
OpenAI Whisper, Montreal Forced Aligner, MQM, Freitag et al. (WMT21 metrics),
and the WMT evaluation task. None is converted into an unsupported automatic
semantic decision rule.

## Experiments and evaluation

Experiments A–F were **not run**. The required conditions are unavailable:

- no audio-directly-checked Japanese gold transcript/semantic annotations;
- no locked-test or blind human-review package with answers;
- no P3/P4 random-audit labels;
- no independent ASR family or direct human-listening record for the relevant
  scenes.

Accordingly, PR-AUC, recall/precision at 5/10/20% review budget, critical-error
recall, title-level generalization, calibration error, false-negative analysis,
and baseline-vs-improvement comparisons are not reported.

## Required next evidence

1. Adjudicate scenes into `evaluation/gold/` and keep `locked-test` unseen by
   development.
2. Collect P3/P4 random-audit labels for each title before estimating recall.
3. Add direct-listening provenance and, where needed, an actual independent
   ASR-family result.
4. Generate a blinded review package, run the requested A–F comparisons, and
   only then replace this decision with `improved`, `mixed`, or `regressed`.
