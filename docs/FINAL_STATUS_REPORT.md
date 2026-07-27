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

## 2026-07-26 translation execution addendum

The user requested all 13 supplied titles be translated. The execution created
three complete but explicitly unverified external machine-translation drafts
(SSIS-642, SSIS-652, SSIS-908). Structural validation found zero errors, but
readability/repetition warnings remain and the two draft variants are identical.
They are not text-crosschecked or final.

Seven titles were blocked before an SRT could be emitted because of malformed
source or a non-functional local MT route. Three further titles have only a
small P2-reviewed subset. Three incomplete drafts containing ellipses for
untranslated blocks were withdrawn. Exact paths, counts, provider provenance,
and next actions are recorded in `docs/MACHINE_DRAFT_EXECUTION_2026-07-26.md`.

## 2026-07-26 closed-world validation addendum

The repository now also provides a separate `closed-world-validated` track.
It does not replace human listening, adjudicated gold, blind review, or
`final`. Across the 13 supplied titles it assigned every one of 10,380 blocks
to `accepted` or `abstained`: 870 were accepted by the conservative local
candidate and explicit-slot checks, and 9,510 were abstained. All 13 packages
passed structure, hash, decision-coverage, claim-boundary, and deterministic
rerun validation.

Human-reference equality remains `unidentifiable`, `100_percent_equal` remains
false, and `final_promotion_allowed` remains false for every title. Detailed
counts and package locations are in
`docs/CLOSED_WORLD_BATCH_REPORT_2026-07-26.md`.

## 2026-07-27 autonomous-release implementation addendum

The repository now contains an opt-in hybrid autonomous pipeline using
`gpt-5.6-terra` for structured decisions, `gpt-5.6-sol` for risk-only
critique/repair, and `gpt-4o-transcribe` for compatible risky audio clips.
It adds explicit network and cost preflight, local-ASR reuse or offline
generation, cache replay, checkpoint resume, request/response/environment
hashes, dual source/viewer SRTs, evidence graphs, uncertainty maps, formal v1
record schemas, and named regression gates.

All 13 workspaces and 10,380 blocks pass non-mutating input/evidence discovery.
Live model execution has not run because this environment currently has no
`OPENAI_API_KEY` and no optional OpenAI SDK; preflight correctly stops before
creating a release package. `autonomous-release` remains outside the human
`final` stage list and cannot be promoted through that gate. Operational usage
and the exact current boundary are documented in `docs/AUTONOMOUS_RELEASE.md`.
